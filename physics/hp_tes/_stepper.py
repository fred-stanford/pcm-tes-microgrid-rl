"""
hp_tes_rl_dropin/_stepper.py - Single-node PCM stepper backed by the v2
PCMBattery calibration, baked into ``_baked_calibration.json``.

This is a data-free re-implementation of the exact v2 ``_euler`` single-step
PCM physics.  It reproduces ``PCMBattery.charge`` / ``PCMBattery.discharge``
without importing any FlexLab data or the v2 ``best_calibration`` machinery --
everything it needs (the temperature grid, the effective-heat-capacity curves,
and the per-chemistry NTU-epsilon constants) is read once from the baked JSON.

numpy-only.  No project-specific imports at module load, so the stepper can be
exercised in isolation.

Chemistries
-----------
* ``"48"`` -- hot SAT tank (Sunamp Q9 hot)
* ``"11"`` -- cold ester tank (Sunamp Q9 cold)

Single-step physics (the v2 ``_euler`` step), given a node at ``T_C``, water
supply temp ``supply_C``, volumetric flow ``flow_lpm`` [L/min], timestep ``dt``
[s], and the tank's direction-selected constants (K1, K2, fexp, gate) and
effective-heat-capacity curve ``ceff_fn``::

    CP   = 4186.0
    mdot = max(flow_lpm, 0.05) / 60.0 * 0.997           # kg/s
    UA   = 1.0 / (K1 * max(flow_lpm, 0.05)**(-fexp) + K2)
    eps  = 1 - exp(-UA / (mdot * CP))
    Q    = eps * mdot * CP * (supply_C - T_C)            # W, + = heat INTO pcm
    if gate == "cold":  Q = min(Q, 0.0)
    elif gate == "hot": Q = max(Q, 0.0)
    T_new = T_C + dt * Q / max(ceff_fn(T_C), 1e4)

Direction parameterization (matches PCMBattery, field/shallow convention):
    charge    -> ceff_charge, K1=K1c, K2=K2c, fexp=fexp_c, gate=gate_c
    discharge -> ceff_charge, K1=K1d, K2=K2c, fexp=fexp_d, gate=gate_d
(field discharge is shallow, so ceff_charge is used for BOTH directions; the
deep-lab ceff_discharge curve is unused in this field system.)
"""

import json
import os

import numpy as np

# ── Load the baked calibration once ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_CAL_PATH = os.path.join(_HERE, "_baked_calibration.json")

with open(_CAL_PATH, "r") as _f:
    _CAL = json.load(_f)

CP = float(_CAL["CP"])                              # 4186.0 J/(kg·K)
_TGRID = np.asarray(_CAL["TGRID"], dtype=float)     # deg C grid (1051 pts)
_CHEM = _CAL["chem"]                                # keyed by "48" (hot), "11" (cold)

_RHO_LPM = 0.997                                    # kg/L, matches the v2 stepper

HOT_CHEM = "48"
COLD_CHEM = "11"


# ═══════════════════════════════════════════════════════════════════════════════
# SOC mapping (Celsius) -- numerically identical to hp_tes.tes_model.compute_soc.
# Inlined here so the stepper stays import-light (no hvac_components pull-in).
# ═══════════════════════════════════════════════════════════════════════════════

def _hot_soc_C(T_C):
    """Hot PCM SOC from node temperature [deg C]. Matches compute_hot_soc."""
    T_BREAK_LOW = 44.0
    T_BREAK_HIGH = 52.0
    R_BASE = 0.07575
    R_ACTIVE = 0.20076
    if T_C <= T_BREAK_LOW:
        return R_BASE * (T_C - T_BREAK_LOW)
    elif T_C <= T_BREAK_HIGH:
        return R_ACTIVE * (T_C - T_BREAK_LOW)
    else:
        soc_at_high = R_ACTIVE * (T_BREAK_HIGH - T_BREAK_LOW)
        return soc_at_high + R_BASE * (T_C - T_BREAK_HIGH)


def _cold_soc_C(T_C):
    """Cold PCM SOC from node temperature [deg C]. Matches compute_cold_soc."""
    T_BREAK_LOW = 11.0
    T_BREAK_HIGH = 13.0
    R_BASE = 0.08135
    R_ACTIVE = 0.58135
    if T_C <= T_BREAK_LOW:
        return 1.0 + R_BASE * (T_BREAK_LOW - T_C)
    elif T_C <= T_BREAK_HIGH:
        return 1.0 - R_ACTIVE * (T_C - T_BREAK_LOW)
    else:
        soc_at_high = 1.0 - R_ACTIVE * (T_BREAK_HIGH - T_BREAK_LOW)
        return soc_at_high - R_BASE * (T_C - T_BREAK_HIGH)


# ═══════════════════════════════════════════════════════════════════════════════
# PCMNode
# ═══════════════════════════════════════════════════════════════════════════════

class PCMNode:
    """
    Single-node PCM tank holding a bulk temperature ``T`` in deg C.

    Parameters
    ----------
    chem : str or int
        "48" for the hot SAT tank, "11" for the cold ester tank.
    T_init_C : float, optional
        Initial node temperature [deg C]. Defaults to 51 (hot) / 11 (cold).
    """

    def __init__(self, chem, T_init_C=None):
        chem = str(chem)
        if chem not in _CHEM:
            raise KeyError(
                f"Unknown PCM chemistry {chem!r}; expected one of {list(_CHEM)}"
            )
        self.chem = chem
        self.is_hot = (chem == HOT_CHEM)

        rec = _CHEM[chem]
        self.cap_scale = float(rec["cap_scale"])        # informational (already baked into ceff)

        # Charge-direction NTU-epsilon constants
        self._K1c = float(rec["K1c"])
        self._K2c = float(rec["K2c"])
        self._fexp_c = float(rec["fexp_c"])
        self._gate_c = rec["gate_c"]

        # Discharge-direction (field/shallow): K1=K1d, K2=K2c, fexp=fexp_d, gate=gate_d
        self._K1d = float(rec["K1d"])
        self._fexp_d = float(rec["fexp_d"])
        self._gate_d = rec["gate_d"]

        # Field convention: BOTH directions use the charge C_eff curve.
        self._ceff_charge = np.asarray(rec["ceff_charge"], dtype=float)

        if T_init_C is None:
            T_init_C = 51.0 if self.is_hot else 11.0
        self.T = float(T_init_C)

    # ── heat capacity ──
    def ceff(self, T_C=None):
        """Effective heat capacity [J/K] at ``T_C`` (defaults to current node T)."""
        if T_C is None:
            T_C = self.T
        return float(np.interp(T_C, _TGRID, self._ceff_charge))

    # ── direction constants ──
    def _params(self, mode):
        if mode == "charge":
            return self._K1c, self._K2c, self._fexp_c, self._gate_c
        elif mode == "discharge":
            # field/shallow discharge: K1d, K2c (NOT K2d), fexp_d, gate_d
            return self._K1d, self._K2c, self._fexp_d, self._gate_d
        raise ValueError(f"mode must be 'charge' or 'discharge', got {mode!r}")

    # ── single Euler step ──
    def step(self, supply_C, flow_lpm, dt, mode, apply=True):
        """
        Advance the node one substep against water at ``supply_C`` [deg C].

        Parameters
        ----------
        supply_C : float   -- water temperature reaching the tank [deg C]
        flow_lpm : float   -- volumetric water flow through the tank [L/min]
        dt : float         -- timestep [s]
        mode : str         -- "charge", "discharge", or "idle"
        apply : bool       -- when True (default) update ``self.T``; when False
                              compute and return Q without mutating the node
                              (used by the system loop to control update timing).

        Returns
        -------
        Q : float          -- heat rate INTO the pcm [W]. + heats the node,
                              - cools it. Always 0.0 for mode == "idle".
        """
        if mode == "idle":
            return 0.0

        K1, K2, fexp, gate = self._params(mode)
        f = max(float(flow_lpm), 0.05)
        mdot = f / 60.0 * _RHO_LPM
        UA = 1.0 / (K1 * f ** (-fexp) + K2)
        eps = 1.0 - np.exp(-UA / (mdot * CP))
        Q = eps * mdot * CP * (float(supply_C) - self.T)

        if gate == "cold":
            Q = min(Q, 0.0)
        elif gate == "hot":
            Q = max(Q, 0.0)

        if apply:
            self.T = self.T + dt * Q / max(self.ceff(self.T), 1e4)

        return float(Q)

    @property
    def soc(self):
        """State of charge (hot: warmer=higher, cold: colder=higher)."""
        return _hot_soc_C(self.T) if self.is_hot else _cold_soc_C(self.T)

    @property
    def T_K(self):
        """Node temperature in Kelvin."""
        return self.T + 273.15

    def __repr__(self):
        tag = "hot" if self.is_hot else "cold"
        return f"PCMNode(chem={self.chem!r}, {tag}, T={self.T:.2f}C, soc={self.soc:.3f})"
