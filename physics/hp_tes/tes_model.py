"""
hp_tes/tes_model.py - PCM thermal energy storage: SOC mapping plus thin
wrappers over the shared hvac-component-models physics library.

The PCM physics (effective heat capacity, liquid fraction, water-to-PCM UA,
water properties) lives in the hvac-component-models submodule and is shared
with other projects. The wrappers below convert this project's Kelvin and
hot-vs-cold conventions to the library's Celsius / preset-based API, so that
system.py and the calibration scripts need no changes.

SOC mapping (compute_hot_soc / compute_cold_soc / compute_soc) is a
project-specific reporting and controller signal, so it stays here.
"""

import os
import sys

# Make the hvac-component-models submodule importable IF present. The RL run path
# (run_fmu -> compute_soc) does NOT use it; only the legacy C_eff / UA / water-property
# helpers below do. So keep the import lazy/optional: the package imports and run_fmu
# works standalone (no submodule needed); calling those legacy helpers without the
# submodule raises a clear error.
_HVAC_LIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hvac-component-models",
)
if _HVAC_LIB not in sys.path:
    sys.path.insert(0, _HVAC_LIB)

try:
    import hvac_components as hc
except ImportError:  # submodule not present in the consumer's repo
    hc = None

from .parameters import K_OFFSET


# ═══════════════════════════════════════════════════════════════════════════════
# SOC mapping — project-specific, not in the shared library
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hot_soc(pcm_temp_K):
    """
    Hot PCM SOC matching FMU convention (empirical from FMU probe).

    Piecewise-linear with two rates:
      - Base rate (sensible only): 0.07575/°C outside active zone
      - Active rate (sensible + latent): 0.20076/°C in [44, 52]°C
      - SOC = 0 at 44°C (lower_sensible), SOC ≈ 1.0 at 49°C (solidus)
    """
    T_C = pcm_temp_K - K_OFFSET
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


def compute_cold_soc(pcm_temp_K):
    """
    Cold PCM SOC matching FMU convention (empirical from FMU probe).

    INVERTED: colder = higher SOC (more cold stored).
    Piecewise-linear with two rates:
      - Base rate: 0.08135/°C (SOC decreases as T increases)
      - Active rate: 0.58135/°C in [11, 13]°C
      - SOC = 1.0 at 11°C, SOC ≈ 0 at ~12.7°C
    """
    T_C = pcm_temp_K - K_OFFSET
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


def compute_soc(pcm_temp_K, tsol_K, tliq_K):
    """Dispatch to hot or cold SOC based on solidus temperature."""
    if tsol_K > 300:  # Hot PCM (solidus ~317K)
        return compute_hot_soc(pcm_temp_K)
    else:  # Cold PCM (solidus ~284K)
        return compute_cold_soc(pcm_temp_K)


# ═══════════════════════════════════════════════════════════════════════════════
# Wrappers over hvac-component-models PCM physics
# ═══════════════════════════════════════════════════════════════════════════════
# The library API is Celsius and preset-based; this project is Kelvin and uses
# a hot/cold flag. These wrappers bridge the two and keep the call sites in
# system.py and the calibration scripts unchanged.

def effective_heat_capacity(pcm_temp_K, tsol_K, tliq_K, mass_fraction=1.0,
                            is_charging=None):
    """
    Effective thermal capacity [J/K] of a PCM node (apparent heat capacity
    with smoothed latent boundaries). Delegates to the shared library.

    tsol_K selects the hot/cold preset. is_charging is accepted for backward
    compatibility and ignored: C_eff is symmetric (it represents stored
    energy, not heat transfer rate).
    """
    params = hc.PCM_Q9_HOT if tsol_K > 300 else hc.PCM_Q3_COLD
    return hc.effective_heat_capacity(pcm_temp_K - K_OFFSET, params, mass_fraction)


def liquid_fraction(pcm_temp_K, tsol_K, tliq_K):
    """Liquid fraction [0-1] for UA_sc conductance scaling. Delegates to the
    shared library; tsol_K selects the hot/cold preset."""
    params = hc.PCM_Q9_HOT if tsol_K > 300 else hc.PCM_Q3_COLD
    return hc.liquid_fraction(pcm_temp_K - K_OFFSET, params)


def water_properties(T_water_C):
    """Temperature-dependent water properties (mu [Pa.s], Pr [-], k [W/m/K])
    for convective HT. Delegates to the shared library."""
    return hc.water_properties(T_water_C)


def compute_water_to_pcm_ua(m_dot, is_hot, is_discharge=False, T_water_C=45.0):
    """
    Overall water-to-PCM UA [W/K] using the series resistance model
    (1/UA_total = 1/UA_water + 1/UA_pcm). Delegates to the shared library.

    is_hot selects the hot/cold preset. Returns 0 when m_dot <= 0.
    """
    params = hc.PCM_Q9_HOT if is_hot else hc.PCM_Q3_COLD
    return hc.water_to_pcm_ua(m_dot, is_discharge, T_water_C, params)
