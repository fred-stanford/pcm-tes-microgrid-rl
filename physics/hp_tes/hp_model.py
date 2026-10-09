"""
hp_tes/hp_model.py - Heat pump COP/capacity from Viessmann AM2V 020028 lookup tables.

Replaces biquadratic polynomials with manufacturer data interpolation.
Heating: 2D bilinear interpolation on (outdoor_temp, supply_water_temp) grid.
Cooling: 1D interpolation on outdoor_temp (SWT fixed at 7C).
"""

import numpy as np

from .parameters import (
    K_OFFSET,
    HEATING_COP_COEFF, COOLING_COP_COEFF,
    HEATING_CAP_COEFF, COOLING_CAP_COEFF,
    HP_HEATING_CAP_SCALE, HP_COOLING_CAP_SCALE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Viessmann AM2V 020028 Lookup Tables (embedded for zero file I/O)
# ═══════════════════════════════════════════════════════════════════════════════

# Heating: 2D grids indexed by (outdoor_temp_C, supply_water_temp_C)
_HEAT_OAT = np.array([-25., -20., -15., -10., -5., 0., 5., 10., 15., 20.])
_HEAT_SWT = np.array([35., 41., 45., 50., 55., 60.])

# COP[i, j] = COP at outdoor_temp=_HEAT_OAT[i], supply_water_temp=_HEAT_SWT[j]
_HEAT_COP = np.array([
    [1.559, 1.340, 1.191, 1.098, 0.995, 0.995],  # -25C
    [1.927, 1.620, 1.421, 1.248, 1.075, 1.029],  # -20C
    [2.606, 2.380, 2.215, 1.777, 1.320, 1.087],  # -15C
    [2.905, 2.550, 2.307, 1.970, 1.559, 1.271],  # -10C
    [3.239, 2.860, 2.618, 2.169, 1.743, 1.363],  #  -5C
    [3.366, 2.890, 2.560, 2.192, 1.812, 1.501],  #   0C
    [3.884, 3.430, 3.136, 2.733, 2.295, 1.904],  #   5C
    [4.367, 3.660, 3.308, 2.951, 2.580, 2.238],  #  10C
    [4.609, 4.060, 3.700, 3.290, 2.870, 2.491],  #  15C
    [5.000, 4.450, 4.102, 3.630, 3.159, 2.744],  #  20C
])

# Capacity in kW (converted to W at lookup time)
_HEAT_CAP_KW = np.array([
    [1.7, 1.4, 1.2, 1.0, 0.8, 0.6],  # -25C
    [2.7, 2.3, 2.0, 1.6, 1.2, 0.9],  # -20C
    [4.3, 3.7, 3.3, 2.7, 2.1, 1.4],  # -15C
    [5.5, 4.8, 4.2, 3.5, 2.8, 2.1],  # -10C
    [6.3, 5.6, 5.0, 4.2, 3.4, 2.5],  #  -5C
    [6.8, 6.0, 5.4, 4.5, 3.7, 2.8],  #   0C
    [7.8, 7.0, 6.3, 5.3, 4.4, 3.4],  #   5C
    [8.7, 7.8, 7.1, 6.0, 5.0, 4.0],  #  10C
    [9.2, 8.3, 7.6, 6.5, 5.5, 4.5],  #  15C
    [9.8, 8.8, 8.0, 7.2, 6.3, 5.3],  #  20C
])

# Cooling: 1D arrays indexed by outdoor_temp (SWT fixed at 7C)
# Points below 27C are extrapolated from the 27-43C trend (slope ~0.1 COP/C, ~0.08 kW/C).
# At low outdoor temps, cooling is thermodynamically easy (large condenser dT),
# so COP rises substantially. Measured data confirms COP ~5.2 avg at 7-15C outdoor.
_COOL_OAT = np.array([-5., 0., 5., 10., 15., 20., 25., 27., 29., 31., 33., 35., 37., 39., 41., 43.])
_COOL_COP = np.array([6.85, 6.33, 5.81, 5.29, 4.77, 4.25, 3.73, 3.52, 3.31, 3.08, 2.87, 2.64, 2.46, 2.26, 2.05, 1.85])
_COOL_CAP_KW = np.array([6.50, 6.40, 6.25, 6.10, 6.00, 5.90, 5.82, 5.75, 5.59, 5.40, 5.21, 5.00, 4.78, 4.62, 4.56, 4.47])


# ═══════════════════════════════════════════════════════════════════════════════
# Interpolation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _interp2d(oat_C, swt_C, grid):
    """Bilinear interpolation on the heating grid, clamped to bounds."""
    oat = np.clip(oat_C, _HEAT_OAT[0], _HEAT_OAT[-1])
    swt = np.clip(swt_C, _HEAT_SWT[0], _HEAT_SWT[-1])

    # Find bounding indices for outdoor temp
    i = np.searchsorted(_HEAT_OAT, oat, side='right') - 1
    i = max(0, min(i, len(_HEAT_OAT) - 2))
    # Find bounding indices for supply water temp
    j = np.searchsorted(_HEAT_SWT, swt, side='right') - 1
    j = max(0, min(j, len(_HEAT_SWT) - 2))

    # Fractional positions
    oat_frac = (oat - _HEAT_OAT[i]) / (_HEAT_OAT[i + 1] - _HEAT_OAT[i])
    swt_frac = (swt - _HEAT_SWT[j]) / (_HEAT_SWT[j + 1] - _HEAT_SWT[j])

    # Bilinear interpolation
    v00 = grid[i, j]
    v10 = grid[i + 1, j]
    v01 = grid[i, j + 1]
    v11 = grid[i + 1, j + 1]

    return (v00 * (1 - oat_frac) * (1 - swt_frac) +
            v10 * oat_frac * (1 - swt_frac) +
            v01 * (1 - oat_frac) * swt_frac +
            v11 * oat_frac * swt_frac)


def _interp1d(oat_C, x_arr, y_arr):
    """1D linear interpolation, clamped to bounds."""
    return float(np.interp(oat_C, x_arr, y_arr))


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cop_and_capacity(Ta_K, T_hp_supply_K, heating):
    """
    Compute COP and thermal capacity from Viessmann lookup tables.

    Parameters
    ----------
    Ta_K : float  -- outdoor air temperature [K]
    T_hp_supply_K : float  -- HP supply water temperature [K]
    heating : bool  -- True for heating mode, False for cooling

    Returns
    -------
    cop : float  -- coefficient of performance (>=1)
    capacity : float  -- thermal capacity [W] (>=0)
    """
    Ta_C = Ta_K - K_OFFSET
    T_hp_C = T_hp_supply_K - K_OFFSET

    if heating:
        cop = _interp2d(Ta_C, T_hp_C, _HEAT_COP)
        cap_kw = _interp2d(Ta_C, T_hp_C, _HEAT_CAP_KW)
    else:
        cop = _interp1d(Ta_C, _COOL_OAT, _COOL_COP)
        cap_kw = _interp1d(Ta_C, _COOL_OAT, _COOL_CAP_KW)

    cop = max(cop, 1.0)
    cap = max(cap_kw * 1000.0, 0.0)  # kW -> W
    # Scale for actual HP (Aermec vs Viessmann lookup)
    cap *= HP_HEATING_CAP_SCALE if heating else HP_COOLING_CAP_SCALE
    return cop, cap


def compute_cop(Ta_K, T_hp_supply_K, heating):
    """
    Compute COP only (no capacity) from Viessmann lookup tables.

    Used by system.py for the cop_heating/cop_cooling snapshot values.

    Parameters
    ----------
    Ta_K : float  -- outdoor air temperature [K]
    T_hp_supply_K : float  -- HP supply water temperature [K]
    heating : bool  -- True for heating mode, False for cooling

    Returns
    -------
    cop : float  -- coefficient of performance (>=1)
    """
    Ta_C = Ta_K - K_OFFSET
    T_hp_C = T_hp_supply_K - K_OFFSET

    if heating:
        cop = _interp2d(Ta_C, T_hp_C, _HEAT_COP)
    else:
        cop = _interp1d(Ta_C, _COOL_OAT, _COOL_COP)

    return max(cop, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy polynomial (kept for backward compat — validate_components.py uses it)
# ═══════════════════════════════════════════════════════════════════════════════

def _poly_eval(coeffs, Ta_C, T_hp_C):
    """Evaluate biquadratic polynomial: a0 + a1*Ta + a2*Thp + a3*Ta*Thp + a4*Ta^2 + a5*Thp^2"""
    a = coeffs
    return (a[0] + a[1] * Ta_C + a[2] * T_hp_C +
            a[3] * Ta_C * T_hp_C + a[4] * Ta_C**2 + a[5] * T_hp_C**2)
