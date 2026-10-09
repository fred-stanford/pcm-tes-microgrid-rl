"""
hp_tes_rl_dropin - Drop-in replacement for the hp_tes hybrid HVAC system model.

Exposes the IDENTICAL public API and run_fmu output contract as the old hp_tes
package, so code bound to hp_tes (e.g. the RL environment) works unchanged.
Only the PCM (TES) physics differs: the old 2-node shell/core PCM model is
replaced by a single-node PCMNode per tank driven by the validated v2
PCMBattery calibration (baked, data-free -- see _baked_calibration.json).

Re-exports the same names as hp_tes.__init__, plus:
  * run_python_hp_tes  -- alias of run_fmu (the name the RL environment calls)
  * PCMNode            -- the single-node PCM stepper (new physics)
"""

from .parameters import *
from .mixing_volume import MixingVolume
from .tes_model import (
    compute_soc, compute_hot_soc, compute_cold_soc,
    effective_heat_capacity, liquid_fraction,
    compute_water_to_pcm_ua,
)
from .hp_model import compute_cop_and_capacity, compute_cop, _poly_eval
from .fcu_model import FanCoilUnit
from .controller import (
    determine_mode, update_thermostat, update_soc_hysteresis,
)
from .system import HybridSystemState, run_fmu, _simulate_interval
from ._stepper import PCMNode

# The RL environment calls this name.
run_python_hp_tes = run_fmu
