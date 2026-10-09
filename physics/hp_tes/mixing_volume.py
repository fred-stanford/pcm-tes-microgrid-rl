"""
hp_tes/mixing_volume.py - Simple well-mixed water volume for thermal inertia.
"""

from .parameters import CP_WATER, RHO_WATER


class MixingVolume:
    """Simple well-mixed water volume for thermal inertia modeling."""

    def __init__(self, volume_L, initial_temp_K):
        self.C = volume_L * RHO_WATER * CP_WATER / 1000.0  # J/K  (liters → m³)
        self.temp = initial_temp_K

    def update(self, T_in, m_dot, dt):
        """Euler step: dT/dt = m_dot * cp * (T_in - T) / C.  Returns outlet temp."""
        if m_dot > 0 and self.C > 0 and dt > 0:
            dT = m_dot * CP_WATER * (T_in - self.temp) * dt / self.C
            self.temp += dT
        return self.temp
