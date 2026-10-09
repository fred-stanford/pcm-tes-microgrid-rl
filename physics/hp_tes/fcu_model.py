"""
hp_tes/fcu_model.py - Fan coil unit model.
"""

from .parameters import UA_HEATING, UA_COOLING


class FanCoilUnit:
    """Simple Q=UA*dT fan coil unit model (stateless)."""

    def compute_heat(self, fcu_mode, water_temp_K, room_temp_K):
        """
        Compute FCU heat transfer.

        Parameters
        ----------
        fcu_mode : int — -1=cooling, 0=off, 1=heating
        water_temp_K : float — entering water temperature [K]
        room_temp_K : float — room air temperature [K]

        Returns
        -------
        fcu_heat : float — heat delivered [W]. Positive=heating, negative=cooling.
        """
        if fcu_mode == 0:
            return 0.0
        elif fcu_mode == 1:
            fcu_heat = UA_HEATING * (water_temp_K - room_temp_K)
            return max(fcu_heat, 0.0)
        elif fcu_mode == -1:
            fcu_heat = UA_COOLING * (water_temp_K - room_temp_K)
            return min(fcu_heat, 0.0)
        return 0.0
