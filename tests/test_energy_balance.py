"""Energy-balance / grid-accounting invariants for physics.loadmatch.gridtrading_events.

Covers the grid-flow outputs that the existing tests capture-and-discard
(test_battery_dod.py uses `b_new, _, _, _, _`) and pins the natural-dispatch energy balance
so a silent accounting/unit regression (e.g. the historical x12 PV over-count) can't pass CI.

Pure-function tests — no gym env / SB3 needed. Authored from a static reading of loadmatch.py;
run once in the `microgrid` env (`pytest tests/test_energy_balance.py -q`) to confirm.

gridtrading_events returns: (B_SOC_new, grid_purchase, grid_sale, curtailment, loss_of_load).
"""
import pytest

import Input_Parameters as IP
from physics.loadmatch import gridtrading_events

CAP = 10.0
SOC_MIN = IP.battery_soc_min_kwh(CAP)


@pytest.mark.parametrize("battery_action", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("dr,outage", [(0, 0), (1, 0), (0, 1)])
def test_grid_flow_invariants(battery_action, dr, outage):
    """Universal invariants that must hold for any correct dispatch."""
    _b, gp, gs, curt, lol = gridtrading_events(
        Gen=0.3, E_Load=0.2, M_Load=0.1, B_SOC=0.5 * CAP,
        battery_action=battery_action, DR_event=dr, Grid_outage=outage,
        battery_capacity=CAP,
    )
    for v in (gp, gs, curt, lol):          # nothing negative
        assert v >= -1e-9
    assert gp * gs == pytest.approx(0.0, abs=1e-9)   # never import and export in one step
    if outage:                             # M6/M12: only a physical OUTAGE blocks the grid
        assert gp == pytest.approx(0.0, abs=1e-9)   # import blocked (islanded)
        assert gs == pytest.approx(0.0, abs=1e-9)   # export blocked too
    # NB: DR is now a critical-peak PRICE event (grid stays up), so it no longer blocks import.
    assert SOC_MIN - 1e-6 <= _b <= CAP + 1e-6


def test_surplus_exports_generation():
    """Battery full + PV surplus, natural dispatch (action 0): surplus is exported/curtailed,
    not created or lost. A x12 PV over-count would break grid_sale == Gen - load."""
    gen, load = 0.5, 0.2
    _b, gp, gs, curt, lol = gridtrading_events(
        Gen=gen, E_Load=load, M_Load=0.0, B_SOC=CAP,
        battery_action=0, DR_event=0, Grid_outage=0, battery_capacity=CAP,
    )
    assert gp == pytest.approx(0.0, abs=1e-9)
    assert lol == pytest.approx(0.0, abs=1e-9)
    assert gs + curt == pytest.approx(gen - load, abs=1e-6)


def test_deficit_imports_shortfall():
    """Battery at floor + deficit, natural dispatch: shortfall is imported, no loss of load."""
    gen, load = 0.1, 0.5
    _b, gp, gs, curt, lol = gridtrading_events(
        Gen=gen, E_Load=load, M_Load=0.0, B_SOC=SOC_MIN,
        battery_action=0, DR_event=0, Grid_outage=0, battery_capacity=CAP,
    )
    assert gs == pytest.approx(0.0, abs=1e-9)
    assert gp == pytest.approx(load - gen, abs=1e-6)
    assert lol == pytest.approx(0.0, abs=1e-9)


def test_outage_deficit_is_loss_of_load():
    """During an outage with a deficit the (empty) battery can't cover, shortfall = loss of load."""
    _b, gp, gs, curt, lol = gridtrading_events(
        Gen=0.0, E_Load=1.0, M_Load=0.0, B_SOC=SOC_MIN,
        battery_action=0, DR_event=0, Grid_outage=1, battery_capacity=CAP,
    )
    assert gp == pytest.approx(0.0, abs=1e-9)
    assert gs == pytest.approx(0.0, abs=1e-9)
    assert lol > 0.0
