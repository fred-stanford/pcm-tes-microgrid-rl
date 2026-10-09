# -*- coding: utf-8 -*-
"""System schematic: grid-tied PV + battery + reversible HP + hot/cold PCM + FCU + building, with controller."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_MIRROR = os.environ.get("FIG_MIRROR")                    # optional extra output folder (its main/ subfolder)
MAINS = [os.path.join(_HERE, "main")] + ([os.path.join(_MIRROR, "main")] if _MIRROR and os.path.isdir(_MIRROR) else [])
# palette
ELEC_F, ELEC_E = "#D9E8F5", "#0072B2"   # electrical
BATT_F, BATT_E = "#D6F0E6", "#009E73"   # battery
HOT_F,  HOT_E  = "#FBE3D2", "#D55E00"   # hot / thermal
COLD_F, COLD_E = "#D9EEF9", "#56B4E9"   # cold
PV_F,   PV_E   = "#FCEFD0", "#E69F00"   # PV
CTRL_F, CTRL_E = "#ECECEC", "#444444"   # controller
BLDG_F, BLDG_E = "#F2F2F2", "#666666"   # building
GRID_F, GRID_E = "#EDEDED", "#333333"

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(13.5, 8.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(cx, cy, w, h, text, fc, ec, fs=10, bold=True):
    x, y = cx - w/2, cy - h/2
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                       linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight=("bold" if bold else "normal"), color="#1a1a1a")

def arrow(p1, p2, color, ls="-", two=False, lw=2.2, z=2):
    ax.annotate("", xy=p2, xytext=p1, zorder=z,
                arrowprops=dict(arrowstyle=("<|-|>" if two else "-|>"), color=color,
                                lw=lw, linestyle=ls, shrinkA=3, shrinkB=3,
                                mutation_scale=18))

# ---- electrical bus ----
BUS_Y = 64
ax.plot([13, 90], [BUS_Y, BUS_Y], color=ELEC_E, lw=5, solid_capstyle="round", zorder=1)
ax.text(51.5, BUS_Y + 2.0, "AC electrical bus", ha="center", fontsize=9.5, style="italic", color=ELEC_E)

# ---- grid ----
box(7, 64, 12, 15, "Grid\nPG&E E-ELEC\nTOU import /\nNBT export", GRID_F, GRID_E, fs=9)
arrow((13.2, 64), (16, 64), ELEC_E, two=True)
ax.text(7, 52, "grid events:\nDR · outage · PSPS", ha="center", fontsize=9.5, color="#8a2b06", style="italic")

# ---- PV / battery / load on the bus ----
box(24, 88, 14, 12, "PV array\n3.84 kW", PV_F, PV_E, fs=9.5)
arrow((24, 82), (24, BUS_Y + 0.5), PV_E)
box(46, 88, 16, 12, "Battery\n2×3.6 kWh LFP\n(7.2 kWh)", BATT_F, BATT_E, fs=9.5)
arrow((46, 82), (46, BUS_Y + 0.5), BATT_E, two=True)
box(81, 88, 16, 12, "Building\nelectric load", ELEC_F, ELEC_E, fs=9.5)
arrow((81, BUS_Y + 0.5), (81, 82), ELEC_E)

# ---- heat pump (bridges electrical <-> thermal) ----
box(60, 47, 19, 15, "Reversible\nair-to-water\nheat pump\n(Viessmann\nVitocal 100-AW)", ELEC_F, ELEC_E, fs=9.2)
arrow((60, BUS_Y - 0.5), (60, 55), ELEC_E)          # electricity into HP
ax.text(61.5, 59.5, "electricity", fontsize=9.5, color=ELEC_E, ha="left", va="center")

# ---- controller (same size as HP box, parallel to it) ----
box(86, 47, 19, 15,
    "Supervisory\ncontroller\n(RL  /  RBC)\n15-min · 20 actions\n(5 batt × 4 HP/TES)",
    CTRL_F, CTRL_E, fs=8.8)
arrow((76.5, 47), (69.5, 47), "#666666", ls=(0, (5, 4)), lw=1.8)               # cmd -> HP
arrow((81, 54.5), (48, 82), "#666666", ls=(0, (5, 4)), lw=1.8)                 # cmd -> battery
ax.text(73, 50, "mode cmd", fontsize=9.5, color="#555555", style="italic", ha="center")
ax.text(66, 70, "battery cmd", fontsize=9.5, color="#555555", style="italic", ha="center")
arrow((89, 26), (88, 39), "#999999", ls=(0, (1, 2)), lw=1.7)                   # observations into controller
ax.text(89, 15, "observations:\nSOC · indoor T · price\nPV · grid events\n(forecast-free)",
        fontsize=9, color="#777777", style="italic", ha="center")

# ---- thermal storage + FCU + building ----
box(30, 26, 16, 13, "Hot PCM TES\n(Sunamp)", HOT_F, HOT_E, fs=9)
box(74, 26, 16, 13, "Cold PCM TES\n(Sunamp)", COLD_F, COLD_E, fs=9)
box(52, 24, 15, 11, "Fan-coil\nunit (FCU)", HOT_F, HOT_E, fs=9)
box(52, 7, 56, 9, "Building — Half Moon Bay, CA\nindoor temperature · comfort band 21.5–24.5 °C", BLDG_F, BLDG_E, fs=9.5)

# hydronic (thermal) flows
arrow((53, 41), (36, 32.5), HOT_E, two=True, lw=2.4)     # HP <-> hot PCM
arrow((67, 41), (74, 32.5), COLD_E, two=True, lw=2.4)    # HP <-> cold PCM
arrow((38, 24), (44.5, 24), HOT_E, lw=2.2)               # hot PCM -> FCU
arrow((66, 24), (59.5, 24), COLD_E, lw=2.2)              # cold PCM -> FCU
arrow((60, 40), (55, 29.5), HOT_E, ls=(0, (5, 3)), lw=1.8)  # HP direct -> FCU
ax.text(61, 35, "direct", fontsize=9.5, color=HOT_E, style="italic", ha="left")
arrow((52, 18.5), (52, 11.5), HOT_E, two=True, lw=2.4)   # FCU <-> building (deliver heat/cool)

# domain labels
ax.text(2, 97, "Electrical domain", fontsize=10.5, fontweight="bold", color=ELEC_E)
ax.text(2, 37, "Thermal / hydronic domain", fontsize=10.5, fontweight="bold", color=HOT_E)
ax.axhline(0, 0, 0)  # noop

# legend
leg = [
    Line2D([0], [0], color=ELEC_E, lw=3, label="Electrical flow"),
    Line2D([0], [0], color=HOT_E, lw=3, label="Thermal flow (hot)"),
    Line2D([0], [0], color=COLD_E, lw=3, label="Thermal flow (cold)"),
    Line2D([0], [0], color="#666666", lw=2, ls=(0, (5, 4)), label="Control command"),
    Line2D([0], [0], color="#999999", lw=2, ls=(0, (1, 2)), label="Observation"),
]
ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.005, 0.005), fontsize=8.6,
          frameon=True, framealpha=0.9, ncol=1)

fig.tight_layout()
for main in MAINS:
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(main, f"Fig1_system_schematic.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote Fig1_system_schematic")
