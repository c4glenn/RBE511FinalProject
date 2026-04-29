"""
viz.py — SolaraViz visualization for SwarmModel
Run with:  solara run viz.py
"""
import numpy as np
import solara
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from mesa.visualization import SolaraViz, make_plot_component
from mesa.visualization.utils import update_counter

from model import SwarmModel
from agents import State, STATE_COLORS 


# ── Model parameters exposed in the Solara sidebar ───────────────────────────

model_params = {
    "n_robots": {
        "type": "SliderInt",
        "value": 7,
        "label": "Number of robots",
        "min": 2,
        "max": 40,
        "step": 1,
    },
    "task_dist_calc":{
        "type": "SliderInt",
        "value": 70,
        "label": "Percent for task 1",
        "min": 1,
        "max": 100,
        "step": 1
    },
    "speed": {
        "type": "SliderFloat",
        "value": 20.0,
        "label": "Robot speed",
        "min": 0.5,
        "max": 50.0,
        "step": 0.5,
    },
    "arena_width": 600.0,   # fixed — not shown in sidebar
    "arena_height": 200.0,  # fixed — not shown in sidebar
    "interface_gap": 25.0,  # fixed — not shown in sidebar
    "task_distribution": np.array([None]),
    "n_tasks": 1,
}


# ── Arena / pipeline drawing helper ──────────────────────────────────────────

def _draw_pipeline(ax, model):
    """Draw the static pipeline geometry onto *ax*."""
    pipe = model.pipeline
    W, H = pipe.arena_width, pipe.arena_height

    # Arena border
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), W, H,
        boxstyle="round,pad=2", linewidth=1.2,
        edgecolor="#888", facecolor="#F7F7F7", zorder=0,
    ))

    mid_y = H / 2

    # Segment lanes (alternating shading)
    seg_edges = (
        [0]
        + [x for t in pipe.tasks for x in (t.entry[0], t.exit[0])]
        + [W]
    )
    seg_left  = seg_edges[0::2]
    seg_right = seg_edges[1::2]
    for i, (lx, rx) in enumerate(zip(seg_left, seg_right)):
        color = "#E8F4FD" if i % 2 == 0 else "#EDF7ED"
        ax.add_patch(mpatches.Rectangle(
            (lx, 0), rx - lx, H,
            facecolor=color, edgecolor="none", zorder=1,
        ))
        label = f"Seg {i}"
        ax.text((lx + rx) / 2, H - 8, label,
                ha="center", va="top", fontsize=7,
                color="#999", zorder=5)

    # Task interface zones
    for task in pipe.tasks:
        lx, rx = task.entry[0], task.exit[0]
        ax.add_patch(mpatches.Rectangle(
            (lx, 0), rx - lx, H,
            facecolor="#FFF3CD", edgecolor="#E6A817",
            linewidth=0.8, zorder=2,
        ))
        ax.text((lx + rx) / 2, H / 2, task.label,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="#9A6500", zorder=6)
        # Interface boundary lines
        for bx in (lx, rx):
            ax.plot([bx, bx], [0, H], color="#E6A817",
                    linewidth=0.8, linestyle="--", zorder=3)

    # Source marker
    sx, sy = pipe.source_pos
    ax.add_patch(mpatches.Rectangle(
            (0, 0), sx, H,
            facecolor="#2196F3", edgecolor="#4618EF",
            linewidth=0.8, zorder=2,
        ))
    ax.text(sx / 2, sy, "SRC", ha="center", va="top",
            fontsize=7, color="white", zorder=7)

    # Nest marker
    nx, ny = pipe.nest_pos
    ax.add_patch(mpatches.Rectangle(
            (nx, 0), W-nx, H,
            facecolor="#2196F3", edgecolor="#9C27B0",
            linewidth=0.8, zorder=2,
        ))
    ax.text(nx + 0.5 * (W-nx), ny, "NEST", ha="center", va="top",
            fontsize=7, color="white", zorder=7)

    # Centreline
    # ax.plot([0, W], [mid_y, mid_y],
    #         color="#CCC", linewidth=0.6, linestyle=":", zorder=3)


def _draw_robots(ax, model):
    """Scatter-plot all robots onto *ax*, coloured by state."""
    for agent in model.agents:
        x, y = float(agent.pos[0]), float(agent.pos[1])
        color = STATE_COLORS[agent.state]

        # Main circle
        ax.plot(x, y, "o", markersize=8,
                color=color, markeredgecolor="#333",
                markeredgewidth=0.6, zorder=10)

        # Outline ring when carrying an object
        if agent.has_object:
            ax.plot(x, y, "o", markersize=13,
                    color="none", markeredgecolor="#555",
                    markeredgewidth=1.4, zorder=9)


# ── Custom Solara component ───────────────────────────────────────────────────

@solara.component # pyright: ignore[reportPrivateImportUsage]
def ArenaView(model):
    """Live matplotlib view of the arena, pipeline, and robots."""
    update_counter.get()   # required — tells Solara to re-render each step

    pipe = model.pipeline
    W, H = pipe.arena_width, pipe.arena_height

    fig = Figure(figsize=(9, 3.2))
    ax  = fig.add_subplot(111)

    _draw_pipeline(ax, model)
    _draw_robots(ax, model)

    # Legend for states
    handles = [
        mpatches.Patch(color=STATE_COLORS[s], label=s.name.replace("_", " ").title())
        for s in State
    ]
    ax.legend(
        handles=handles,
        loc="upper right", fontsize=7,
        framealpha=0.85, handlelength=1.0,
    )

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Step {model.steps}  |  Deliveries: {model.total_deliveries}  "
        f"|  Throughput: {model._throughput():.2f}/step",
        fontsize=9, pad=6,
    )
    fig.tight_layout(pad=0.4)

    solara.FigureMatplotlib(fig)


# ── Throughput / delivery plots ───────────────────────────────────────────────

DeliveryPlot = make_plot_component("total_deliveries")
ThroughputPlot = make_plot_component("throughput")
Task_assignment_plot = make_plot_component([f"Segment {s}" for s in range(2)])

# StateBreakdownPlot = make_plot_component("")


# ── Page entry-point ──────────────────────────────────────────────────────────

model = SwarmModel(
    n_robots=20,
    n_tasks=1,
    speed=3.0,
    task_distribution=np.array([50,50])
)

page = SolaraViz(
    model,
    components=[
        ArenaView,
        DeliveryPlot,
        Task_assignment_plot,
    ], # pyright: ignore[reportArgumentType]
    model_params=model_params,
    name="Swarm Pipeline",
)