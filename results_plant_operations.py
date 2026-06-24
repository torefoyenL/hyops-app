# -*- coding: utf-8 -*-
"""
results_plant_operations.py
3-zone KPIs and plots for dual-compressor hydrogen plant.

Time zones per container
------------------------
  external_wait  : arrival        → dock_step        (outside filling area)
  docked_wait    : dock_step      → start_fill_step  (parked, waiting for compressor)
  fill_time      : start_fill_step → completion_step (compressor pumping)
  total_time     : arrival        → completion_step
"""

from plant_operations import adiabatic_compression_energy


# ============================================================
# Architecture diagram colour palette
# ============================================================

_AC = {
    "bg":     "#1a1a2e",
    "ez":     "#16213e",
    "stack":  "#0f3460",
    "comp":   "#533483",
    "fill":   "#e94560",
    "arrow":  "#a0a0c0",
    "header": "#c0c0d8",
    "label":  "#e0e0f0",
}


# ============================================================
# KPI Computation
# ============================================================

def compute_kpis(results, plant, container_types,
                 avg_arrivals_per_day, days):
    import numpy as np
    completed    = results["completed"]
    energy_log   = results["energy_log"]
    step_minutes = results["step_minutes"]
    TIMESTEPS    = len(results["production_log"])

    # ----------------------------------------------------------
    # Per-container time breakdowns
    # ----------------------------------------------------------
    external_waits = []   # outside → dock
    docked_waits   = []   # docked  → filling starts
    fill_times     = []   # filling → done
    total_times    = []

    for c in completed:
        # dock_step may be None if container went straight to dock
        dock  = c.dock_step       if c.dock_step       is not None else c.arrival_step
        start = c.start_fill_step if c.start_fill_step is not None else dock

        ext   = (dock  - c.arrival_step) * step_minutes
        doc   = (start - dock)            * step_minutes
        fill  = (c.completion_step - start) * step_minutes
        total = (c.completion_step - c.arrival_step) * step_minutes

        external_waits.append(max(ext,  0))
        docked_waits.append(  max(doc,  0))
        fill_times.append(    max(fill, 0))
        total_times.append(   max(total,0))

    def stats(arr):
        if not arr:
            return dict(avg=0, max=0, median=0, p95=0, zero=0)
        return dict(
            avg    = np.mean(arr),
            max    = np.max(arr),
            median = np.median(arr),
            p95    = np.percentile(arr, 95),
            zero   = sum(v == 0 for v in arr),
        )

    ext_stats  = stats(external_waits)
    doc_stats  = stats(docked_waits)
    fill_stats = stats(fill_times)

    # ----------------------------------------------------------
    # Queue lengths
    # ----------------------------------------------------------
    ext_log    = results["queue_log"]          # outside filling area
    docked_log = results.get("docked_log", [])
    fill_log   = results.get("filling_log")

    # ----------------------------------------------------------
    # Per-compressor stats
    # ----------------------------------------------------------
    # Per-compressor stats (shared-line model: only fill utilisation)
    # ----------------------------------------------------------
    comp_filling_log = results.get("comp_filling_log", None)

    comp_stats = []
    if comp_filling_log is not None:
        for i in range(len(comp_filling_log)):
            comp_stats.append({
                "id":               i,
                "fill_utilization": np.mean(comp_filling_log[i]),
            })

    # ----------------------------------------------------------
    # Energy
    # ----------------------------------------------------------
    max_possible = plant.total_production * TIMESTEPS
    utilization  = plant.total_dispensed / max_possible if max_possible > 0 else 0

    total_energy = sum(e * (step_minutes / 60) for e in energy_log)
    avg_e_per_kg = total_energy / plant.total_dispensed if plant.total_dispensed > 0 else 0

    ideal_kwh    = adiabatic_compression_energy(plant.total_dispensed, P1_bar=30, P2_bar=380)
    ref_kwh      = ideal_kwh / plant.Isentropic_efficiency if plant.Isentropic_efficiency > 0 else 0
    ref_per_kg   = ref_kwh / plant.total_dispensed if plant.total_dispensed > 0 else 0

    container_counts = {
        t.name: sum(1 for c in completed if c.container_type.name == t.name)
        for t in container_types
    }

    return {
        "simulation_time_min": TIMESTEPS * step_minutes,
        "containers_filled":   len(completed),
        "container_counts":    container_counts,

        # External wait (outside filling area)
        "ext_avg":    ext_stats["avg"],
        "ext_max":    ext_stats["max"],
        "ext_median": ext_stats["median"],
        "ext_p95":    ext_stats["p95"],
        "ext_zero":   ext_stats["zero"],

        # Docked wait (parked, compressor busy)
        "doc_avg":    doc_stats["avg"],
        "doc_max":    doc_stats["max"],
        "doc_median": doc_stats["median"],
        "doc_p95":    doc_stats["p95"],
        "doc_zero":   doc_stats["zero"],

        # Fill time (compressor pumping)
        "fill_avg":    fill_stats["avg"],
        "fill_max":    fill_stats["max"],
        "fill_median": fill_stats["median"],
        "fill_p95":    fill_stats["p95"],

        # Total
        "avg_total_time_min": np.mean(total_times) if total_times else 0,

        # Queue occupancy
        "avg_external_queue": np.mean(ext_log),
        "max_external_queue": np.max(ext_log),
        "avg_docked_queue":   np.mean(docked_log) if docked_log else 0,
        "max_docked_queue":   np.max(docked_log)  if docked_log else 0,
        "avg_filling":        np.mean(fill_log),
        "max_filling":        np.max(fill_log),

        "comp_stats": comp_stats,

        # Energy
        "plant_utilization":                utilization,
        "total_dispensed_kg":               plant.total_dispensed,
        "max_possible_dispense_kg":         max_possible,
        "total_energy_kwh":                 total_energy,
        "avg_energy_kwh_per_kg":            avg_e_per_kg,
        "reference_compression_kwh_per_kg": ref_per_kg,

        # Raw arrays for plots
        "_external_waits": external_waits,
        "_docked_waits":   docked_waits,
        "_fill_times":     fill_times,
        "_completed":      completed,
    }


# ============================================================
# KPI Printing
# ============================================================

def print_kpis(kpis):

    print("\n=== Simulation Summary ===")
    print(f"  Simulation time:   {kpis['simulation_time_min']} minutes")
    print(f"  Containers filled: {kpis['containers_filled']}")
    for name, count in kpis["container_counts"].items():
        print(f"    {name}: {count}")

    print("\n=== Waiting Time — Outside Filling Area (external queue) ===")
    print(f"  Average:         {kpis['ext_avg']:.1f} min")
    print(f"  Max:             {kpis['ext_max']:.1f} min")
    print(f"  Median:          {kpis['ext_median']:.1f} min")
    print(f"  95th percentile: {kpis['ext_p95']:.1f} min")
    print(f"  Zero-wait:       {kpis['ext_zero']}")

    print("\n=== Docked Wait — Parked at Fill Line, Compressor Busy ===")
    print(f"  Average:         {kpis['doc_avg']:.1f} min")
    print(f"  Max:             {kpis['doc_max']:.1f} min")
    print(f"  Median:          {kpis['doc_median']:.1f} min")
    print(f"  95th percentile: {kpis['doc_p95']:.1f} min")
    print(f"  Zero-wait:       {kpis['doc_zero']}")

    print("\n=== Fill Time — Compressor Actively Pumping ===")
    print(f"  Average:         {kpis['fill_avg']:.1f} min")
    print(f"  Max:             {kpis['fill_max']:.1f} min")
    print(f"  Median:          {kpis['fill_median']:.1f} min")
    print(f"  95th percentile: {kpis['fill_p95']:.1f} min")

    print(f"\n  Average total time in system: {kpis['avg_total_time_min']:.1f} min")

    print("\n=== Zone Occupancy (avg containers) ===")
    print(f"  External queue:  avg {kpis['avg_external_queue']:.2f}  max {kpis['max_external_queue']}")
    print(f"  Docked (waiting): avg {kpis['avg_docked_queue']:.2f}  max {kpis['max_docked_queue']}")
    print(f"  Filling:          avg {kpis['avg_filling']:.2f}       max {kpis['max_filling']}")

    if kpis["comp_stats"]:
        print("\n=== Per-Compressor Fill Utilisation ===")
        print("  (External queue and docked pool are shared between compressors)")
        for cs in kpis["comp_stats"]:
            print(f"  Compressor {cs['id']}: fill utilization {cs['fill_utilization']:.1%}")

    print("\n=== Energy ===")
    print(f"  Plant utilization:        {kpis['plant_utilization']:.2%}")
    print(f"  Total dispensed:          {kpis['total_dispensed_kg']:.2f} kg")
    print(f"  Total energy:             {kpis['total_energy_kwh']:.2f} kWh")
    print(f"  Avg energy per kg:        {kpis['avg_energy_kwh_per_kg']:.2f} kWh/kg")
    print(f"  Reference compression/kg: {kpis['reference_compression_kwh_per_kg']:.2f} kWh/kg")


# ============================================================
# Plotting
# ============================================================

def plot_results(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    step_minutes      = results["step_minutes"]
    pressure_log      = results["pressure_log"]
    production_log    = results["production_log"]
    energy_log        = results["energy_log"]
    arrival_log       = results["arrival_log"]
    ext_log           = results["queue_log"]
    docked_log        = results.get("docked_log", [])
    filling_log       = results.get("filling_log", results["active_log"])
    comp_filling_log  = results.get("comp_filling_log",  None)
    completed         = results["completed"]

    TIMESTEPS    = len(pressure_log)
    time_minutes = np.arange(TIMESTEPS) * step_minutes

    # =========================================================
    # 1️⃣  Pressure + Production
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(14, 5))
    l1, = ax1.plot(time_minutes, pressure_log,
                   color="blue", linewidth=1.5, label="Pressure (bar)")
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Pressure (bar)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    l2, = ax2.plot(time_minutes, production_log,
                   color="green", linestyle="--", linewidth=1.5,
                   label="Production (kg/h)")
    ax2.set_ylabel("Production (kg/h)", color="green")
    ax2.tick_params(axis='y', labelcolor="green")
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left")
    ax1.set_title("Plant Pressure and Production Rate")
    fig.tight_layout()
    plt.show()

    # =========================================================
    # 2️⃣  3-Zone Queue Overview (plant level)
    # =========================================================
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.step(time_minutes, ext_log, where="post",
            color="red", linewidth=2, label="External queue (outside)")
    if docked_log:
        ax.step(time_minutes, docked_log, where="post",
                color="orange", linewidth=2, linestyle="--",
                label="Docked — waiting for compressor")
    ax.step(time_minutes, filling_log, where="post",
            color="steelblue", linewidth=2, linestyle=":",
            label="Filling (compressor active)")

    # Arrival dots
    arrival_times = [time_minutes[i] for i in range(TIMESTEPS) if arrival_log[i] > 0]
    if arrival_times:
        ax.scatter(arrival_times, [0] * len(arrival_times),
                   color="black", s=40, zorder=5, label="Arrivals", marker="|")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Number of Containers")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_title("Container Zones Over Time: External → Docked → Filling")
    fig.tight_layout()
    plt.show()

    # =========================================================
    # 3️⃣  Per-Compressor Filling Utilisation
    #     (Shared-line model: external queue and docked pool are
    #      plant-wide, not per-compressor)
    # =========================================================
    if comp_filling_log is not None:
        n_comp = len(comp_filling_log)
        comp_colors = ["steelblue", "mediumorchid"]

        fig, ax = plt.subplots(figsize=(14, 4))
        for i in range(n_comp):
            ax.step(time_minutes, comp_filling_log[i], where="post",
                    color=comp_colors[i % len(comp_colors)],
                    linewidth=1.5, alpha=0.85,
                    label=f"Compressor {i} filling (0/1)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Filling (1 = active)")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("Per-Compressor Filling Activity (Shared Fill Lines)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

        # ---- Bar: avg filling utilisation per compressor ----
        fig, ax = plt.subplots(figsize=(6, 4))
        x         = np.arange(n_comp)
        avg_fill  = [np.mean(comp_filling_log[i]) for i in range(n_comp)]
        bars = ax.bar(x, avg_fill, color=comp_colors[:n_comp], alpha=0.85, edgecolor="black")
        ax.bar_label(bars, fmt="{:.1%}", padding=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Compressor {i}" for i in range(n_comp)])
        ax.set_ylabel("Fill utilisation (fraction of time active)")
        ax.set_ylim(0, 1.15)
        ax.set_title("Average Fill Utilisation per Compressor")
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        plt.show()

    # =========================================================
    # 4️⃣  Time Distribution Histograms — all 3 zones
    # =========================================================
    ext_waits  = [(c.dock_step  - c.arrival_step)    * step_minutes
                  for c in completed
                  if c.dock_step is not None]
    doc_waits  = [(c.start_fill_step - c.dock_step)  * step_minutes
                  for c in completed
                  if c.dock_step is not None and c.start_fill_step is not None]
    fill_times = [(c.completion_step - c.start_fill_step) * step_minutes
                  for c in completed
                  if c.start_fill_step is not None]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if ext_waits:
        axes[0].hist(ext_waits,  bins=25, color="firebrick",  edgecolor="black", alpha=0.85)
    axes[0].set_title("External Wait\n(outside filling area)")
    axes[0].set_xlabel("Minutes")
    axes[0].set_ylabel("Containers")
    axes[0].grid(True, alpha=0.3)

    if doc_waits:
        axes[1].hist(doc_waits,  bins=25, color="darkorange", edgecolor="black", alpha=0.85)
    axes[1].set_title("Docked Wait\n(parked, compressor busy)")
    axes[1].set_xlabel("Minutes")
    axes[1].grid(True, alpha=0.3)

    if fill_times:
        axes[2].hist(fill_times, bins=25, color="steelblue",  edgecolor="black", alpha=0.85)
    axes[2].set_title("Fill Time\n(compressor pumping)")
    axes[2].set_xlabel("Minutes")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Time Distribution by Zone", fontsize=13)
    fig.tight_layout()
    plt.show()

    # =========================================================
    # 5️⃣  Box plots by container type — all 3 zones
    # =========================================================
    types = sorted({c.ctype.name for c in completed})

    if len(types) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        zone_data = [
            ("External Wait",  "firebrick",
             [[max((c.dock_step - c.arrival_step)*step_minutes, 0)
               for c in completed if c.ctype.name == t and c.dock_step is not None]
              for t in types]),
            ("Docked Wait",    "darkorange",
             [[max((c.start_fill_step - c.dock_step)*step_minutes, 0)
               for c in completed
               if c.ctype.name == t and c.dock_step is not None
               and c.start_fill_step is not None]
              for t in types]),
            ("Fill Time",      "steelblue",
             [[max((c.completion_step - c.start_fill_step)*step_minutes, 0)
               for c in completed
               if c.ctype.name == t and c.start_fill_step is not None]
              for t in types]),
        ]

        for ax, (title, color, data) in zip(axes, zone_data):
            bp = ax.boxplot(data, labels=types, patch_artist=True,
                            boxprops=dict(facecolor=color, alpha=0.6))
            ax.set_title(title)
            ax.set_ylabel("Minutes")
            ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle("Time by Container Type and Zone", fontsize=13)
        fig.tight_layout()
        plt.show()

    # =========================================================
    # 6️⃣  Power
    # =========================================================
    plt.figure(figsize=(12, 4))
    plt.plot(time_minutes, energy_log, color="black", linewidth=1.5, label="Power (kW)")
    plt.title("Instantaneous Power Use")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # =========================================================
    # 7️⃣  Pressure Distribution
    # =========================================================
    plt.figure(figsize=(9, 4))
    plt.hist(pressure_log, bins=30, color="skyblue", edgecolor="black")
    plt.title("Plant Pressure Distribution")
    plt.xlabel("Pressure (bar)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# ============================================================
# Plant Architecture Diagram
# ============================================================

def _arch_box(ax, x, y, w, h, color, label, sublabel=None):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h, boxstyle="round,pad=0",
        facecolor=color, edgecolor="white", linewidth=0.6, zorder=3))
    if sublabel:
        ax.text(x, y + h*0.16, label, zorder=4, ha="center", va="center",
                fontsize=6.8, color="white", fontweight="bold", clip_on=False)
        ax.text(x, y - h*0.22, sublabel, zorder=4, ha="center", va="center",
                fontsize=5.3, color="white", style="italic", clip_on=False, alpha=0.88)
    else:
        ax.text(x, y, label, zorder=4, ha="center", va="center",
                fontsize=7.2, color="white", fontweight="bold", clip_on=False)

def _arch_hl(ax, y, x0, x1, lw=0.7):
    ax.plot([x0, x1], [y, y], color=_AC["arrow"], lw=lw, zorder=1)

def _arch_vl(ax, x, y0, y1, lw=0.7):
    ax.plot([x, x], [y0, y1], color=_AC["arrow"], lw=lw, zorder=1)

def _draw_single_block(ax, topology, x0, x1, y_top, y_bottom, dedicated_lines=False):
    n_ez   = getattr(topology, '_tmp_n_ez',   topology.n_electrolyzers if topology.mode != "trains" else 2)
    n_comp = getattr(topology, '_tmp_n_comp',  topology.n_compressors  if topology.mode != "trains" else 1)
    n_fill = getattr(topology, '_tmp_n_fill',  topology.total_fill_lines())
    stacks = getattr(topology, '_tmp_stacks',  topology.stacks_per_electrolyzer if topology.mode != "trains" else 2)
    ez_rate   = getattr(topology, '_tmp_ez_rate',   topology.electrolyzer_kg_per_hr_each if topology.mode != "trains" else 1.0)
    comp_rate = getattr(topology, '_tmp_comp_rate',  topology.compressor_flow_kg_per_hr_each if topology.mode != "trains" else 1.0)

    span = x1 - x0
    BW = max(min(0.14, span / max(n_ez, 1) * 0.65), 0.085)
    BH = 0.044
    SBW = BW * 0.55
    SBH = 0.030

    total_h = y_top - y_bottom
    y_ez     = y_top
    y_stack  = y_top - total_h * 0.14
    y_header = y_stack - total_h * 0.10
    y_comp   = y_header - total_h * 0.12
    y_fill   = y_comp   - total_h * 0.16

    x_ez = [x0 + span * (i + 0.5) / n_ez for i in range(n_ez)]

    for i, xez in enumerate(x_ez):
        _arch_box(ax, xez, y_ez, BW, BH, _AC["ez"], f"EZ {i+1}", f"{ez_rate:.2f} kg/h  β=2.5")
        st_coll = y_stack + SBH/2 + 0.014
        _arch_vl(ax, xez, y_ez - BH/2, st_coll)
        dx = SBW * 0.62
        xa, xb = xez - dx, xez + dx
        if stacks >= 2:
            _arch_hl(ax, st_coll, xa, xb)
            _arch_vl(ax, xa, st_coll, y_stack + SBH/2)
            _arch_vl(ax, xb, st_coll, y_stack + SBH/2)
            for k in range(stacks):
                xs = xa + (xb - xa) * k / max(stacks - 1, 1)
                _arch_box(ax, xs, y_stack, SBW * 0.85 / max(stacks-1,1) * 1.6, SBH, _AC["stack"], f"S{k+1}")
            sm = y_stack - SBH/2 - 0.014
            _arch_vl(ax, xa, y_stack - SBH/2, sm)
            _arch_vl(ax, xb, y_stack - SBH/2, sm)
            _arch_hl(ax, sm, xa, xb)
            _arch_vl(ax, xez, sm, y_header)
        else:
            _arch_box(ax, xez, y_stack, SBW, SBH, _AC["stack"], "Stk")
            _arch_vl(ax, xez, y_stack - SBH/2, y_header)

    _arch_hl(ax, y_header, x_ez[0], x_ez[-1], lw=1.1)
    ax.text((x_ez[0]+x_ez[-1])/2, y_header - 0.010, "gas header",
            ha="center", va="top", fontsize=5.5, color=_AC["header"], style="italic", zorder=4)

    x_comp = [x0 + span * (j + 0.5) / n_comp for j in range(n_comp)]
    for j, xc in enumerate(x_comp):
        _arch_vl(ax, xc, y_header, y_comp + BH/2)
        _arch_box(ax, xc, y_comp, BW*1.25, BH*1.05, _AC["comp"], f"Comp {j+1}", f"{comp_rate:.2f} kg/h  β=3.0")

    if dedicated_lines:
        lines_per_comp = max(n_fill // max(n_comp, 1), 1)
        for j, xc in enumerate(x_comp):
            cspan = span / n_comp
            cx0, cx1 = xc - cspan/2*0.8, xc + cspan/2*0.8
            x_fl = [cx0 + (cx1-cx0)*(k+0.5)/lines_per_comp for k in range(lines_per_comp)]
            header_y = y_comp - BH*0.6 - (y_comp - BH*0.6 - y_fill)*0.35
            _arch_hl(ax, header_y, x_fl[0], x_fl[-1], lw=0.9)
            _arch_vl(ax, xc, y_comp - BH/2, header_y)
            for xf in x_fl:
                _arch_vl(ax, xf, header_y, y_fill + BH*0.45)
                _arch_box(ax, xf, y_fill, BW*0.62, BH*0.85, _AC["fill"], f"F{j*lines_per_comp+list(x_fl).index(xf)+1}")
        ax.text(x0+span/2, y_fill+BH*0.7, "dedicated lines per compressor",
                ha="center", va="bottom", fontsize=5.3, color=_AC["header"], style="italic", zorder=4)
    else:
        header_y = y_comp - BH*0.6 - (y_comp - BH*0.6 - y_fill)*0.3
        for xc in x_comp:
            _arch_vl(ax, xc, y_comp - BH/2, header_y)
        _arch_hl(ax, header_y, x_comp[0], x_comp[-1], lw=1.1)
        x_fill = [x0 + span*(k+0.5)/n_fill for k in range(n_fill)]
        for xf in x_fill:
            _arch_vl(ax, xf, header_y, y_fill + BH*0.45)
        for k, xf in enumerate(x_fill):
            _arch_box(ax, xf, y_fill, BW*0.62, BH*0.85, _AC["fill"], f"F{k+1}")
        ax.text(x0+span/2, y_fill+BH*0.7, "shared filling header",
                ha="center", va="bottom", fontsize=5.3, color=_AC["header"], style="italic", zorder=4)


def draw_plant_architecture(topology, ax=None, title=True):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    if ax is None:
        n_blocks = len(topology.trains) if topology.mode == "trains" else 1
        width = max(10, 3 * n_blocks)
        fig, ax = plt.subplots(figsize=(width, width * 0.42))
    else:
        fig = ax.figure
    ax.set_facecolor(_AC["bg"])
    fig.patch.set_facecolor(_AC["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0.32, 1.0)
    ax.axis("off")
    if title:
        ax.set_title(f"Hydrogen Plant — Reliability Architecture  ({topology.mode})",
                     fontsize=12.5, fontweight="bold", color=_AC["label"], pad=10)

    Y_TOP, Y_BOTTOM = 0.85, 0.06

    if topology.mode == "trains":
        n_trains = len(topology.trains)
        margin = 0.03
        block_w = (1.0 - margin*(n_trains+1)) / n_trains
        for t_idx, train in enumerate(topology.trains):
            x0 = margin + t_idx*(block_w+margin)
            x1 = x0 + block_w
            topology._tmp_n_ez      = train.n_electrolyzers
            topology._tmp_n_comp    = train.n_compressors
            topology._tmp_n_fill    = train.n_fill_lines
            topology._tmp_stacks    = train.stacks_per_electrolyzer
            topology._tmp_ez_rate   = train.electrolyzer_kg_per_hr_each
            topology._tmp_comp_rate = train.compressor_flow_kg_per_hr_each
            _draw_single_block(ax, topology, x0, x1, Y_TOP, Y_BOTTOM, dedicated_lines=False)
            ax.text((x0+x1)/2, Y_TOP+0.05, train.label, ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=_AC["label"], zorder=5)
        for attr in ("_tmp_n_ez","_tmp_n_comp","_tmp_n_fill","_tmp_stacks","_tmp_ez_rate","_tmp_comp_rate"):
            if hasattr(topology, attr):
                delattr(topology, attr)
    else:
        dedicated = topology.mode == "pooled_ez_dedicated_comp"
        _draw_single_block(ax, topology, 0.06, 0.94, Y_TOP, Y_BOTTOM, dedicated_lines=dedicated)

    legend_items = [
        ("Electrolyzer (Weibull β2.5)", _AC["ez"]),
        ("Stack (Weibull β2.5)", _AC["stack"]),
        ("Compressor (Weibull β3.0)", _AC["comp"]),
        ("Fill line (Exp)", _AC["fill"]),
    ]
    from matplotlib.patches import FancyBboxPatch
    for k, (lbl, col) in enumerate(legend_items):
        bx = 0.02 + k*0.20
        ax.add_patch(FancyBboxPatch((bx, 0.965), 0.185, 0.024,
            boxstyle="round,pad=0", facecolor=col, edgecolor="white", lw=0.5,
            transform=ax.transAxes, clip_on=False, zorder=5))
        ax.text(bx+0.0925, 0.977, lbl, transform=ax.transAxes, ha="center", va="center",
                fontsize=6.2, color="white", fontweight="bold", clip_on=False, zorder=6)

    theo = topology.theoretical_capacity_kg_per_day()
    bneck = "compressors" if topology.compressor_is_bottleneck() else "electrolyzers"
    ax.text(0.5, -0.03,
            f"Theoretical: {theo:.1f} kg/day  (bottleneck: {bneck})  |  "
            f"{topology.total_electrolyzers()} EZ · {topology.total_compressors()} comp · "
            f"{topology.total_fill_lines()} fill lines",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color=_AC["header"], style="italic", zorder=5)
    return fig, ax


def plot_plant_architecture(topology, save_path=None):
    import matplotlib.pyplot as plt
    fig, ax = draw_plant_architecture(topology)
    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_AC["bg"])
    plt.show()


# ============================================================
# Reliability Timeline Plot
# ============================================================

_TIMELINE_BANDS = [
    (1.00, 1.01, "#1D9E75", "100%"),
    (0.75, 1.00, "#639922", "75-99%"),
    (0.50, 0.75, "#EF9F27", "50-74%"),
    (0.25, 0.50, "#D85A30", "25-49%"),
    (0.00, 0.25, "#E24B4A", "0-24%"),
]

def plot_reliability_timeline(timeline_result, topology, save_path=None, title_suffix=None):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import numpy as np

    cap     = np.array(timeline_result["capacity_history"])
    pm_mask = np.array(timeline_result["pm_history"], dtype=bool)
    model   = timeline_result["model"]

    hours_per_year = 8760
    sim_years = len(cap) // hours_per_year
    years = np.arange(len(cap)) / hours_per_year

    theoretical_kg_per_hr = topology.theoretical_capacity_kg_per_hr()
    bottleneck_kg_per_hr  = min(theoretical_kg_per_hr,
                                topology.theoretical_compressor_capacity_kg_per_hr())

    roll     = np.convolve(cap, np.ones(720)/720, mode="same") if len(cap) >= 720 else cap
    yr_avail = [cap[y*hours_per_year:(y+1)*hours_per_year].mean()*100 for y in range(sim_years)]
    overall  = cap.mean() * 100
    zero_mask = cap == 0.0

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#F8F7F4")
    suffix = f"  |  {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"Hydrogen Plant — {sim_years}-Year Availability Timeline  ({topology.mode}){suffix}\n"
        f"Weibull wear-out + staggered planned maintenance",
        fontsize=13, fontweight="bold", color="#2C2C2A", y=0.99
    )
    gs = gridspec.GridSpec(4, 2, figure=fig,
        height_ratios=[3, 0.55, 1.1, 1.3],
        hspace=0.55, wspace=0.30, left=0.07, right=0.97, top=0.93, bottom=0.05)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#F0EFE9")
    for lo, hi, col, _ in _TIMELINE_BANDS:
        filled = np.where((cap >= lo) & (cap < hi + 0.001), cap*100, np.nan)
        ax1.fill_between(years, 0, filled, color=col, alpha=0.60, linewidth=0)
    ax1.plot(years, roll*100, color="#2C2C2A", linewidth=1.1, label="30-day rolling mean", zorder=5)
    ax1.fill_between(years, 0, 108, where=pm_mask, color="#2196F3", alpha=0.12, linewidth=0, label="Planned PM")
    ax1.fill_between(years, 0, 108, where=zero_mask, color="#E24B4A", alpha=0.18, linewidth=0, label="Complete outage")
    for y in range(1, sim_years+1):
        ax1.axvline(y, color="#888780", linewidth=0.4, linestyle="--")
    ax1.set_xlim(0, sim_years); ax1.set_ylim(0, 108)
    ax1.set_xlabel("Year", fontsize=10); ax1.set_ylabel("Plant capacity (%)", fontsize=10)
    ax1.set_title("Hourly Plant Capacity", fontsize=10, color="#444441")
    ax1.set_xticks(range(0, sim_years+1))

    band_patches = [mpatches.Patch(color=c, alpha=0.75, label=l) for _, _, c, l in _TIMELINE_BANDS]
    ax1.legend(handles=band_patches + [
        plt.Line2D([0],[0], color="#2C2C2A", lw=1.4, label="30-day rolling mean"),
        mpatches.Patch(color="#2196F3", alpha=0.25, label="Planned PM"),
        mpatches.Patch(color="#E24B4A", alpha=0.25, label="Complete outage"),
    ], loc="lower left", fontsize=7, ncol=4, framealpha=0.88)

    n_zero = int(zero_mask.sum())
    n_deg  = int(((cap > 0) & (cap < 1)).sum())
    n_pm   = int(pm_mask.sum())
    stats_text = (f"Availability: {overall:.2f}%\n"
                  f"Zero-output: {n_zero:,} h\n"
                  f"Degraded: {n_deg:,} h\n"
                  f"In PM: {n_pm:,} h")
    ax1.text(0.015, 0.96, stats_text, transform=ax1.transAxes, ha="left", va="top",
              fontsize=7.5, color="#2C2C2A",
              bbox=dict(boxstyle="round,pad=0.7", facecolor="white", edgecolor="#D3D1C7", alpha=1.0))

    ax_pm = fig.add_subplot(gs[1, :])
    ax_pm.set_facecolor("#F0EFE9")
    ax_pm.fill_between(years, 0, pm_mask.astype(float)*0.9+0.05,
                        color="#185FA5", alpha=0.6, linewidth=0, label="Any unit in PM")
    for y in range(1, sim_years+1):
        ax_pm.axvline(y, color="#888780", linewidth=0.4, linestyle="--")
    ax_pm.set_xlim(0, sim_years); ax_pm.set_ylim(0, 1)
    ax_pm.set_yticks([]); ax_pm.set_xticks(range(0, sim_years+1))
    ax_pm.set_xlabel("Year", fontsize=9)
    ax_pm.set_title("Planned Maintenance Schedule", fontsize=9, color="#444441")
    ax_pm.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_facecolor("#F0EFE9")
    cum_lost_t = np.cumsum((1.0 - cap) * bottleneck_kg_per_hr) / 1000.0
    cum_pm_t   = np.cumsum(np.where(pm_mask, (1.0-cap)*bottleneck_kg_per_hr, 0.0)) / 1000.0
    cum_corr_t = cum_lost_t - cum_pm_t
    ax2.fill_between(years, 0, cum_lost_t, color="#D85A30", alpha=0.20, linewidth=0)
    ax2.fill_between(years, 0, cum_pm_t, color="#607D8B", alpha=0.35, linewidth=0, label="PM loss")
    ax2.plot(years, cum_corr_t, color="#993C1D", linewidth=1.2, linestyle="--", label="Corrective loss")
    ax2.plot(years, cum_lost_t, color="#993C1D", linewidth=1.8, label="Total loss")
    for y in range(1, sim_years+1):
        ax2.axvline(y, color="#888780", linewidth=0.4, linestyle="--")
    ax2.set_xlim(0, sim_years)
    ax2.set_xlabel("Year", fontsize=10); ax2.set_ylabel("Cum. lost prod. (t H₂)", fontsize=10)
    ax2.set_title("Cumulative Lost Production", fontsize=10, color="#444441")
    ax2.set_xticks(range(0, sim_years+1))
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.annotate(f"  {cum_lost_t[-1]:,.1f} t total", xy=(years[-1], cum_lost_t[-1]),
                  xytext=(-90, -16), textcoords="offset points", fontsize=8.5, color="#993C1D",
                  fontweight="bold", arrowprops=dict(arrowstyle="->", color="#993C1D", lw=0.9))

    ax3 = fig.add_subplot(gs[3, 0])
    ax3.set_facecolor("#F0EFE9")
    bar_colors = ["#1D9E75" if v >= 95 else "#639922" if v >= 85 else "#EF9F27" if v >= 70 else "#D85A30"
                  for v in yr_avail]
    bars = ax3.bar(range(1, sim_years+1), yr_avail, color=bar_colors, edgecolor="white", lw=0.6, width=0.7)
    for bar, val in zip(bars, yr_avail):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                  f"{val:.1f}%", ha="center", va="bottom", fontsize=7, color="#2C2C2A")
    ax3.axhline(overall, color="#2C2C2A", linewidth=1.1, linestyle="--", label=f"Mean {overall:.1f}%")
    ax3.set_ylim(0, 112); ax3.set_xlabel("Year", fontsize=9); ax3.set_ylabel("Availability (%)", fontsize=9)
    ax3.set_title("Yearly Availability", fontsize=10, color="#444441")
    ax3.legend(fontsize=8, framealpha=0.85)
    ax3.grid(axis="y", linestyle="--", alpha=0.35)

    ax4 = fig.add_subplot(gs[3, 1])
    ax4.set_facecolor("#F0EFE9")
    band_pct, band_lbls, band_cols = [], [], []
    for lo, hi, col, lbl in _TIMELINE_BANDS:
        h = int(((cap >= lo) & (cap < hi+0.001)).sum())
        band_pct.append(h/len(cap)*100)
        band_lbls.append(f"{lbl}   {h:,} h")
        band_cols.append(col)
    bars2 = ax4.barh(range(len(_TIMELINE_BANDS)), band_pct, color=band_cols, edgecolor="white", lw=0.6, height=0.6)
    ax4.set_yticks(range(len(_TIMELINE_BANDS)))
    ax4.set_yticklabels(band_lbls, fontsize=8)
    ax4.set_xlabel("% of total hours", fontsize=9)
    ax4.set_title("Capacity Band Distribution", fontsize=10, color="#444441")
    ax4.set_xlim(0, 105)
    ax4.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, pct in zip(bars2, band_pct):
        if pct > 2.5:
            ax4.text(pct-1.0, bar.get_y()+bar.get_height()/2,
                      f"{pct:.1f}%", ha="right", va="center", fontsize=8, color="white", fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#F8F7F4")
    plt.show()
    return fig
