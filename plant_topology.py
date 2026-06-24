# -*- coding: utf-8 -*-
"""
plant_topology.py
==================
Static / theoretical plant definition - the "plan" the operations
simulation starts from, before any availability/failure effects are
applied. This answers: how many electrolyzers, stacks, compressors,
fill lines does the plant have, how are they wired together, and what
is the plant's theoretical (100%-availability) capacity?

Three selectable topologies
----------------------------
"common"  (A - this project's current default)
    Electrolyzers pool into a single capacity budget (a low-pressure
    header, conceptually). ALL compressors draw from that one pooled
    budget. ALL compressors share ALL fill lines - any idle compressor
    can pick up any docked container (today's plant_operations.py
    behaviour, generalised from a hardcoded 2 compressors to N).

"trains"  (B)
    The plant is split into independent parallel trains. Each train has
    its own electrolyzers, compressors and fill lines - zero sharing
    across trains at any stage. Arrivals join ONE shared external queue
    and are routed to whichever train has a free fill-line slot first
    (round-robin/first-available - see plant_operations.route_to_train).

"pooled_ez_dedicated_comp"  (C)
    Electrolyzers pool into a single shared low-pressure-header budget
    (same as "common"), but there is NO shared high-pressure header
    downstream: each compressor has its OWN dedicated fill line(s).
    A container docks at a specific compressor's line and is filled by
    that compressor only - it cannot be picked up by a different
    compressor the way it can under "common".

Stack derate model
-------------------
Within any topology, each electrolyzer has `stacks_per_electrolyzer`
stacks; only 1 needs to be up to keep that electrolyzer producing, but
output is derated proportionally (1-of-2 stacks down = that
electrolyzer makes 50% of its rated kg/hr). This matches RAM_simV2.py's
logic and is handled by reliability.ReliabilityModel - this module only
needs the stack COUNT to size the reliability model correctly.
"""

from dataclasses import dataclass, field


@dataclass
class Train:
    """One independent train, used only when topology mode == 'trains'."""
    label: str
    n_electrolyzers: int
    electrolyzer_kg_per_hr_each: float
    n_compressors: int
    compressor_flow_kg_per_hr_each: float
    n_fill_lines: int
    stacks_per_electrolyzer: int = 2


@dataclass
class PlantTopology:
    """
    Theoretical plant definition. Build one of these, then pass it to
    plant_operations.HydrogenPlant(topology=...) to get a sized,
    multi-electrolyzer/multi-compressor plant, and optionally to
    reliability.ReliabilityModel(topology=...) to get a matching
    availability model.

    For mode="common" / "pooled_ez_dedicated_comp", fill in the
    plant-level fields directly. For mode="trains", fill in `trains`
    instead and leave the plant-level fields at their defaults (they
    are ignored).
    """

    mode: str = "common"   # "common" | "trains" | "pooled_ez_dedicated_comp"

    # --- plant-level fields (mode in {"common", "pooled_ez_dedicated_comp"}) ---
    n_electrolyzers: int = 3
    stacks_per_electrolyzer: int = 2
    electrolyzer_kg_per_hr_each: float = 44.0   # 132 kg/hr / 3 EZ

    n_compressors: int = 2
    compressor_flow_kg_per_hr_each: float = 66.0  # 132 kg/hr / 2 comp

    # mode="common": total shared fill lines.
    # mode="pooled_ez_dedicated_comp": fill lines PER COMPRESSOR (so total
    #   fill lines = n_compressors * n_fill_lines_per_compressor).
    n_fill_lines: int = 4
    n_fill_lines_per_compressor: int = 2

    # --- train-level fields (mode="trains") ---
    trains: list = field(default_factory=list)

    pressure_thresholds: list = field(default_factory=lambda: [94, 278, 500])
    Isentropic_efficiency: float = 0.4

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    def __post_init__(self):
        valid_modes = {"common", "trains", "pooled_ez_dedicated_comp"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode!r}")
        if self.mode == "trains" and not self.trains:
            raise ValueError("mode='trains' requires a non-empty `trains` list")
        if self.mode != "trains" and self.trains:
            raise ValueError("`trains` is only used when mode='trains'")

    # ------------------------------------------------------------
    # Sizing helpers
    # ------------------------------------------------------------

    def total_fill_lines(self) -> int:
        if self.mode == "common":
            return self.n_fill_lines
        if self.mode == "pooled_ez_dedicated_comp":
            return self.n_compressors * self.n_fill_lines_per_compressor
        # trains
        return sum(t.n_fill_lines for t in self.trains)

    def total_electrolyzers(self) -> int:
        if self.mode == "trains":
            return sum(t.n_electrolyzers for t in self.trains)
        return self.n_electrolyzers

    def total_compressors(self) -> int:
        if self.mode == "trains":
            return sum(t.n_compressors for t in self.trains)
        return self.n_compressors

    def theoretical_capacity_kg_per_hr(self) -> float:
        """
        Plant's theoretical (100%-availability) hourly capacity: the
        electrolyzer budget, since that's the hard ceiling regardless of
        how much compressor flow is nominally available downstream.
        Compressor flow CAN bottleneck below this if under-sized versus
        the electrolyzer budget - see compressor_is_bottleneck().
        """
        if self.mode == "trains":
            return sum(t.n_electrolyzers * t.electrolyzer_kg_per_hr_each
                       for t in self.trains)
        return self.n_electrolyzers * self.electrolyzer_kg_per_hr_each

    def theoretical_compressor_capacity_kg_per_hr(self) -> float:
        if self.mode == "trains":
            return sum(t.n_compressors * t.compressor_flow_kg_per_hr_each
                       for t in self.trains)
        return self.n_compressors * self.compressor_flow_kg_per_hr_each

    def compressor_is_bottleneck(self) -> bool:
        return self.theoretical_compressor_capacity_kg_per_hr() < self.theoretical_capacity_kg_per_hr()

    def theoretical_capacity_kg_per_day(self) -> float:
        return min(
            self.theoretical_capacity_kg_per_hr(),
            self.theoretical_compressor_capacity_kg_per_hr(),
        ) * 24.0

    def summary(self) -> str:
        lines = [f"PlantTopology mode={self.mode!r}"]
        if self.mode == "trains":
            for t in self.trains:
                lines.append(
                    f"  Train {t.label}: {t.n_electrolyzers} EZ x "
                    f"{t.electrolyzer_kg_per_hr_each:.2f} kg/hr "
                    f"({t.stacks_per_electrolyzer} stacks/EZ), "
                    f"{t.n_compressors} comp x {t.compressor_flow_kg_per_hr_each:.2f} kg/hr, "
                    f"{t.n_fill_lines} fill lines"
                )
        else:
            lines.append(
                f"  {self.n_electrolyzers} EZ x {self.electrolyzer_kg_per_hr_each:.2f} kg/hr "
                f"({self.stacks_per_electrolyzer} stacks/EZ) = "
                f"{self.theoretical_capacity_kg_per_hr():.2f} kg/hr pooled"
            )
            lines.append(
                f"  {self.n_compressors} compressors x "
                f"{self.compressor_flow_kg_per_hr_each:.2f} kg/hr = "
                f"{self.theoretical_compressor_capacity_kg_per_hr():.2f} kg/hr"
            )
            if self.mode == "common":
                lines.append(f"  {self.n_fill_lines} shared fill lines")
            else:
                lines.append(
                    f"  {self.n_fill_lines_per_compressor} dedicated fill lines "
                    f"per compressor ({self.total_fill_lines()} total)"
                )
        bottleneck = "compressors" if self.compressor_is_bottleneck() else "electrolyzers"
        lines.append(
            f"  Theoretical capacity: {self.theoretical_capacity_kg_per_day():.1f} kg/day "
            f"(bottleneck: {bottleneck})"
        )
        return "\n".join(lines)


# ============================================================
# Convenience presets
# ============================================================

def default_topology_132kg_day() -> PlantTopology:
    """
    This project's current default: 3 electrolyzers (6 stacks total),
    both compressors pooled, common header, shared fill lines.
    """
    return PlantTopology(
        mode="common",
        n_electrolyzers=3,
        stacks_per_electrolyzer=2,
        electrolyzer_kg_per_hr_each=(132  / 3),
        n_compressors=2,
        compressor_flow_kg_per_hr_each=(132 / 2),
        n_fill_lines=4,
    )
