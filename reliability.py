# -*- coding: utf-8 -*-
"""
reliability.py
===============
Generic Reliability / Availability / Maintainability (RAM) engine for the
hydrogen filling plant. Adapted from the standalone RAM_simV2.py prototype:
the failure-model building blocks (ExpUnit, WeibullPMUnit, BOM rollup) are
reused as-is, but the fixed plant topology from that file (4 rectifiers,
8 electrolyzers, 20/80 HV split, etc. - a different, larger project) is
REPLACED by a topology driven from PlantTopology (see plant_topology.py),
so the same engine works for whatever electrolyzer/compressor/fill-line
count and wiring this project's plant actually has.

Two layers
----------
1. Static / theoretical layer
   PlantTopology.theoretical_capacity_kg_per_hr() — what the plant could
   produce with everything up, before any failure model runs at all.
   This is the "plan" the operations simulation should start from.

2. Dynamic / stepped layer
   ReliabilityModel — one WeibullPMUnit per electrolyzer body, one
   WeibullPMUnit per stack (proportional derate: 1-of-2 stacks down on an
   electrolyzer = 50% of THAT electrolyzer's output, not the whole plant),
   one Weibull unit-group per compressor (block/motor/seals, all must be
   up for the compressor to be available - matches RAM_simV2's logic),
   and one ExpUnit per fill line. Auxiliary instrumentation/valves on
   each electrolyzer/compressor are rolled into a single ExpUnit per
   node via the same BOM -> series-reliability approach RAM_simV2 used,
   just with a smaller, project-appropriate BOM.

   model.step(step_index, step_minutes) advances every unit by one
   simulation tick and returns a CapacityState snapshot:
       ez_frac[i]      : float 0..1, fraction of electrolyzer i's rated
                         output currently available (stack derate)
       comp_up[j]       : bool, True if compressor j is fully available
       fill_line_up[k]  : bool, True if fill line k is available

   plant_operations.HydrogenPlant consumes this snapshot each step to
   scale the electrolyzer budget, zero out down compressors, and reduce
   the effective fill-line slot count - see HydrogenPlant.reliability
   in plant_operations.py. When no ReliabilityModel is attached, the
   plant behaves exactly as before (100% theoretical availability,
   unchanged from the pre-RAM-merge code).
"""

import math
import random
from dataclasses import dataclass, field


# ============================================================
# Sampling helpers (unchanged from RAM_simV2.py)
# ============================================================

def exp_sample(mtbf):
    return random.expovariate(1.0 / mtbf)


def weibull_sample(beta, eta):
    """Inverse-CDF Weibull sample."""
    u = random.random()
    return eta * (-math.log(1.0 - u + 1e-12)) ** (1.0 / beta)


# ============================================================
# Unit classes (unchanged from RAM_simV2.py - generic, reusable)
# ============================================================

class ExpUnit:
    """Standard exponential repairable unit."""

    def __init__(self, mtbf, mttr):
        self.mtbf     = mtbf
        self.mttr     = mttr
        self.failed   = False
        self.tte      = exp_sample(mtbf)
        self.failures = 0

    def step(self, dt):
        new_failure = 0
        self.tte -= dt
        if self.tte <= 0:
            if not self.failed:
                self.failed    = True
                self.failures += 1
                new_failure    = 1
                self.tte       = exp_sample(self.mttr)
            else:
                self.failed = False
                self.tte    = exp_sample(self.mtbf)
        return new_failure

    @property
    def up(self):
        return not self.failed


class WeibullPMUnit:
    """
    Repairable unit with:
      - Weibull time-to-failure (wear-out)
      - Exponential corrective repair time
      - Scheduled planned maintenance at fixed intervals
        (PM resets the age - as-good-as-new)

    pm_offset_h: hour within simulation at which first PM is due.
                 Subsequent PMs every pm_interval_h hours.
    Set pm_interval_h=None to disable PM entirely (corrective-only unit).
    """

    def __init__(self, beta, eta, mttr_corrective,
                 pm_interval_h=None, pm_duration_h=0, pm_offset_h=0):
        self.beta        = beta
        self.eta         = eta
        self.mttr_corr   = mttr_corrective
        self.pm_interval = pm_interval_h
        self.pm_duration = pm_duration_h
        self.pm_offset   = pm_offset_h
        self.pm_enabled  = pm_interval_h is not None

        self.age      = 0.0
        self.failed   = False
        self.in_pm    = False
        self.failures = 0
        self.pm_count = 0
        self.sim_hour = 0.0

        self._next_failure = weibull_sample(beta, eta)
        if self.pm_enabled:
            self._next_pm = pm_offset_h if pm_offset_h > 0 else pm_interval_h
        else:
            self._next_pm = float("inf")

    def _reset_age(self):
        self.age           = 0.0
        self._next_failure = weibull_sample(self.beta, self.eta)

    def step(self, dt):
        new_failure = 0
        self.sim_hour += dt

        if self.in_pm:
            self._next_pm -= dt
            if self._next_pm <= 0:
                self.in_pm      = False
                self.pm_count  += 1
                self._reset_age()
                self._next_pm   = self.pm_interval
        elif self.failed:
            self._next_failure -= dt
            if self._next_failure <= 0:
                self.failed   = False
                self._reset_age()
                if self.pm_enabled:
                    self._next_pm = self.pm_interval
        else:
            self.age           += dt
            self._next_failure -= dt
            if self.pm_enabled:
                self._next_pm -= dt

            if self.pm_enabled and self._next_pm <= 0 and self._next_failure > 0:
                self.in_pm    = True
                self._next_pm = self.pm_duration
            elif self._next_failure <= 0:
                self.failed         = True
                self.failures      += 1
                new_failure         = 1
                self._next_failure  = exp_sample(self.mttr_corr)

        return new_failure

    @property
    def up(self):
        return not self.failed and not self.in_pm

    @property
    def in_planned_maintenance(self):
        return self.in_pm


# ============================================================
# Project-appropriate BOM / equipment library
# ============================================================
# Smaller, scoped to the 132 kg/day filling-station class of plant - NOT
# the larger multi-rectifier plant RAM_simV2.py modelled. Edit freely;
# these are reasonable generic defaults for small skid-mounted equipment.

EQ_LIB = {
    "pressure_tx":      {"mtbf": 120_000, "mttr": 48},
    "temp_tx":          {"mtbf": 120_000, "mttr": 48},
    "flow_tx":          {"mtbf": 100_000, "mttr": 72},
    "control_valve":    {"mtbf": 100_000, "mttr": 48},
    "manual_valve":     {"mtbf": 200_000, "mttr": 48},
    "check_valve":      {"mtbf": 180_000, "mttr": 48},
    "solenoid":         {"mtbf":  80_000, "mttr": 48},
    "vibration_sensor": {"mtbf":  80_000, "mttr": 48},
    "lube_oil_pump":    {"mtbf":  35_000, "mttr": 72},
    "fan":              {"mtbf":  40_000, "mttr": 48},
    "plc_module":       {"mtbf": 150_000, "mttr": 48},
}

# Auxiliary instrumentation/valves bolted onto each major node (exponential,
# rolled up to one equivalent ExpUnit per node via bom_to_reliability()).
BOM = {
    "ez_aux": {
        "control_valve": 2,
        "pressure_tx":   2,
        "temp_tx":       2,
    },
    "comp_aux": {
        "control_valve":    2,
        "lube_oil_pump":    1,
        "fan":              2,
        "pressure_tx":      2,
        "temp_tx":          2,
        "vibration_sensor": 1,
    },
    "fill_line_aux": {
        "manual_valve":  2,
        "control_valve": 1,
        "pressure_tx":   1,
        "flow_tx":       1,
        "check_valve":   1,
    },
}


def bom_to_reliability(bom_name):
    """Series reliability rollup: returns (equivalent_mtbf, equivalent_mttr)."""
    bom = BOM[bom_name]
    lam_total, weighted_mttr = 0.0, 0.0
    for eq_type, qty in bom.items():
        lib = EQ_LIB[eq_type]
        lam = qty / lib["mtbf"]
        lam_total     += lam
        weighted_mttr += lam * lib["mttr"]
    return 1.0 / lam_total, weighted_mttr / lam_total


NODE_RELIABILITY = {name: bom_to_reliability(name) for name in BOM}


# ============================================================
# Wear-out (Weibull) defaults for major components
# ============================================================
# Edit freely - these drive the realistic long-run availability of the
# plant's core rotating/electrochemical equipment. Smaller skid-class
# units than RAM_simV2.py's bigger reference plant, but same shape.

DEFAULT_RELIABILITY_PARAMS = {
    "electrolyzer_body": dict(beta=2.5, eta=50_000, mttr_corrective=72),
    "stack":              dict(beta=2.5, eta=55_000, mttr_corrective=168),
    "compressor_block":   dict(beta=3.0, eta=40_000, mttr_corrective=240),
    "compressor_motor":   dict(beta=3.0, eta=40_000, mttr_corrective=168),
    "compressor_seals":   dict(beta=3.0, eta=20_000, mttr_corrective=48),
}

# Planned-maintenance defaults (annual, staggered across groups). Set
# pm_interval_h=None on a unit to disable PM for it.
DEFAULT_PM = {
    "interval_h":          8760,    # annual
    "electrolyzer_duration_h": 72,
    "compressor_duration_h":   96,
}


# ============================================================
# Capacity snapshot returned by ReliabilityModel.step()
# ============================================================

@dataclass
class CapacityState:
    ez_frac: list        # fraction 0..1 of rated output per electrolyzer
    comp_up: list        # bool per compressor
    fill_line_up: list   # bool per fill line
    any_pm: bool = False

    def total_ez_frac(self):
        """Sum of available electrolyzer fractions (in 'electrolyzer units'),
        for callers that want a single pooled-capacity multiplier."""
        return sum(self.ez_frac)

    def n_fill_lines_up(self):
        return sum(1 for u in self.fill_line_up if u)


# ============================================================
# Reliability model - built from a PlantTopology
# ============================================================

class ReliabilityModel:
    """
    Steps a Weibull/Exponential failure model for every electrolyzer
    (body + stacks + aux), compressor (block + motor + seals + aux), and
    fill line (aux only - fill lines are simple exponential equipment by
    default) in the plant, given a PlantTopology.

    Usage
    -----
        topology = PlantTopology(...)
        model    = ReliabilityModel(topology, random_seed=17)
        ...
        state = model.step(step, step_minutes)
        # state.ez_frac, state.comp_up, state.fill_line_up

    Planned maintenance is staggered across electrolyzers/compressors in
    round-robin groups so the plant is never fully down for PM at once,
    mirroring RAM_simV2.py's staggered-window approach but generalised to
    however many units the topology actually has.
    """

    def __init__(self, topology, random_seed: int = None,
                 reliability_params: dict = None, pm_config: dict = None,
                 enable_pm: bool = True,
                 eq_lib: dict = None, bom: dict = None):
        if random_seed is not None:
            random.seed(random_seed)

        params = DEFAULT_RELIABILITY_PARAMS if reliability_params is None \
            else {**DEFAULT_RELIABILITY_PARAMS, **reliability_params}
        pm = DEFAULT_PM if pm_config is None else {**DEFAULT_PM, **pm_config}

        # Resolve equipment library and BOM — use module-level defaults if not
        # supplied, so the caller can override either or both independently.
        _eq_lib = EQ_LIB if eq_lib is None else {**EQ_LIB, **eq_lib}
        _bom    = BOM    if bom    is None else {**BOM,    **bom}

        def _node_reliability(bom_name):
            """Series BOM rollup using the resolved eq_lib."""
            b = _bom[bom_name]
            lam_total, weighted_mttr = 0.0, 0.0
            for eq_type, qty in b.items():
                lib = _eq_lib[eq_type]
                lam = qty / lib["mtbf"]
                lam_total     += lam
                weighted_mttr += lam * lib["mttr"]
            return 1.0 / lam_total, weighted_mttr / lam_total

        self.topology = topology
        n_ez   = topology.total_electrolyzers()
        n_comp = topology.total_compressors()
        n_fill = topology.total_fill_lines()

        # Per-electrolyzer stack count. In "common"/"pooled_ez_dedicated_comp"
        # every electrolyzer shares topology.stacks_per_electrolyzer. In
        # "trains", each Train can specify its own stacks_per_electrolyzer,
        # so build the per-electrolyzer list by walking the trains in the
        # same order plant_operations.HydrogenPlant does (train order, then
        # electrolyzer order within each train) - this MUST stay in lockstep
        # with that ordering, since ez_frac[i] is matched positionally
        # against HydrogenPlant.electrolyzer_rated_kg_per_hr[i].
        if topology.mode == "trains":
            stacks_per_ez_list = []
            for train in topology.trains:
                stacks_per_ez_list.extend([train.stacks_per_electrolyzer] * train.n_electrolyzers)
        else:
            stacks_per_ez_list = [topology.stacks_per_electrolyzer] * n_ez

        pm_interval = pm["interval_h"] if enable_pm else None

        # ---- Electrolyzers: body + stacks + aux, staggered PM offset ----
        self.ez_bodies = []
        self.ez_stacks = []   # list[list[unit]] - stacks_per_ez per electrolyzer
        self.ez_aux    = []
        for i in range(n_ez):
            # Round-robin into 2 PM groups (group A offset 0, group B
            # offset half the interval) so EZ PM is staggered, same idea
            # as RAM_simV2's Group A/B windows.
            pm_off = 0 if (i % 2 == 0) else pm["interval_h"] / 2
            self.ez_bodies.append(WeibullPMUnit(
                **params["electrolyzer_body"],
                pm_interval_h=pm_interval,
                pm_duration_h=pm["electrolyzer_duration_h"],
                pm_offset_h=pm_off,
            ))
            self.ez_stacks.append([
                WeibullPMUnit(
                    **params["stack"],
                    pm_interval_h=pm_interval,
                    pm_duration_h=pm["electrolyzer_duration_h"],
                    pm_offset_h=pm_off,
                )
                for _ in range(stacks_per_ez_list[i])
            ])
            self.ez_aux.append(ExpUnit(*_node_reliability("ez_aux")))

        # ---- Compressors: block + motor + seals + aux, staggered PM ----
        self.comp_block = []
        self.comp_motor = []
        self.comp_seals = []
        self.comp_aux   = []
        for j in range(n_comp):
            pm_off = 0 if (j % 2 == 0) else pm["interval_h"] / 2
            self.comp_block.append(WeibullPMUnit(
                **params["compressor_block"],
                pm_interval_h=pm_interval,
                pm_duration_h=pm["compressor_duration_h"],
                pm_offset_h=pm_off,
            ))
            self.comp_motor.append(WeibullPMUnit(
                **params["compressor_motor"],
                pm_interval_h=pm_interval,
                pm_duration_h=pm["compressor_duration_h"],
                pm_offset_h=pm_off,
            ))
            self.comp_seals.append(WeibullPMUnit(
                **params["compressor_seals"],
                pm_interval_h=pm_interval,
                pm_duration_h=pm["compressor_duration_h"],
                pm_offset_h=pm_off,
            ))
            self.comp_aux.append(ExpUnit(*_node_reliability("comp_aux")))

        # ---- Fill lines: aux only (simple exponential equipment) ----
        self.fill_lines = [ExpUnit(*_node_reliability("fill_line_aux")) for _ in range(n_fill)]

        self.total_failures = 0
        self.pm_events       = 0
        self._was_in_pm      = False

    # ----------------------------------------------------------

    def step(self, step_index: int, step_minutes: float) -> CapacityState:
        dt_h = step_minutes / 60.0

        ez_frac = []
        for i in range(len(self.ez_bodies)):
            self.ez_bodies[i].step(dt_h)
            stacks = self.ez_stacks[i]
            stacks_up = 0
            for s in stacks:
                s.step(dt_h)
                if s.up:
                    stacks_up += 1
            aux_fail = self.ez_aux[i].step(dt_h)

            if not self.ez_bodies[i].up or not self.ez_aux[i].up or stacks_up == 0:
                ez_frac.append(0.0)
            else:
                ez_frac.append(stacks_up / len(stacks))

        comp_up = []
        for j in range(len(self.comp_block)):
            b = self.comp_block[j].step(dt_h)
            m = self.comp_motor[j].step(dt_h)
            s = self.comp_seals[j].step(dt_h)
            a = self.comp_aux[j].step(dt_h)
            comp_up.append(
                self.comp_block[j].up and self.comp_motor[j].up
                and self.comp_seals[j].up and self.comp_aux[j].up
            )

        fill_line_up = []
        for u in self.fill_lines:
            u.step(dt_h)
            fill_line_up.append(u.up)

        any_pm = (
            any(b.in_planned_maintenance for b in self.ez_bodies) or
            any(b.in_planned_maintenance for b in self.comp_block)
        )

        return CapacityState(
            ez_frac=ez_frac,
            comp_up=comp_up,
            fill_line_up=fill_line_up,
            any_pm=any_pm,
        )

    # ----------------------------------------------------------

    def summary(self):
        """Lightweight failure/PM tallies, useful for KPI reporting."""
        ez_failures   = sum(b.failures for b in self.ez_bodies)
        stack_failures = sum(s.failures for stacks in self.ez_stacks for s in stacks)
        comp_failures = (
            sum(u.failures for u in self.comp_block)
            + sum(u.failures for u in self.comp_motor)
            + sum(u.failures for u in self.comp_seals)
        )
        fill_failures = sum(u.failures for u in self.fill_lines)
        pm_events = (
            sum(b.pm_count for b in self.ez_bodies)
            + sum(b.pm_count for b in self.comp_block)
        )
        return {
            "electrolyzer_failures": ez_failures,
            "stack_failures":        stack_failures,
            "compressor_failures":   comp_failures,
            "fill_line_failures":    fill_failures,
            "pm_events":             pm_events,
        }


# ============================================================
# Standalone reliability timeline (no operations/queueing sim)
# ============================================================

def run_reliability_timeline(topology, years: int = 10, random_seed: int = None,
                              reliability_params: dict = None, pm_config: dict = None,
                              enable_pm: bool = True,
                              eq_lib: dict = None, bom: dict = None):
    """
    Step a ReliabilityModel hourly for `years` years and return the plant
    CAPACITY history (fraction 0..1 of theoretical kg/hr) and a PM-active
    history - the same shape of data RAM_simV2.simulate_plant() produced,
    but driven by THIS project's PlantTopology instead of that file's
    fixed 8-electrolyzer/4-rectifier plant. Used for the standalone
    availability timeline / architecture plots (see plot_reliability_timeline
    in results_plant_operations.py), independent of the container
    docking/queueing simulation in plant_operations.run_simulation().

    Plant capacity per hour is computed the same bottleneck logic
    HydrogenPlant uses live: pooled electrolyzer output (sum of
    ez_frac * rated kg/hr) capped by the sum of UP compressors' flow
    ("common" / "pooled_ez_dedicated_comp" - dedicated fill lines only
    constrain WHICH container a compressor can pick up, not the
    aggregate kg/hr ceiling, so both modes share this formula). "trains"
    sums each train's own independently-bottlenecked output, correctly
    excluding that train's down compressors from its ceiling.

    Returns
    -------
    dict with:
        capacity_history : list[float], length years*8760, fraction 0..1
        pm_history        : list[bool],  length years*8760
        model              : the ReliabilityModel used (for .summary())
    """
    model = ReliabilityModel(
        topology, random_seed=random_seed,
        reliability_params=reliability_params, pm_config=pm_config,
        enable_pm=enable_pm, eq_lib=eq_lib, bom=bom,
    )

    total_hours = years * 8760
    theoretical_kg_per_hr = max(topology.theoretical_capacity_kg_per_hr(), 1e-9)

    if topology.mode == "trains":
        # Positional maps, built in the same train-then-electrolyzer /
        # train-then-compressor order ReliabilityModel and HydrogenPlant
        # both use, so state.ez_frac[i] / state.comp_up[j] line up.
        ez_train_of = []
        comp_train_of = []
        ez_rate_of = []
        comp_rate_of = []
        for t_idx, train in enumerate(topology.trains):
            ez_train_of.extend([t_idx] * train.n_electrolyzers)
            ez_rate_of.extend([train.electrolyzer_kg_per_hr_each] * train.n_electrolyzers)
            comp_train_of.extend([t_idx] * train.n_compressors)
            comp_rate_of.extend([train.compressor_flow_kg_per_hr_each] * train.n_compressors)
        n_trains = len(topology.trains)
    else:
        ez_rate = topology.electrolyzer_kg_per_hr_each
        comp_rate = topology.compressor_flow_kg_per_hr_each

    capacity_history = []
    pm_history = []

    for h in range(total_hours):
        state = model.step(h, step_minutes=60.0)

        if topology.mode == "trains":
            produced = 0.0
            for t_idx in range(n_trains):
                ez_kg_t = sum(
                    f * rate for f, rate, tt in zip(state.ez_frac, ez_rate_of, ez_train_of)
                    if tt == t_idx
                )
                comp_flow_t = sum(
                    rate for up, rate, tt in zip(state.comp_up, comp_rate_of, comp_train_of)
                    if tt == t_idx and up
                )
                produced += min(ez_kg_t, comp_flow_t)
            capacity_frac = produced / theoretical_kg_per_hr
        else:
            ez_kg = sum(f * ez_rate for f in state.ez_frac)
            comp_flow = sum(comp_rate for up in state.comp_up if up)
            produced = min(ez_kg, comp_flow)
            capacity_frac = produced / theoretical_kg_per_hr

        capacity_history.append(min(max(capacity_frac, 0.0), 1.0))
        pm_history.append(state.any_pm)

    return {
        "capacity_history": capacity_history,
        "pm_history":       pm_history,
        "model":            model,
    }
