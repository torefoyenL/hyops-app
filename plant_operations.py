import random
import uuid
from collections import deque
from dataclasses import dataclass


@dataclass
class ContainerType:
    name: str
    max_capacity_kg: float
    max_fill_rate_kg_per_hr: float
    fleet_fraction: float


def adiabatic_compression_energy(mass_kg, P1_bar=30, P2_bar=380, T=298,
                                  Isentropic_efficiency=0.4):
    """Returns energy in kWh to compress H2 mass from P1 to P2 adiabatically."""
    if mass_kg <= 0:
        return 0.0
    gamma = 1.41
    R     = 8.314
    M     = 0.002016
    n     = mass_kg / M
    P1    = P1_bar * 1e5
    P2    = P2_bar * 1e5
    work_joules = (n * R * T) / (gamma - 1) * ((P2 / P1)**((gamma - 1) / gamma) - 1)
    return (work_joules / 3.6e6) / Isentropic_efficiency


class Container:
    def __init__(self, container_type, arrival_step):
        self.container_type   = container_type
        self.name             = container_type.name
        self.max_capacity_kg  = container_type.max_capacity_kg
        self.max_fill_rate_kg_per_hr = container_type.max_fill_rate_kg_per_hr
        self.arrival_step     = arrival_step
        self.dock_step        = None
        self.start_fill_step  = None
        self.completion_step  = None
        self.compressor_id    = None
        self.filled_kg        = 0.0

    def required_fill(self):
        return self.max_capacity_kg - self.filled_kg

    @property
    def full(self):
        return self.filled_kg >= self.max_capacity_kg - 0.001


class Compressor:
    def __init__(self, compressor_id, max_flow_kg_per_hr,
                 pressure_thresholds=None, Isentropic_efficiency=0.4, step_minutes=1.0):
        self.compressor_id      = compressor_id
        self.max_flow_kg_per_hr = max_flow_kg_per_hr
        self.max_flow_per_step  = max_flow_kg_per_hr / (60 / step_minutes)
        self.pressure_thresholds = pressure_thresholds or [94, 278, 500]
        self.Isentropic_efficiency = Isentropic_efficiency
        self.step_minutes       = step_minutes
        self.active             = None
        self.total_dispensed    = 0.0
        self.reliability_up     = True

    def compression_energy_kwh(self, mass_kg, pressure_bar):
        return adiabatic_compression_energy(mass_kg, P2_bar=pressure_bar,
                                             Isentropic_efficiency=self.Isentropic_efficiency) \
               if mass_kg > 0 else 0.0

    def fill_step(self, electrolyzer_budget_kg):
        if not self.reliability_up:
            return 0.0, 30.0
        if self.active is None or electrolyzer_budget_kg <= 0:
            return 0.0, 30.0
        dispensed = min(self.max_flow_per_step, electrolyzer_budget_kg,
                        self.active.required_fill())
        self.active.filled_kg    += dispensed
        self.total_dispensed     += dispensed
        pressure = self.pressure_thresholds[-1]
        for p in self.pressure_thresholds:
            ratio = self.active.filled_kg / self.active.max_capacity_kg
            if ratio < p / self.pressure_thresholds[-1]:
                pressure = p
                break
        return dispensed, pressure


class HydrogenPlant:
    def __init__(self, electrolyzer_kg_per_hr=None, compressor_flow_kg_per_hr=None,
                 n_fill_lines=6, pressure_thresholds=None, Isentropic_efficiency=0.4,
                 electrolyzer_energy_kwh_per_kg=0.0, base_power_kw=0.0,
                 step_minutes=1.0, topology=None, reliability_model=None):
        self.step_minutes = step_minutes
        self.electrolyzer_energy_kwh_per_kg = electrolyzer_energy_kwh_per_kg
        self.base_power_kw = base_power_kw
        self.Isentropic_efficiency = Isentropic_efficiency
        self.reliability_model = reliability_model
        if pressure_thresholds is None:
            pressure_thresholds = [94, 278, 500]

        if topology is None:
            assert electrolyzer_kg_per_hr is not None
            if compressor_flow_kg_per_hr is None:
                half = electrolyzer_kg_per_hr / 2.0
                compressor_flow_kg_per_hr = [half, half]
            self.mode = "common"
            self.electrolyzer_rated_kg_per_hr = [electrolyzer_kg_per_hr]
            self.n_fill_lines = n_fill_lines
            self.trains = None
            self.compressors = [
                Compressor(i, compressor_flow_kg_per_hr[i],
                           pressure_thresholds, Isentropic_efficiency, step_minutes)
                for i in range(len(compressor_flow_kg_per_hr))
            ]
            self.comp_fill_line_ids = None
            self.train_of_compressor = None
            self.train_of_electrolyzer = None
        else:
            self.mode = topology.mode
            self.topology = topology
            if topology.mode == "common":
                self.electrolyzer_rated_kg_per_hr = [
                    topology.electrolyzer_kg_per_hr_each] * topology.n_electrolyzers
                self.train_of_electrolyzer = None
                self.n_fill_lines = topology.n_fill_lines
                self.trains = None
                self.compressors = [
                    Compressor(i, topology.compressor_flow_kg_per_hr_each,
                               topology.pressure_thresholds or pressure_thresholds,
                               topology.Isentropic_efficiency, step_minutes)
                    for i in range(topology.n_compressors)
                ]
                self.comp_fill_line_ids = None
                self.train_of_compressor = None
            elif topology.mode == "pooled_ez_dedicated_comp":
                self.electrolyzer_rated_kg_per_hr = [
                    topology.electrolyzer_kg_per_hr_each] * topology.n_electrolyzers
                self.train_of_electrolyzer = None
                self.n_fill_lines = topology.total_fill_lines()
                self.trains = None
                self.compressors = [
                    Compressor(i, topology.compressor_flow_kg_per_hr_each,
                               topology.pressure_thresholds or pressure_thresholds,
                               topology.Isentropic_efficiency, step_minutes)
                    for i in range(topology.n_compressors)
                ]
                per = topology.n_fill_lines_per_compressor
                self.comp_fill_line_ids = {
                    i: list(range(i * per, (i + 1) * per))
                    for i in range(topology.n_compressors)
                }
                self.train_of_compressor = None
            elif topology.mode == "trains":
                self.electrolyzer_rated_kg_per_hr = []
                self.compressors = []
                self.comp_fill_line_ids = {}
                self.train_of_compressor = {}
                self.train_of_electrolyzer = []
                self.train_fill_line_ids = []
                self.train_n_fill_lines = []
                comp_id = 0
                slot_cursor = 0
                for t_idx, train in enumerate(topology.trains):
                    self.electrolyzer_rated_kg_per_hr.extend(
                        [train.electrolyzer_kg_per_hr_each] * train.n_electrolyzers)
                    self.train_of_electrolyzer.extend([t_idx] * train.n_electrolyzers)
                    train_slots = list(range(slot_cursor, slot_cursor + train.n_fill_lines))
                    self.train_fill_line_ids.append(train_slots)
                    self.train_n_fill_lines.append(train.n_fill_lines)
                    slot_cursor += train.n_fill_lines
                    for _ in range(train.n_compressors):
                        self.compressors.append(Compressor(
                            comp_id, train.compressor_flow_kg_per_hr_each,
                            topology.pressure_thresholds or pressure_thresholds,
                            topology.Isentropic_efficiency, step_minutes))
                        self.comp_fill_line_ids[comp_id] = train_slots
                        self.train_of_compressor[comp_id] = t_idx
                        comp_id += 1
                self.n_fill_lines = slot_cursor
                self.trains = topology.trains

        self.external_queue  = deque()
        self.shared_docked   = []
        self.total_dispensed = 0.0
        self.total_production = sum(self.electrolyzer_rated_kg_per_hr) / (60 / step_minutes)
        self.max_parallel_fills = self.n_fill_lines
        self._ez_frac       = [1.0] * len(self.electrolyzer_rated_kg_per_hr)
        self._fill_line_up  = [True] * self.n_fill_lines
        self.electrolyzer_step_kg = self._compute_electrolyzer_step_kg()

        if self.trains is not None:
            self.fill_line_owner_train = {}
            for t_idx, slots in enumerate(self.train_fill_line_ids):
                for s in slots:
                    self.fill_line_owner_train[s] = t_idx
        else:
            self.fill_line_owner_train = None

    def apply_reliability_state(self, state):
        self._ez_frac = list(state.ez_frac)
        for comp, up in zip(self.compressors, state.comp_up):
            comp.reliability_up = up
        self._fill_line_up = list(state.fill_line_up)
        self.electrolyzer_step_kg = self._compute_electrolyzer_step_kg()

    def _compute_electrolyzer_step_kg(self):
        step_factor = 60 / self.step_minutes
        if self.trains is None:
            return sum(r * f for r, f in zip(
                self.electrolyzer_rated_kg_per_hr, self._ez_frac)) / step_factor
        budgets = {}
        for t_idx in range(len(self.train_fill_line_ids)):
            budgets[t_idx] = sum(
                r * f for r, f, t in zip(
                    self.electrolyzer_rated_kg_per_hr,
                    self._ez_frac, self.train_of_electrolyzer)
                if t == t_idx) / step_factor
        return budgets

    def n_filling(self): return sum(1 for c in self.compressors if c.active is not None)
    def n_docked(self): return len(self.shared_docked)
    def n_fill_lines_occupied(self): return self.n_docked() + self.n_filling()
    def n_fill_lines_up(self): return sum(1 for u in self._fill_line_up if u)
    def fill_line_slots_free(self): return self.n_fill_lines_up() - self.n_fill_lines_occupied()
    def n_external(self): return len(self.external_queue)

    def dock_from_external(self, step):
        if self.trains is not None:
            self._dock_from_external_trains(step)
        elif self.comp_fill_line_ids is not None:
            self._dock_from_external_dedicated(step)
        else:
            while self.fill_line_slots_free() > 0 and self.external_queue:
                c = self.external_queue.popleft()
                c.dock_step = step
                self.shared_docked.append((c, None))

    def _slot_pool_free(self, slot_ids):
        occupied = sum(1 for c, a in self.shared_docked
                       if a is not None and set(a) == set(slot_ids))
        occupied += sum(1 for comp in self.compressors
                        if comp.active is not None
                        and self.comp_fill_line_ids.get(comp.compressor_id) == slot_ids)
        return sum(1 for s in slot_ids if self._fill_line_up[s]) - occupied

    def _dock_from_external_dedicated(self, step):
        progressed = True
        while progressed and self.external_queue:
            progressed = False
            for cid in sorted(self.comp_fill_line_ids, key=self._dedicated_pool_load):
                slot_ids = self.comp_fill_line_ids[cid]
                if self._slot_pool_free(slot_ids) > 0 and self.external_queue:
                    c = self.external_queue.popleft()
                    c.dock_step = step
                    self.shared_docked.append((c, slot_ids))
                    progressed = True
                    break

    def _dedicated_pool_load(self, comp_id):
        slot_ids = self.comp_fill_line_ids[comp_id]
        return (sum(1 for c, a in self.shared_docked
                    if a is not None and set(a) == set(slot_ids))
                + (1 if self.compressors[comp_id].active is not None else 0))

    def _dock_from_external_trains(self, step):
        progressed = True
        while progressed and self.external_queue:
            progressed = False
            for t_idx in sorted(range(len(self.train_fill_line_ids)),
                                  key=self._train_load):
                slot_ids = self.train_fill_line_ids[t_idx]
                if self._slot_pool_free(slot_ids) > 0 and self.external_queue:
                    c = self.external_queue.popleft()
                    c.dock_step = step
                    self.shared_docked.append((c, slot_ids))
                    progressed = True
                    break

    def _train_load(self, t_idx):
        slot_ids = self.train_fill_line_ids[t_idx]
        return (sum(1 for c, a in self.shared_docked
                    if a is not None and set(a) == set(slot_ids))
                + sum(1 for comp in self.compressors
                      if comp.active is not None
                      and self.train_of_compressor[comp.compressor_id] == t_idx))

    def assign_idle_compressors(self, step):
        for comp in self.compressors:
            if comp.active is not None or not comp.reliability_up:
                continue
            if self.comp_fill_line_ids is None:
                eligible = [c for c, a in self.shared_docked]
            else:
                my_slots = set(self.comp_fill_line_ids[comp.compressor_id])
                eligible = [c for c, a in self.shared_docked
                             if a is not None and set(a) == my_slots]
            if not eligible:
                continue
            next_c = min(eligible, key=lambda c: c.required_fill())
            self.shared_docked = [(c, a) for c, a in self.shared_docked if c is not next_c]
            next_c.start_fill_step = step
            next_c.compressor_id   = comp.compressor_id
            comp.active = next_c

    def admit_all(self, step):
        self.dock_from_external(step)
        self.assign_idle_compressors(step)

    def fill_containers(self):
        base_energy = self.base_power_kw * (self.step_minutes / 60)
        def active_remaining(comp):
            return comp.active.required_fill() if comp.active else float("inf")
        priority_order = sorted(self.compressors, key=active_remaining)
        total_dispensed_step = 0.0
        pressures  = []
        energy_kwh = base_energy
        if self.trains is None:
            remaining_by_group = {None: self.electrolyzer_step_kg}
            group_of_comp = {c.compressor_id: None for c in self.compressors}
        else:
            remaining_by_group = dict(self.electrolyzer_step_kg)
            group_of_comp = dict(self.train_of_compressor)
        for comp in priority_order:
            group     = group_of_comp[comp.compressor_id]
            remaining = remaining_by_group[group]
            budget    = min(comp.max_flow_per_step, remaining)
            dispensed, pressure = comp.fill_step(budget)
            remaining_by_group[group] = max(remaining - dispensed, 0.0)
            if dispensed > 0:
                energy_kwh += (comp.compression_energy_kwh(dispensed, pressure)
                               + dispensed * self.electrolyzer_energy_kwh_per_kg)
                pressures.append(pressure)
            total_dispensed_step += dispensed
        self.total_dispensed += total_dispensed_step
        return max(pressures) if pressures else 30.0, energy_kwh

    def collect_completed(self, step):
        done = []
        for comp in self.compressors:
            if comp.active is not None and comp.active.full:
                comp.active.completion_step = step
                done.append(comp.active)
                comp.active = None
        return done

    def _reset(self):
        self.total_dispensed = 0.0
        self.external_queue.clear()
        self.shared_docked.clear()
        for comp in self.compressors:
            comp.total_dispensed = 0.0
            comp.active = None
            comp.reliability_up = True
        self._ez_frac = [1.0] * len(self.electrolyzer_rated_kg_per_hr)
        self._fill_line_up = [True] * self.n_fill_lines
        self.electrolyzer_step_kg = self._compute_electrolyzer_step_kg()


def choose_container_type(types):
    r = random.random()
    cumulative = 0.0
    for t in types:
        cumulative += t.fleet_fraction
        if r <= cumulative:
            return t
    return types[-1]


class StaffSchedule:
    def __init__(self, manned=True, weekday_hours="8-20", weekend="open"):
        self.manned = manned
        if not manned:
            self._weekday = (0, 24)
            self._weekend_open = True
        else:
            if weekday_hours == "24/7":
                self._weekday = (0, 24)
            else:
                start, end = weekday_hours.split("-")
                self._weekday = (int(start), int(end))
            self._weekend_open = (weekend == "open")

    def is_manned(self, step, steps_per_day):
        if not self.manned:
            return False
        day  = step // steps_per_day
        frac = (step % steps_per_day) / steps_per_day
        hour = frac * 24
        is_weekend = (day % 7) in (5, 6)
        if is_weekend:
            return self._weekend_open
        return self._weekday[0] <= hour < self._weekday[1]


class ArrivalPattern:
    def __init__(self, pattern_type="uniform", peak_hour=8.0, peak_hour_2=16.0,
                 peak_width_hours=3.0, peak_weight=0.5):
        self.pattern_type      = pattern_type
        self.peak_hour         = peak_hour
        self.peak_hour_2       = peak_hour_2
        self.peak_width_hours  = peak_width_hours
        self.peak_weight       = peak_weight

    def rate_at_hour(self, hour):
        import numpy as np
        if self.pattern_type == "uniform":
            return 1.0
        def gaussian(h, mu, sigma):
            return np.exp(-0.5 * ((h - mu) / sigma) ** 2)
        sigma = self.peak_width_hours / 2.355
        if self.pattern_type == "single_peak":
            return gaussian(hour, self.peak_hour, sigma) + 0.05
        w = self.peak_weight
        return (w * gaussian(hour, self.peak_hour, sigma)
                + (1 - w) * gaussian(hour, self.peak_hour_2, sigma) + 0.05)


def run_simulation(container_types, plant, avg_arrivals_per_day, days, step_minutes,
                   random_seed=17, schedule=None, arrival_pattern=None,
                   reliability_model=None):
    import numpy as np
    random.seed(random_seed)
    np.random.seed(random_seed)

    if schedule is None:
        schedule = StaffSchedule(manned=True)
    if arrival_pattern is None:
        arrival_pattern = ArrivalPattern(pattern_type="uniform")

    N_step_day          = int(24 * 60 / step_minutes)
    TIMESTEPS           = N_step_day * days
    base_lambda         = avg_arrivals_per_day / N_step_day

    plant._reset()

    completed = []
    external_queue_log = []
    docked_log         = []
    filling_log        = []
    pressure_log       = []
    production_log     = []
    energy_log         = []
    arrival_log        = []
    manned_log         = []
    ez_availability_log  = []
    comp_up_log          = []
    fill_lines_up_log    = []
    pm_active_log        = []

    comp_filling_log = [[] for _ in plant.compressors]

    for step in range(TIMESTEPS):
        if reliability_model is not None:
            cap_state = reliability_model.step(step, step_minutes)
            plant.apply_reliability_state(cap_state)
            n_ez = max(len(cap_state.ez_frac), 1)
            ez_availability_log.append(sum(cap_state.ez_frac) / n_ez)
            comp_up_log.append(list(cap_state.comp_up))
            fill_lines_up_log.append(cap_state.n_fill_lines_up())
            pm_active_log.append(cap_state.any_pm)
        else:
            ez_availability_log.append(1.0)
            comp_up_log.append([True] * len(plant.compressors))
            fill_lines_up_log.append(plant.n_fill_lines)
            pm_active_log.append(False)

        is_manned = schedule.is_manned(step, N_step_day)
        manned_log.append(int(is_manned))

        hour = ((step % N_step_day) / N_step_day) * 24
        rate_multiplier = arrival_pattern.rate_at_hour(hour)
        lam = base_lambda * rate_multiplier
        n_arrivals = np.random.poisson(lam)
        for _ in range(n_arrivals):
            ct = choose_container_type(container_types)
            plant.external_queue.append(Container(ct, step))
        arrival_log.append(n_arrivals)

        if is_manned or not plant.manned if hasattr(plant, 'manned') else is_manned:
            plant.admit_all(step)
        elif not schedule.manned:
            plant.admit_all(step)
        else:
            plant.assign_idle_compressors(step)

        pressure, energy = plant.fill_containers()
        done = plant.collect_completed(step)
        completed.extend(done)

        external_queue_log.append(plant.n_external())
        docked_log.append(plant.n_docked())
        filling_log.append(plant.n_filling())
        pressure_log.append(pressure)
        production_log.append(plant.electrolyzer_step_kg
                               if isinstance(plant.electrolyzer_step_kg, float)
                               else sum(plant.electrolyzer_step_kg.values()))
        energy_log.append(energy)
        for j, comp in enumerate(plant.compressors):
            comp_filling_log[j].append(1 if comp.active is not None else 0)

    queue_log = external_queue_log

    return {
        "completed":          completed,
        "external_queue_log": external_queue_log,
        "docked_log":         docked_log,
        "filling_log":        filling_log,
        "queue_log":          queue_log,
        "manned_log":         manned_log,
        "comp_filling_log":   comp_filling_log,
        "pressure_log":       pressure_log,
        "production_log":     production_log,
        "energy_log":         energy_log,
        "arrival_log":        arrival_log,
        "total_dispensed":    plant.total_dispensed,
        "ez_availability_log":  ez_availability_log,
        "comp_up_log":          comp_up_log,
        "fill_lines_up_log":    fill_lines_up_log,
        "pm_active_log":        pm_active_log,
        "reliability_summary":  reliability_model.summary() if reliability_model else None,
        "step_minutes":         step_minutes,
    }
