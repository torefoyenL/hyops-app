# -*- coding: utf-8 -*-
"""
monte_carlo.py
==============
Monte Carlo sweep across all scenario dimensions for the dual-compressor
hydrogen filling plant.

Sweep dimensions
----------------
  - arrival_rates       : list of avg arrivals/day
  - schedules           : list of StaffSchedule configs
  - plant_configs       : list of plant configuration dicts
  - container_mixes     : list of container fleet mix dicts
  - n_runs              : number of random seeds per scenario combination

Storage
-------
  DuckDB file (local, no server needed).
  One row per run in table `runs`.
  Query with pandas or DuckDB SQL.

Usage
-----
  import monte_carlo as mc

  results_df = mc.run_sweep(
      arrival_rates   = [4, 8, 12, 16],
      schedules       = mc.default_schedules(),
      plant_configs   = mc.default_plant_configs(),
      container_mixes = mc.default_container_mixes(),
      n_runs          = 10,
      days            = 31,
      step_minutes    = 1,
      db_path         = "hydrogen_mc.duckdb",
  )

  # Query results
  df = mc.query(db_path="hydrogen_mc.duckdb", sql="SELECT * FROM runs")
"""

import itertools
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

import duckdb

import plant_topology as pt
import reliability as rel
import plant_operations as po
import results_plant_operations as rpo
import economics as eco


# ============================================================
# Schema — one row per simulation run
# ============================================================

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    -- Identity
    run_id            VARCHAR PRIMARY KEY,
    scenario_id       VARCHAR,       -- shared across seeds of same scenario
    seed              INTEGER,

    -- Sweep dimensions
    arrival_rate      DOUBLE,
    schedule_label    VARCHAR,
    manned            BOOLEAN,
    weekday_hours     VARCHAR,
    weekend           VARCHAR,
    electrolyzer_kg_hr DOUBLE,
    comp_flow_0       DOUBLE,
    comp_flow_1       DOUBLE,
    n_fill_lines      INTEGER,   -- shared fill lines (single pool)
    mix_label         VARCHAR,
    frac_A            DOUBLE,
    frac_B            DOUBLE,
    frac_C            DOUBLE,

    -- Arrival timing pattern (separate from arrival_rate, which is volume)
    arrival_pattern_label VARCHAR,
    pattern_type      VARCHAR,
    peak_hour         DOUBLE,
    peak_hour_2       DOUBLE,
    peak_width_hours  DOUBLE,
    peak_weight       DOUBLE,

    -- Simulation config
    days              INTEGER,
    step_minutes      DOUBLE,

    -- === KPIs ===

    -- Summary
    containers_filled INTEGER,
    count_A           INTEGER,
    count_B           INTEGER,
    count_C           INTEGER,

    -- External wait (outside filling area)
    ext_wait_avg      DOUBLE,
    ext_wait_max      DOUBLE,
    ext_wait_median   DOUBLE,
    ext_wait_p95      DOUBLE,
    ext_wait_zero     INTEGER,

    -- Docked wait (parked, compressor busy)
    doc_wait_avg      DOUBLE,
    doc_wait_max      DOUBLE,
    doc_wait_median   DOUBLE,
    doc_wait_p95      DOUBLE,
    doc_wait_zero     INTEGER,

    -- Fill time (compressor pumping)
    fill_avg          DOUBLE,
    fill_max          DOUBLE,
    fill_median       DOUBLE,
    fill_p95          DOUBLE,

    -- Total time
    total_time_avg    DOUBLE,

    -- Queue occupancy
    avg_external_queue DOUBLE,
    max_external_queue DOUBLE,
    avg_docked_queue   DOUBLE,
    max_docked_queue   DOUBLE,
    avg_filling        DOUBLE,

    -- Exact trailer-hours spent in the external queue while UNMANNED
    -- (sum over unmanned steps of queue_len * step_minutes/60). This is
    -- the precise quantity economics.py needs for queue-cost — unlike
    -- avg_external_queue, which doesn't distinguish manned vs unmanned
    -- time and would overstate cost for partially-manned schedules.
    unmanned_queue_kg_hours DOUBLE,

    -- Per-compressor (shared-line model: only fill utilisation per compressor)
    comp0_utilization  DOUBLE,
    comp1_utilization  DOUBLE,

    -- Energy
    plant_utilization  DOUBLE,
    total_dispensed_kg DOUBLE,
    total_energy_kwh   DOUBLE,
    avg_energy_per_kg  DOUBLE,
    ref_compression_per_kg DOUBLE,

    -- Timing
    elapsed_sec        DOUBLE
)
"""


# ============================================================
# Default sweep configurations
# ============================================================

def default_schedules():
    """
    Returns list of (label, StaffSchedule) tuples.
    Covers the main operational modes.
    """
    return [
        ("unmanned",          po.StaffSchedule(manned=False)),
        ("8-16_closed",       po.StaffSchedule(manned=True, weekday_hours="8-16",  weekend="closed")),
        ("8-20_closed",       po.StaffSchedule(manned=True, weekday_hours="8-20",  weekend="closed")),
        ("8-24_closed",       po.StaffSchedule(manned=True, weekday_hours="8-24",  weekend="closed")),
        ("8-16_open",         po.StaffSchedule(manned=True, weekday_hours="8-16",  weekend="open")),
        ("8-20_open",         po.StaffSchedule(manned=True, weekday_hours="8-20",  weekend="open")),
        ("8-24_open",         po.StaffSchedule(manned=True, weekday_hours="8-24",  weekend="open")),
        ("24_7",              po.StaffSchedule(manned=True, weekday_hours="24/7",   weekend="open")),
    ]


def default_plant_configs():
    """
    Returns list of plant config dicts.
    Each dict maps directly to HydrogenPlant kwargs.
    n_fill_lines is a single int (shared pool for both compressors).
    """
    ELEC  = 132 / 24          # 5.5 kg/hr  (132 kg/day plant)
    CFLOW = ELEC / 2           # 2.75 kg/hr each

    return [
        {
            "label":                    "6_lines",
            "electrolyzer_kg_per_hr":   ELEC,
            "compressor_flow_kg_per_hr":[CFLOW, CFLOW],
            "n_fill_lines":             6,
            "pressure_thresholds":      [94, 278, 500],
            "Isentropic_efficiency":    0.4,
            "electrolyzer_energy_kwh_per_kg": 0,
            "base_power_kw":            0,
        },
        {
            "label":                    "4_lines",
            "electrolyzer_kg_per_hr":   ELEC,
            "compressor_flow_kg_per_hr":[CFLOW, CFLOW],
            "n_fill_lines":             4,
            "pressure_thresholds":      [94, 278, 500],
            "Isentropic_efficiency":    0.4,
            "electrolyzer_energy_kwh_per_kg": 0,
            "base_power_kw":            0,
        },
        {
            "label":                    "2_lines",
            "electrolyzer_kg_per_hr":   ELEC,
            "compressor_flow_kg_per_hr":[CFLOW, CFLOW],
            "n_fill_lines":             2,
            "pressure_thresholds":      [94, 278, 500],
            "Isentropic_efficiency":    0.4,
            "electrolyzer_energy_kwh_per_kg": 0,
            "base_power_kw":            0,
        },
    ]


def default_container_mixes():
    """
    Returns list of container mix dicts.
    Each dict has label + one entry per ContainerType.
    """
    return [
        {
            "label": "base_mix",
            "types": [
                po.ContainerType("Type-A", 1000, 180, 0.3),
                po.ContainerType("Type-B",  600, 180, 0.5),
                po.ContainerType("Type-C",  300, 150, 0.2),
            ]
        },
        {
            "label": "heavy_mix",
            "types": [
                po.ContainerType("Type-A", 1000, 180, 0.6),
                po.ContainerType("Type-B",  600, 180, 0.3),
                po.ContainerType("Type-C",  300, 150, 0.1),
            ]
        },
        {
            "label": "light_mix",
            "types": [
                po.ContainerType("Type-A", 1000, 180, 0.1),
                po.ContainerType("Type-B",  600, 180, 0.3),
                po.ContainerType("Type-C",  300, 150, 0.6),
            ]
        },
    ]


def default_arrival_patterns():
    """
    Returns list of (label, ArrivalPattern) tuples.

    Covers the three pattern types with sensible default peak hours —
    edit or replace with your own list to sweep different peak
    hours/widths. Volume (arrival_rate) is swept separately; these only
    change WHEN containers arrive, never how many per day.
    """
    return [
        ("uniform",       po.ArrivalPattern(pattern_type="uniform")),
        ("morning_peak",  po.ArrivalPattern(pattern_type="single_peak",
                                             peak_hour=8.0, peak_width_hours=1.5)),
        ("morning_afternoon_peak", po.ArrivalPattern(pattern_type="double_peak",
                                             peak_hour=8.0, peak_hour_2=16.0,
                                             peak_width_hours=1.5, peak_weight=0.5)),
    ]


# ============================================================
# Single run
# ============================================================

def _run_single(
    seed: int,
    scenario_id: str,
    arrival_rate: float,
    schedule_label: str,
    schedule: po.StaffSchedule,
    plant_cfg: dict,
    mix: dict,
    days: int,
    step_minutes: float,
    arrival_pattern_label: str = "uniform",
    arrival_pattern: "po.ArrivalPattern" = None,
) -> dict:

    t0 = time.perf_counter()

    if arrival_pattern is None:
        arrival_pattern = po.ArrivalPattern(pattern_type="uniform")

    plant = po.HydrogenPlant(
        electrolyzer_kg_per_hr          = plant_cfg["electrolyzer_kg_per_hr"],
        compressor_flow_kg_per_hr       = plant_cfg["compressor_flow_kg_per_hr"],
        n_fill_lines                    = plant_cfg["n_fill_lines"],
        pressure_thresholds             = plant_cfg.get("pressure_thresholds", [94, 278, 500]),
        Isentropic_efficiency           = plant_cfg.get("Isentropic_efficiency", 0.4),
        electrolyzer_energy_kwh_per_kg  = plant_cfg.get("electrolyzer_energy_kwh_per_kg", 0),
        base_power_kw                   = plant_cfg.get("base_power_kw", 0),
        step_minutes                    = step_minutes,
    )

    results = po.run_simulation(
        container_types     = mix["types"],
        plant               = plant,
        avg_arrivals_per_day= arrival_rate,
        days                = days,
        step_minutes        = step_minutes,
        random_seed         = seed,
        schedule            = schedule,
        arrival_pattern     = arrival_pattern,
    )

    kpis = rpo.compute_kpis(
        results,
        plant,
        mix["types"],
        avg_arrivals_per_day = arrival_rate,
        days                 = days,
    )

    unmanned_kg_hours = eco.unmanned_queue_kg_hours(results, step_minutes)

    elapsed = time.perf_counter() - t0

    # Extract fractions from mix
    total_frac = sum(t.fleet_fraction for t in mix["types"])
    frac = {t.name: t.fleet_fraction / total_frac for t in mix["types"]}

    # Per-compressor stats
    cs = kpis.get("comp_stats", [{}, {}])
    def cget(i, key): return cs[i].get(key, 0) if i < len(cs) else 0

    row = {
        # Identity
        "run_id":           str(uuid.uuid4()),
        "scenario_id":      scenario_id,
        "seed":             seed,

        # Sweep dimensions
        "arrival_rate":     arrival_rate,
        "schedule_label":   schedule_label,
        "manned":           schedule.manned,
        "weekday_hours":    str(getattr(schedule, "_weekday", None)),
        "weekend":          "open" if getattr(schedule, "_weekend_open", False) else "closed",
        "electrolyzer_kg_hr": plant_cfg["electrolyzer_kg_per_hr"],
        "comp_flow_0":      plant_cfg["compressor_flow_kg_per_hr"][0],
        "comp_flow_1":      plant_cfg["compressor_flow_kg_per_hr"][1],
        "n_fill_lines":     plant_cfg["n_fill_lines"],   # int, shared pool
        "mix_label":        mix["label"],
        "frac_A":           frac.get("Type-A", 0),
        "frac_B":           frac.get("Type-B", 0),
        "frac_C":           frac.get("Type-C", 0),

        # Arrival timing pattern
        "arrival_pattern_label": arrival_pattern_label,
        "pattern_type":          arrival_pattern.pattern_type,
        "peak_hour":             getattr(arrival_pattern, "peak_hour", None),
        "peak_hour_2":           getattr(arrival_pattern, "peak_hour_2", None),
        "peak_width_hours":      getattr(arrival_pattern, "peak_width_hours", None),
        "peak_weight":           getattr(arrival_pattern, "peak_weight", None),

        # Simulation config
        "days":             days,
        "step_minutes":     step_minutes,

        # KPIs — summary
        "containers_filled": kpis["containers_filled"],
        "count_A":           kpis["container_counts"].get("Type-A", 0),
        "count_B":           kpis["container_counts"].get("Type-B", 0),
        "count_C":           kpis["container_counts"].get("Type-C", 0),

        # External wait
        "ext_wait_avg":     kpis["ext_avg"],
        "ext_wait_max":     kpis["ext_max"],
        "ext_wait_median":  kpis["ext_median"],
        "ext_wait_p95":     kpis["ext_p95"],
        "ext_wait_zero":    kpis["ext_zero"],

        # Docked wait
        "doc_wait_avg":     kpis["doc_avg"],
        "doc_wait_max":     kpis["doc_max"],
        "doc_wait_median":  kpis["doc_median"],
        "doc_wait_p95":     kpis["doc_p95"],
        "doc_wait_zero":    kpis["doc_zero"],

        # Fill time
        "fill_avg":         kpis["fill_avg"],
        "fill_max":         kpis["fill_max"],
        "fill_median":      kpis["fill_median"],
        "fill_p95":         kpis["fill_p95"],

        # Total
        "total_time_avg":   kpis["avg_total_time_min"],

        # Queue occupancy
        "avg_external_queue": kpis["avg_external_queue"],
        "max_external_queue": kpis["max_external_queue"],
        "avg_docked_queue":   kpis["avg_docked_queue"],
        "max_docked_queue":   kpis["max_docked_queue"],
        "avg_filling":        kpis["avg_filling"],

        # Exact trailer-hours in external queue while unmanned
        "unmanned_queue_kg_hours": unmanned_kg_hours,

        # Per-compressor (shared-line model: only fill utilisation)
        "comp0_utilization":  cget(0, "fill_utilization"),
        "comp1_utilization":  cget(1, "fill_utilization"),

        # Energy
        "plant_utilization":     kpis["plant_utilization"],
        "total_dispensed_kg":    kpis["total_dispensed_kg"],
        "total_energy_kwh":      kpis["total_energy_kwh"],
        "avg_energy_per_kg":     kpis["avg_energy_kwh_per_kg"],
        "ref_compression_per_kg":kpis["reference_compression_kwh_per_kg"],

        # Timing
        "elapsed_sec":           elapsed,
    }

    return row


# ============================================================
# DB helpers
# ============================================================

def init_db(db_path: str):
    con = duckdb.connect(db_path)
    con.execute(DB_SCHEMA)
    con.close()


def insert_rows(db_path: str, rows: list):
    import pandas as pd
    if not rows:
        return
    df = pd.DataFrame(rows)
    con = duckdb.connect(db_path)
    con.execute("INSERT INTO runs SELECT * FROM df")
    con.close()


def query(db_path: str, sql: str = "SELECT * FROM runs"):
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(sql).df()
    con.close()
    return df


def clear_db(db_path: str):
    """Delete all rows — keeps schema intact."""
    con = duckdb.connect(db_path)
    con.execute("DELETE FROM runs")
    con.close()


# ============================================================
# Internal helper
# ============================================================

def _load_existing_keys(db_path: str) -> set:
    """Return set of (scenario_id, seed) already in DB."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        df  = con.execute("SELECT scenario_id, seed FROM runs").df()
        con.close()
        return set(zip(df["scenario_id"], df["seed"]))
    except Exception:
        return set()


# ============================================================
# Main sweep
# ============================================================

def run_sweep(
    arrival_rates:    list,
    schedules:        list  = None,
    plant_configs:    list  = None,
    container_mixes:  list  = None,
    arrival_patterns: list  = None,
    n_runs:           int   = 10,
    days:             int   = 31,
    step_minutes:     float = 1.0,
    db_path:          str   = "hydrogen_mc.duckdb",
    batch_size:       int   = 50,    # insert to DB every N runs
    verbose:          bool  = True,
) -> "pd.DataFrame":
    """
    Run Monte Carlo sweep across all dimension combinations.

    Parameters
    ----------
    arrival_rates    : list of floats, e.g. [4, 8, 12, 16] — VOLUME (containers/day)
    schedules        : list of (label, StaffSchedule) — default_schedules() if None
    plant_configs    : list of plant config dicts     — default_plant_configs() if None
    container_mixes  : list of mix dicts              — default_container_mixes() if None
    arrival_patterns : list of (label, ArrivalPattern) — TIMING (when arrivals
                       land during the day). default_arrival_patterns() if None.
                       Independent of arrival_rates — each arrival_rate is run
                       under each pattern, so you can separate the volume effect
                       from the timing effect in the results.
    n_runs           : random seeds per scenario combination
    days             : simulation days per run
    step_minutes     : simulation timestep
    db_path          : path to DuckDB file
    batch_size       : rows buffered before DB insert
    verbose          : print progress

    Returns
    -------
    pd.DataFrame of all runs (also stored in DB)
    """

    if schedules        is None: schedules        = default_schedules()
    if plant_configs    is None: plant_configs    = default_plant_configs()
    if container_mixes  is None: container_mixes  = default_container_mixes()
    if arrival_patterns is None: arrival_patterns = default_arrival_patterns()

    import pandas as pd
    init_db(db_path)

    # Load already-completed (scenario_id, seed) pairs to skip duplicates
    existing = _load_existing_keys(db_path)
    if verbose and existing:
        print(f"  Skipping {len(existing)} already-completed runs (resume mode)")

    # Build all scenario combinations
    combos = list(itertools.product(
        arrival_rates,
        schedules,
        plant_configs,
        container_mixes,
        arrival_patterns,
    ))

    total_runs = len(combos) * n_runs  # upper bound; may be fewer if resuming
    if verbose:
        print(f"\n{'='*60}")
        print(f"Monte Carlo Sweep")
        print(f"  Scenarios : {len(combos)}")
        print(f"  Seeds/scenario: {n_runs}")
        print(f"  Total runs: {total_runs}")
        print(f"  DB: {db_path}")
        print(f"{'='*60}")

    all_rows = []
    buffer   = []
    run_count = 0
    t_start  = time.perf_counter()

    for arrival_rate, (sched_label, schedule), plant_cfg, mix, (pattern_label, pattern) in combos:

        scenario_id = (
            f"arr{arrival_rate}_"
            f"{sched_label}_"
            f"{plant_cfg.get('label','plant')}_"
            f"{mix['label']}_"
            f"{pattern_label}"
        )

        for seed in range(n_runs):

            if (scenario_id, seed) in existing:
                continue   # already in DB — skip

            row = _run_single(
                seed                  = seed,
                scenario_id           = scenario_id,
                arrival_rate          = arrival_rate,
                schedule_label        = sched_label,
                schedule              = schedule,
                plant_cfg             = plant_cfg,
                mix                   = mix,
                days                  = days,
                step_minutes          = step_minutes,
                arrival_pattern_label = pattern_label,
                arrival_pattern       = pattern,
            )

            buffer.append(row)
            all_rows.append(row)
            run_count += 1

            # Flush to DB
            if len(buffer) >= batch_size:
                insert_rows(db_path, buffer)
                buffer.clear()

            if verbose and run_count % 50 == 0:
                elapsed = time.perf_counter() - t_start
                rate    = run_count / elapsed
                eta     = (total_runs - run_count) / rate if rate > 0 else 0
                print(f"  [{run_count}/{total_runs}]  "
                      f"{elapsed:.0f}s elapsed  "
                      f"ETA {eta:.0f}s  "
                      f"({rate:.1f} runs/s)")

    # Final flush
    if buffer:
        insert_rows(db_path, buffer)

    elapsed_total = time.perf_counter() - t_start
    if verbose:
        print(f"\nDone. {run_count} runs in {elapsed_total:.1f}s "
              f"({run_count/elapsed_total:.1f} runs/s)")
        print(f"DB: {db_path}")

    return pd.DataFrame(all_rows)


# ============================================================
# Quick analysis helpers
# ============================================================

def summary_by(db_path: str, group_by: list) -> "pd.DataFrame":
    """
    Return mean KPIs grouped by the given columns.

    Example
    -------
    mc.summary_by("hydrogen_mc.duckdb",
                  ["arrival_rate", "schedule_label"])
    """
    cols = ", ".join(group_by)
    sql  = f"""
        SELECT
            {cols},
            COUNT(*)                    AS n_runs,
            AVG(containers_filled)      AS avg_containers_filled,
            AVG(ext_wait_avg)           AS avg_ext_wait,
            AVG(doc_wait_avg)           AS avg_doc_wait,
            AVG(fill_avg)               AS avg_fill_time,
            AVG(total_time_avg)         AS avg_total_time,
            AVG(avg_external_queue)     AS avg_ext_queue,
            AVG(avg_docked_queue)       AS avg_doc_queue,
            AVG(plant_utilization)      AS avg_utilization,
            AVG(total_dispensed_kg)     AS avg_dispensed_kg,
            AVG(comp0_utilization)      AS avg_comp0_util,
            AVG(comp1_utilization)      AS avg_comp1_util
        FROM runs
        GROUP BY {cols}
        ORDER BY {cols}
    """
    return query(db_path, sql)


def run_count(db_path: str) -> int:
    df = query(db_path, "SELECT COUNT(*) AS n FROM runs")
    return int(df["n"].iloc[0])
