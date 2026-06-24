# -*- coding: utf-8 -*-
"""
economics.py
============
Economic layer on top of the dual-compressor hydrogen plant simulator.

Cost / revenue model
---------------------
Revenue
    margin_kr_per_kg * total_dispensed_kg
    Default 30 kr/kg.

Staff cost (annual, fixed — pay + overhead)
    Looked up per StaffSchedule from STAFF_ANNUAL_COST (kr/year).
    Independent of simulated arrival rate or plant config.

Unmanned external-queue cost
    Only meaningful for manned=False (or unmanned hours within a mixed
    schedule): when nobody is on site, arriving drivers must hook up
    their own trailers, so every hour a trailer sits in the external
    queue is modelled as a cost (default 1500 kr/hr).
    Cost = queue_cost_kr_per_hr * (external-queue-hours accumulated
    during unmanned periods).

    Implementation note: external wait is tracked in compute_kpis() as
    average + max queue length, not as "kg of external-wait-time
    accumulated during unmanned hours" directly. This module derives
    that quantity from the raw per-step external_queue_log captured in
    the simulation results, restricted to steps where the schedule was
    unmanned at that instant (manned_log == 0). For schedule=manned=False
    every step is unmanned, so this reduces to mean(queue_log) * hours.

Net economics (per simulated run)
    revenue - staff_cost_pro_rated - external_queue_cost

All run-level costs are pro-rated linearly to the run's simulated
`days` so they can be summed/compared on an annualised (365-day) basis.
"""

# economics.py — no top-level numpy/pandas imports (Streamlit compatibility)

# ============================================================
# Default cost & revenue assumptions — override freely
# ============================================================

MARGIN_KR_PER_KG       = 30.0     # revenue per kg H2 sold
QUEUE_COST_KR_PER_HR   = 1500.0   # cost per trailer-hour in external queue while unmanned

# Annual staff cost (pay + overhead), kr/year.
# Keyed by schedule label as used in monte_carlo.default_schedules() /
# the notebook's schedules list. Edit these to match real wage data.
STAFF_ANNUAL_COST = {
    "unmanned":     0,
    "8-16_closed":  650_000,
    "8-20_closed":  950_000,
    "8-24_closed": 1_550_000,
    "8-16_open":    850_000,
    "8-20_open":   1_200_000,
    "8-24_open":   1_750_000,
    "24_7":        2_800_000,
}

DAYS_PER_YEAR = 365.0


def staff_annual_cost(schedule_label: str, overrides: dict = None) -> float:
    """Annual staff cost (kr/year) for a schedule label."""
    table = STAFF_ANNUAL_COST if overrides is None else {**STAFF_ANNUAL_COST, **overrides}
    if schedule_label not in table:
        raise KeyError(
            f"No annual staff cost defined for schedule_label={schedule_label!r}. "
            f"Add it to STAFF_ANNUAL_COST or pass via overrides."
        )
    return table[schedule_label]


# ============================================================
# External-queue cost while unmanned
# ============================================================

def unmanned_queue_kg_hours(results, step_minutes: float) -> float:
    """
    Exact trailer-hours spent in the external queue during unmanned
    steps, for the simulated run.

    Uses results['queue_log'] (external queue length per step) and
    results['manned_log'] (1 manned / 0 unmanned per step), both
    produced by plant_operations.run_simulation().

    trailer_hours = sum_over_unmanned_steps(queue_len) * (step_minutes/60)
    """
    queue_log  = results["queue_log"]
    manned_log = results.get("manned_log")

    if manned_log is None:
        # No schedule info logged -> assume fully unmanned (legacy results)
        return sum(queue_log) * (step_minutes / 60.0)

    return sum(
        q for q, m in zip(queue_log, manned_log) if m == 0
    ) * (step_minutes / 60.0)


def unmanned_queue_cost_kr(results, step_minutes: float,
                            queue_cost_kr_per_hr: float = QUEUE_COST_KR_PER_HR) -> float:
    """
    Cost (kr) of trailers waiting in the external queue during
    unmanned steps, for the simulated run.

    cost = queue_cost_kr_per_hr * unmanned_queue_kg_hours(...)
    """
    return queue_cost_kr_per_hr * unmanned_queue_kg_hours(results, step_minutes)


# ============================================================
# Per-run economics
# ============================================================

def run_economics(
    results,
    kpis,
    schedule_label: str,
    days: int,
    margin_kr_per_kg: float = MARGIN_KR_PER_KG,
    queue_cost_kr_per_hr: float = QUEUE_COST_KR_PER_HR,
    staff_cost_overrides: dict = None,
) -> dict:
    """
    Compute revenue / cost / margin for one simulated run, both as
    simulated-period totals and annualised (365-day) figures.

    Parameters
    ----------
    results        : dict returned by plant_operations.run_simulation()
    kpis           : dict returned by results_plant_operations.compute_kpis()
    schedule_label : key into STAFF_ANNUAL_COST (e.g. "unmanned", "8-16_closed")
    days           : number of simulated days (for pro-rating to annual)
    """
    step_minutes = results["step_minutes"]

    dispensed_kg = kpis["total_dispensed_kg"]
    revenue_kr   = margin_kr_per_kg * dispensed_kg

    annual_staff_cost_kr = staff_annual_cost(schedule_label, staff_cost_overrides)
    staff_cost_kr_run    = annual_staff_cost_kr * (days / DAYS_PER_YEAR)

    queue_cost_kr_run = unmanned_queue_cost_kr(
        results, step_minutes, queue_cost_kr_per_hr
    )

    total_cost_kr_run = staff_cost_kr_run + queue_cost_kr_run
    net_kr_run        = revenue_kr - total_cost_kr_run

    scale_to_annual = DAYS_PER_YEAR / days

    return {
        "schedule_label":   schedule_label,
        "days_simulated":   days,

        # --- run-period figures (as simulated) ---
        "dispensed_kg":     dispensed_kg,
        "revenue_kr":       revenue_kr,
        "staff_cost_kr":    staff_cost_kr_run,
        "queue_cost_kr":    queue_cost_kr_run,
        "total_cost_kr":    total_cost_kr_run,
        "net_kr":           net_kr_run,

        # --- annualised (365-day) figures ---
        "dispensed_kg_annual":  dispensed_kg   * scale_to_annual,
        "revenue_kr_annual":    revenue_kr     * scale_to_annual,
        "staff_cost_kr_annual": annual_staff_cost_kr,
        "queue_cost_kr_annual": queue_cost_kr_run * scale_to_annual,
        "total_cost_kr_annual": annual_staff_cost_kr + queue_cost_kr_run * scale_to_annual,
        "net_kr_annual":        revenue_kr * scale_to_annual
                                 - (annual_staff_cost_kr + queue_cost_kr_run * scale_to_annual),
    }


def print_economics(econ: dict):
    print(f"\n=== Economics — schedule: {econ['schedule_label']} "
          f"({econ['days_simulated']} days simulated) ===")
    print(f"  H2 dispensed:        {econ['dispensed_kg']:.1f} kg "
          f"({econ['dispensed_kg_annual']:.0f} kg/yr annualised)")
    print(f"  Revenue:              {econ['revenue_kr']:,.0f} kr "
          f"({econ['revenue_kr_annual']:,.0f} kr/yr)")
    print(f"  Staff cost:          -{econ['staff_cost_kr']:,.0f} kr "
          f"({econ['staff_cost_kr_annual']:,.0f} kr/yr)")
    print(f"  Unmanned queue cost: -{econ['queue_cost_kr']:,.0f} kr "
          f"({econ['queue_cost_kr_annual']:,.0f} kr/yr)")
    print(f"  ────────────────────────────")
    print(f"  Net result:           {econ['net_kr']:,.0f} kr "
          f"({econ['net_kr_annual']:,.0f} kr/yr)")


# ============================================================
# Monte Carlo / DuckDB-level economics
# ============================================================

def add_economics_columns(
    df,
    margin_kr_per_kg: float = MARGIN_KR_PER_KG,
    queue_cost_kr_per_hr: float = QUEUE_COST_KR_PER_HR,
    staff_cost_overrides: dict = None,
):
    """
    Add revenue / cost / net columns to a Monte Carlo results DataFrame
    (e.g. from monte_carlo.query() / mc.run_sweep()).

    Requires columns: total_dispensed_kg, schedule_label, days.

    Exact mode (preferred): if the DataFrame has an
    `unmanned_queue_kg_hours` column (written by monte_carlo._run_single
    via economics.unmanned_queue_kg_hours() on the raw per-step logs),
    queue cost is computed exactly from that — correct for any schedule,
    manned, unmanned, or mixed.

    Legacy fallback: if that column is absent (older DB from before this
    column existed), falls back to an approximation using
    avg_external_queue * 24h/day, which is only exact for the fully
    unmanned schedule and OVERSTATES cost for partially-manned schedules
    (it has no way to know which hours of the average were actually
    unmanned). Re-run the sweep to get the exact column.
    """
    df = df.copy()

    table = STAFF_ANNUAL_COST if staff_cost_overrides is None \
        else {**STAFF_ANNUAL_COST, **staff_cost_overrides}

    df["staff_cost_kr_annual"] = df["schedule_label"].map(table)

    df["revenue_kr_annual"] = (
        margin_kr_per_kg * df["total_dispensed_kg"] * (DAYS_PER_YEAR / df["days"])
    )

    if "unmanned_queue_kg_hours" in df.columns:
        # Exact: trailer-hours already restricted to unmanned steps.
        df["queue_cost_kr_annual"] = (
            queue_cost_kr_per_hr * df["unmanned_queue_kg_hours"] * (DAYS_PER_YEAR / df["days"])
        )
    else:
        queue_kg_hours_per_day = df["avg_external_queue"] * 24.0
        df["queue_cost_kr_annual"] = (
            queue_cost_kr_per_hr * queue_kg_hours_per_day * DAYS_PER_YEAR
        )
        # Schedules that are manned 100% of the time (e.g. "24/7" weekday
        # hours with weekend open) never have an unmanned external queue,
        # so the approximation should report zero queue cost for them.
        if "weekday_hours" in df.columns:
            fully_manned = df["weekday_hours"].astype(str).str.contains(
                r"(0, 24)", regex=False
            )
            df.loc[fully_manned, "queue_cost_kr_annual"] = 0.0

    df["total_cost_kr_annual"] = df["staff_cost_kr_annual"] + df["queue_cost_kr_annual"]
    df["net_kr_annual"]        = df["revenue_kr_annual"] - df["total_cost_kr_annual"]

    return df


def economics_summary(df, group_by=("schedule_label",)):
    """Mean economics by group, sorted by net_kr_annual descending."""
    group_by = list(group_by)
    cols = [
        "revenue_kr_annual", "staff_cost_kr_annual",
        "queue_cost_kr_annual", "total_cost_kr_annual", "net_kr_annual",
    ]
    out = (
        df.groupby(group_by)[cols]
        .mean()
        .reset_index()
        .sort_values("net_kr_annual", ascending=False)
    )
    return out
