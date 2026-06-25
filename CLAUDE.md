# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

HyOps is a Streamlit app for simulating and optimising hydrogen filling plant operations. It models electrolyzer → compressor → fill-line plants with configurable topologies, runs discrete-event queueing simulations of trailer arrivals, and evaluates economics across staffing schedules via Monte Carlo sweeps. Results are stored in DuckDB.

## Running the App

```
uv sync                              # install dependencies (first time)
uv run streamlit run streamlit_app.py # launch the app
```

Python 3.14, dependencies: streamlit, pandas, numpy, matplotlib, plotly, duckdb.

## Architecture

The simulation pipeline flows left-to-right through these modules:

```
plant_topology → reliability → plant_operations → results_plant_operations → economics → monte_carlo
                                                                                            ↓
                                                                                     econ_plots
```

**streamlit_app.py** — Single-file Streamlit UI. All configuration lives in sidebar expanders. Session state holds mutable RAM parameters, equipment library, BOM, staff costs, and cached simulation results. Uses `silence_show()` / `show_figs()` context managers to redirect matplotlib figures into Streamlit.

**plant_topology.py** — Static plant definition (`PlantTopology` dataclass). Three topology modes:
- `"common"` — pooled electrolyzers, pooled compressors, shared fill lines
- `"pooled_ez_dedicated_comp"` — pooled EZ, each compressor owns dedicated fill lines
- `"trains"` — independent parallel trains, each with own EZ/comp/fill lines

**reliability.py** — RAM engine. `WeibullPMUnit` (wear-out + planned maintenance) for EZ bodies, stacks, compressor sub-components. `ExpUnit` (exponential) for auxiliary BOM equipment. `ReliabilityModel.step()` returns a `CapacityState` snapshot (ez_frac[], comp_up[], fill_line_up[]) consumed by the plant each timestep. `run_reliability_timeline()` runs the model standalone (hourly, multi-year) without the queueing simulation.

**plant_operations.py** — Discrete-event simulation. `HydrogenPlant` manages external queue → docked → filling flow. `run_simulation()` is the main loop: Poisson arrivals, schedule-aware docking, per-step compressor filling with electrolyzer budget constraints. Returns dict of per-step logs and completed containers.

**results_plant_operations.py** — KPI computation (`compute_kpis`) and matplotlib plotting (`plot_results`, `plot_plant_architecture`, `plot_reliability_timeline`). KPIs: 3-zone wait times (external/docked/fill), queue occupancy, energy, per-compressor utilisation.

**economics.py** — Revenue/cost model. Revenue = margin × kg dispensed. Costs = annual staff cost (per schedule label) + unmanned queue cost (trailer-hours in external queue during unmanned steps). `run_economics()` for single runs, `add_economics_columns()` for DataFrame-level Monte Carlo results.

**monte_carlo.py** — Sweep across arrival rates × schedules × plant configs × container mixes × arrival patterns × reliability on/off. Each combination × N seeds → one row in DuckDB table `runs`. `run_sweep()` is resumable (skips existing scenario_id + seed pairs).

**econ_plots.py** — Economics visualisations (waterfall, stacked bar, net result spread, sensitivity). Imported as `ep` by the Streamlit app.

## Key Design Patterns

- **All modules use flat imports** (`import plant_operations as po`, not packages). Everything runs from the repo root.
- **Simulation timestep is 1 minute** by default (`step_minutes=1`). All per-step quantities scale by `60/step_minutes`.
- **Electrolyzer budget is the hard ceiling** — compressors draw from a pooled (or per-train) EZ production budget each step. Compressors can only dispense what electrolyzers produce.
- **Reliability is optional** — when `ReliabilityModel` is None, plant runs at 100% theoretical capacity. The operations sim works identically either way.
- **Topology ordering matters** — `ReliabilityModel`, `HydrogenPlant`, and `run_reliability_timeline` all iterate electrolyzers/compressors in the same positional order (train-then-unit). `ez_frac[i]` must align with `electrolyzer_rated_kg_per_hr[i]`.
- **DuckDB schema is fixed** in `monte_carlo.DB_SCHEMA`. Adding a new KPI column requires updating the schema, `_run_single()`, and potentially `add_economics_columns()`.
- **Staff schedule labels** (e.g. `"8-16_closed"`, `"24_7"`) are string keys used across economics, monte_carlo, and the UI. Format: `"{start}-{end}_{weekend}"` where weekend is `open`/`closed`.
