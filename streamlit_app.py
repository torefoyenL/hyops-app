import matplotlib
matplotlib.use("Agg")

import contextlib
import time
import streamlit as st

import plant_topology as pt
import reliability as rel
import plant_operations as po
import results_plant_operations as rpo
import monte_carlo as mc
import economics as eco
import econ_plots as ep

st.set_page_config(page_title="HyOps", layout="wide")

# ============================================================
# Constants / defaults
# ============================================================

ALL_SCHEDULES = [
    "unmanned", "8-16_closed", "8-20_closed", "8-24_closed",
    "8-16_open", "8-20_open", "8-24_open", "24_7",
]

_EQ_LIB_DEFAULTS = {
    "pressure_tx":      (120_000, 48),
    "temp_tx":          (120_000, 48),
    "flow_tx":          (100_000, 72),
    "control_valve":    (100_000, 48),
    "manual_valve":     (200_000, 48),
    "check_valve":      (180_000, 48),
    "solenoid":         ( 80_000, 48),
    "vibration_sensor": ( 80_000, 48),
    "lube_oil_pump":    ( 35_000, 72),
    "fan":              ( 40_000, 48),
    "plc_module":       (150_000, 48),
}
_BOM_DEFAULTS = {
    "ez_aux":        {"control_valve": 2, "pressure_tx": 2, "temp_tx": 2},
    "comp_aux":      {"control_valve": 2, "lube_oil_pump": 1, "fan": 2,
                      "pressure_tx": 2, "temp_tx": 2, "vibration_sensor": 1},
    "fill_line_aux": {"manual_valve": 2, "control_valve": 1,
                      "pressure_tx": 1, "flow_tx": 1, "check_valve": 1},
}
_RAM_DEFAULTS = {
    "ez_beta": 2.5, "ez_eta": 50000, "ez_mttr": 72,
    "stk_beta": 2.5, "stk_eta": 55000, "stk_mttr": 168,
    "cb_beta": 3.0, "cb_eta": 40000, "cb_mttr": 240,
    "cm_beta": 3.0, "cm_eta": 40000, "cm_mttr": 168,
    "cs_beta": 3.0, "cs_eta": 20000, "cs_mttr": 48,
    "pm_interval_h": 8760, "pm_ez_dur": 72, "pm_comp_dur": 96,
}
_STAFF_DEFAULTS = {
    "unmanned": 0, "8-16_closed": 650_000, "8-20_closed": 950_000,
    "8-24_closed": 1_550_000, "8-16_open": 850_000,
    "8-20_open": 1_200_000, "8-24_open": 1_750_000, "24_7": 2_800_000,
}

# ============================================================
# Session state — one-time initialisation only
# ============================================================

def _ss_init():
    defs = {
        "eq_lib":        {k: {"mtbf": v[0], "mttr": v[1]} for k, v in _EQ_LIB_DEFAULTS.items()},
        "bom":           {n: dict(c) for n, c in _BOM_DEFAULTS.items()},
        "ram_params":    dict(_RAM_DEFAULTS),
        "staff_costs":   dict(_STAFF_DEFAULTS),
        "margin_kr":     30.0,
        "queue_cost_kr": 1500.0,
        "db_path":       "hydrogen_mc.duckdb",
        # simulation results
        "tl_result":     None,
        "tl_seed_used":  None,
        "single_result": None,
        "fmea_df":       None,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ss_init()

# ============================================================
# Pure helpers — no side effects, read session_state only
# ============================================================

def _build_ram_dicts():
    """Build RAM parameter dicts from session_state. Call only when needed."""
    p = st.session_state["ram_params"]
    params = {
        "electrolyzer_body": dict(beta=p["ez_beta"],  eta=p["ez_eta"],  mttr_corrective=p["ez_mttr"]),
        "stack":             dict(beta=p["stk_beta"], eta=p["stk_eta"], mttr_corrective=p["stk_mttr"]),
        "compressor_block":  dict(beta=p["cb_beta"],  eta=p["cb_eta"],  mttr_corrective=p["cb_mttr"]),
        "compressor_motor":  dict(beta=p["cm_beta"],  eta=p["cm_eta"],  mttr_corrective=p["cm_mttr"]),
        "compressor_seals":  dict(beta=p["cs_beta"],  eta=p["cs_eta"],  mttr_corrective=p["cs_mttr"]),
    }
    pm = {
        "interval_h":              int(p["pm_interval_h"]),
        "electrolyzer_duration_h": int(p["pm_ez_dur"]),
        "compressor_duration_h":   int(p["pm_comp_dur"]),
    }
    return params, pm, dict(st.session_state["eq_lib"]), dict(st.session_state["bom"])


def make_schedule(label):
    if label == "unmanned":
        return po.StaffSchedule(manned=False)
    if label == "24_7":
        return po.StaffSchedule(manned=True, weekday_hours="24/7", weekend="open")
    hours, weekend = label.rsplit("_", 1)
    return po.StaffSchedule(manned=True, weekday_hours=hours, weekend=weekend)


def make_container_types(fa, fb, fc):
    tot = fa + fb + fc or 1.0
    return [
        po.ContainerType("Type-A", 1000, 180, fa / tot),
        po.ContainerType("Type-B",  600, 180, fb / tot),
        po.ContainerType("Type-C",  300, 150, fc / tot),
    ]


def make_arrival_pattern(ptype, ph, ph2, pw, pwt):
    if ptype == "uniform":
        return po.ArrivalPattern(pattern_type="uniform")
    if ptype == "single_peak":
        return po.ArrivalPattern(pattern_type="single_peak", peak_hour=ph, peak_width_hours=pw)
    return po.ArrivalPattern(pattern_type="double_peak",
                              peak_hour=ph, peak_hour_2=ph2,
                              peak_width_hours=pw, peak_weight=pwt)


def make_rel_model(topology, seed, reliability_on):
    if not reliability_on:
        return None
    params, pm, eq_lib, bom = _build_ram_dicts()
    return rel.ReliabilityModel(
        topology, random_seed=int(seed),
        reliability_params=params, pm_config=pm,
        eq_lib=eq_lib, bom=bom,
    )


def safe_count(path):
    try:
        return mc.run_count(path)
    except Exception:
        return 0


@contextlib.contextmanager
def silence_show():
    import matplotlib.pyplot as plt
    plt.close("all")
    _show = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.show = _show


def show_figs():
    import matplotlib.pyplot as plt
    for n in plt.get_fignums():
        st.pyplot(plt.figure(n), clear_figure=False)
    plt.close("all")


def generate_fmea(topology):
    import pandas as pd
    rows = []
    n_ez   = topology.total_electrolyzers()
    n_comp = topology.total_compressors()
    n_fill = topology.total_fill_lines()
    stacks = topology.stacks_per_electrolyzer if topology.mode != "trains" else 2
    for i in range(n_ez):
        rows.append({
            "ID": f"EZ-{i+1}-BODY", "Node": f"Electrolyzer {i+1}", "Type": "Electrolyzer",
            "Failure Mode": "Body / housing failure", "Cause": "Wear, corrosion (Weibull b=2.5)",
            "Local Effect": "Full EZ offline", "System Effect": f"~{100/n_ez:.0f}% capacity loss",
            "Safeguard": "Isolation valves, pressure relief",
            "Severity": 8, "Occurrence": 3, "Detection": 4,
            "Recommended Action": "Annual inspection",
        })
        for s in range(stacks):
            rows.append({
                "ID": f"EZ-{i+1}-STK{s+1}", "Node": f"EZ {i+1} Stack {s+1}", "Type": "Stack",
                "Failure Mode": "Stack degradation", "Cause": "Membrane wear (Weibull b=2.5)",
                "Local Effect": f"EZ {i+1} output -{100//stacks:.0f}%",
                "System Effect": f"~{100/n_ez/stacks:.0f}% loss",
                "Safeguard": "H2 purity monitor",
                "Severity": 6, "Occurrence": 5, "Detection": 3,
                "Recommended Action": "Quarterly test",
            })
    for j in range(n_comp):
        sys_eff = "Full plant offline" if n_comp == 1 else f"~{100/n_comp:.0f}% loss"
        for fm, sev, occ, det in [
            ("Seal failure", 9, 4, 3), ("Motor failure", 8, 3, 4), ("Block/valve failure", 8, 3, 5),
        ]:
            rows.append({
                "ID": f"COMP-{j+1}-{fm[:3].upper()}", "Node": f"Compressor {j+1}", "Type": "Compressor",
                "Failure Mode": fm, "Cause": "Wear (Weibull b=3.0)",
                "Local Effect": "Compressor offline", "System Effect": sys_eff,
                "Safeguard": "HP trip, gas detector",
                "Severity": sev, "Occurrence": occ, "Detection": det,
                "Recommended Action": "6-monthly inspection",
            })
    for k in range(n_fill):
        rows.append({
            "ID": f"FL-{k+1}", "Node": f"Fill Line {k+1}", "Type": "Fill Line",
            "Failure Mode": "Valve/hose failure", "Cause": "Wear (Exponential)",
            "Local Effect": "Fill line out of service",
            "System Effect": f"1 of {n_fill} lines lost",
            "Safeguard": "Breakaway coupling",
            "Severity": 6, "Occurrence": 3, "Detection": 2,
            "Recommended Action": "Monthly inspection",
        })
    df = pd.DataFrame(rows)
    df["RPN"] = df["Severity"] * df["Occurrence"] * df["Detection"]
    return df.sort_values("RPN", ascending=False).reset_index(drop=True)


# ============================================================
# Sidebar — reads widgets → stores to session_state
# ============================================================

st.sidebar.title("⚙️ HyOps Configuration")

# ── Topology ─────────────────────────────────────────────────
with st.sidebar.expander("🏗️ Plant & Topology", expanded=True):
    topology_mode = st.selectbox("Wiring mode", ["common", "pooled_ez_dedicated_comp", "trains"])
    if topology_mode == "common":
        n_ez        = st.slider("Electrolyzers", 1, 8, 3)
        stacks      = st.slider("Stacks per electrolyzer", 1, 4, 2)
        ez_kg_day   = st.number_input("Electrolyzer capacity (kg/day)", 1.0, value=132.0, step=1.0)
        n_comp      = st.slider("Compressors", 1, 6, 2)
        comp_kg_day = st.number_input("Compressor flow (kg/day)", 1.0, value=132.0, step=1.0)
        n_fill      = st.slider("Shared fill lines", 1, 12, 4)
        TOPOLOGY = pt.PlantTopology(
            mode="common", n_electrolyzers=n_ez, stacks_per_electrolyzer=stacks,
            electrolyzer_kg_per_hr_each=(ez_kg_day/24)/n_ez,
            n_compressors=n_comp,
            compressor_flow_kg_per_hr_each=(comp_kg_day/24)/n_comp,
            n_fill_lines=n_fill,
        )
    elif topology_mode == "pooled_ez_dedicated_comp":
        n_ez           = st.slider("Electrolyzers", 1, 8, 3)
        stacks         = st.slider("Stacks per electrolyzer", 1, 4, 2)
        ez_kg_day      = st.number_input("Electrolyzer capacity (kg/day)", 1.0, value=132.0, step=1.0)
        n_comp         = st.slider("Compressors", 1, 6, 2)
        comp_kg_day    = st.number_input("Compressor flow (kg/day)", 1.0, value=132.0, step=1.0)
        lines_per_comp = st.slider("Fill lines per compressor", 1, 6, 2)
        TOPOLOGY = pt.PlantTopology(
            mode="pooled_ez_dedicated_comp",
            n_electrolyzers=n_ez, stacks_per_electrolyzer=stacks,
            electrolyzer_kg_per_hr_each=(ez_kg_day/24)/n_ez,
            n_compressors=n_comp,
            compressor_flow_kg_per_hr_each=(comp_kg_day/24)/n_comp,
            n_fill_lines_per_compressor=lines_per_comp,
        )
    else:
        n_trains          = st.slider("Number of trains", 2, 4, 2)
        ez_per_train      = st.slider("Electrolyzers per train", 1, 4, 2)
        stacks            = st.slider("Stacks per electrolyzer", 1, 4, 2)
        ez_kg_day_train   = st.number_input("EZ capacity per train (kg/day)", 1.0, value=66.0, step=1.0)
        comp_per_train    = st.slider("Compressors per train", 1, 3, 1)
        comp_kg_day_train = st.number_input("Compressor flow per train (kg/day)", 1.0, value=66.0, step=1.0)
        lines_per_train   = st.slider("Fill lines per train", 1, 6, 2)
        TOPOLOGY = pt.PlantTopology(
            mode="trains",
            trains=[
                pt.Train(
                    label=f"Train {i+1}", n_electrolyzers=ez_per_train,
                    electrolyzer_kg_per_hr_each=(ez_kg_day_train/24)/ez_per_train,
                    n_compressors=comp_per_train,
                    compressor_flow_kg_per_hr_each=(comp_kg_day_train/24)/comp_per_train,
                    n_fill_lines=lines_per_train, stacks_per_electrolyzer=stacks,
                )
                for i in range(n_trains)
            ],
        )
    theo  = TOPOLOGY.theoretical_capacity_kg_per_day()
    bneck = "compressor" if TOPOLOGY.compressor_is_bottleneck() else "electrolyzer"
    st.caption(
        f"**{TOPOLOGY.total_electrolyzers()} EZ · {TOPOLOGY.total_compressors()} comp · "
        f"{TOPOLOGY.total_fill_lines()} fill lines**  \n"
        f"Theoretical: **{theo:.1f} kg/day** ({bneck} limited)"
    )

# ── RAM on/off ───────────────────────────────────────────────
with st.sidebar.expander("⚡ Reliability (RAM)", expanded=False):
    RELIABILITY_ON   = st.toggle("Simulate with live availability", value=False)
    RELIABILITY_SEED = st.number_input("Reliability seed", min_value=0, value=42, step=1,
                                        disabled=not RELIABILITY_ON)

# ── Container fleet ──────────────────────────────────────────
with st.sidebar.expander("🚛 Container fleet", expanded=False):
    frac_a = st.slider("Type-A  (1000 kg)", 0.0, 1.0, 0.3, 0.05, key="fa")
    frac_b = st.slider("Type-B  (600 kg)",  0.0, 1.0, 0.5, 0.05, key="fb")
    frac_c = st.slider("Type-C  (300 kg)",  0.0, 1.0, 0.2, 0.05, key="fc")

# ── Arrivals ─────────────────────────────────────────────────
with st.sidebar.expander("📦 Arrivals", expanded=False):
    avg_arrivals = st.number_input("Avg containers / day", min_value=0.01, value=3.0,
                                    step=0.5, format="%.1f")
    pattern_type = st.selectbox("Timing pattern", ["uniform", "single_peak", "double_peak"])
    peak_hour = peak_hour_2 = None
    peak_width = 3.0
    peak_weight = 0.5
    if pattern_type == "single_peak":
        peak_hour  = st.slider("Peak hour", 0.0, 24.0, 8.0, 0.5)
        peak_width = st.slider("Peak width (hours)", 0.5, 8.0, 3.0, 0.5)
    elif pattern_type == "double_peak":
        peak_hour   = st.slider("First peak hour",  0.0, 24.0,  8.0, 0.5)
        peak_hour_2 = st.slider("Second peak hour", 0.0, 24.0, 16.0, 0.5)
        peak_width  = st.slider("Peak width (hours)", 0.5, 8.0, 2.5, 0.5)
        peak_weight = st.slider("Weight on first peak", 0.1, 0.9, 0.5, 0.05)
    sim_days = st.number_input("Simulated days", min_value=1, value=31, step=1)

# ── Cost & Revenue ───────────────────────────────────────────
with st.sidebar.expander("💰 Cost & Revenue", expanded=False):
    margin_kr_per_kg     = st.number_input("Revenue margin (kr/kg)", 0.0, value=float(st.session_state["margin_kr"]),     step=1.0)
    queue_cost_kr_per_hr = st.number_input("Queue cost (kr/trailer-hour)", 0.0, value=float(st.session_state["queue_cost_kr"]), step=50.0)
    if margin_kr_per_kg != st.session_state["margin_kr"]:
        st.session_state["margin_kr"] = margin_kr_per_kg
    if queue_cost_kr_per_hr != st.session_state["queue_cost_kr"]:
        st.session_state["queue_cost_kr"] = queue_cost_kr_per_hr
    st.caption("Annual staff cost per schedule (kr/year)")
    staff_costs = {}
    for lbl, default_v in _STAFF_DEFAULTS.items():
        stored_v = st.session_state["staff_costs"].get(lbl, default_v)
        v = st.number_input(lbl, min_value=0, value=int(stored_v), step=50_000, key=f"sc_{lbl}")
        staff_costs[lbl] = v
    st.session_state["staff_costs"] = staff_costs
    # Apply to economics module
    eco.MARGIN_KR_PER_KG     = st.session_state["margin_kr"]
    eco.QUEUE_COST_KR_PER_HR = st.session_state["queue_cost_kr"]
    eco.STAFF_ANNUAL_COST    = dict(st.session_state["staff_costs"])

# ── Database ─────────────────────────────────────────────────
with st.sidebar.expander("🗄️ Database", expanded=False):
    db_path = st.text_input("DuckDB file", value=st.session_state["db_path"])
    if db_path != st.session_state["db_path"]:
        st.session_state["db_path"] = db_path
    st.caption(f"{safe_count(db_path):,} runs stored")


# ============================================================
# Title
# ============================================================

st.title("🛢️ HyOps — Hydrogen Plant Simulation & Economics")
st.caption(
    f"**{TOPOLOGY.mode}** · {TOPOLOGY.total_electrolyzers()} EZ · "
    f"{TOPOLOGY.total_compressors()} comp · {TOPOLOGY.total_fill_lines()} fill lines · "
    f"{theo:.0f} kg/day · RAM {'🟢 ON' if RELIABILITY_ON else '⚪ OFF'} · "
    f"{avg_arrivals}/day ({pattern_type}) · {int(sim_days)} days"
)

tab_plant, tab_ops, tab_econ, tab_data = st.tabs([
    "🏗️ Plant & RAM", "🚛 Operations", "💰 Economics", "🗄️ Data",
])


# ────────────────────────────────────────────────────────────
# TAB 1 — Plant & RAM
# ────────────────────────────────────────────────────────────
with tab_plant:
    sub_arch, sub_nodes, sub_tl, sub_fmea = st.tabs([
        "🏗️ Architecture", "🔩 Nodes", "📉 Reliability Timeline", "📋 FMEA",
    ])

    # ── Architecture ──────────────────────────────────────────
    with sub_arch:
        st.header("Reliability Architecture")
        if st.button("▶ Draw architecture", type="primary", key="btn_arch"):
            with st.spinner("Drawing..."):
                with silence_show():
                    rpo.plot_plant_architecture(TOPOLOGY)
                show_figs()
        with st.expander("Topology summary"):
            st.code(TOPOLOGY.summary(), language=None)

    # ── Nodes ─────────────────────────────────────────────────
    with sub_nodes:
        st.header("Node RAM Parameters")
        st.caption(
            "Each **node** is a group of components modelled together. "
            "Weibull nodes use wear-out failure distributions. "
            "Auxiliary nodes are BOM rollups of exponential components from the Equipment Library."
        )

        # Read from session_state, mutate a local copy, write back on change
        p = dict(st.session_state["ram_params"])

        # ── Weibull nodes table ────────────────────────────────
        st.subheader("Weibull nodes")
        st.caption("β = shape (>1 → wear-out),  η = characteristic life (h),  MTTR = corrective repair time (h)")

        _WB_NODES = [
            ("Electrolyzer body", "ez",  "Housing, membrane assembly. Series with stacks and aux."),
            ("Stack",             "stk", "One stack per EZ. 1-of-N needed → proportional derate."),
            ("Compressor block",  "cb",  "Compressor main block. All of block + motor + seals must be up."),
            ("Compressor motor",  "cm",  "Compressor drive motor."),
            ("Compressor seals",  "cs",  "Seal system — faster wear, lower η."),
        ]
        hdr = st.columns([3, 1, 1, 1])
        hdr[0].markdown("**Node**")
        hdr[1].markdown("**β**")
        hdr[2].markdown("**η (h)**")
        hdr[3].markdown("**MTTR (h)**")
        st.divider()
        for label, key, tooltip in _WB_NODES:
            c0, c1, c2, c3 = st.columns([3, 1, 1, 1])
            c0.markdown(f"**{label}**")
            c0.caption(tooltip)
            p[f"{key}_beta"] = c1.number_input("β", min_value=0.5, max_value=10.0,
                value=float(p[f"{key}_beta"]), step=0.1, format="%.1f",
                key=f"ni_{key}_beta", label_visibility="collapsed")
            p[f"{key}_eta"]  = c2.number_input("η", min_value=1000,
                value=int(p[f"{key}_eta"]),  step=1000,
                key=f"ni_{key}_eta", label_visibility="collapsed")
            p[f"{key}_mttr"] = c3.number_input("MTTR", min_value=1,
                value=int(p[f"{key}_mttr"]), step=8,
                key=f"ni_{key}_mttr", label_visibility="collapsed")
            st.divider()

        # ── Planned Maintenance ────────────────────────────────
        st.subheader("Planned Maintenance")
        st.caption("Staggered across units: group A offset 0, group B offset interval/2.")
        _unit_to_h = {"hours": 1, "days": 24, "weeks": 168, "months": 730, "years": 8760}

        c1, c2 = st.columns(2)
        pm_int_val  = c1.number_input("PM interval", min_value=1, value=1, step=1, key="ni_pm_int_val")
        pm_int_unit = c2.selectbox("Unit##int", ["years","months","weeks","days","hours"],
                                    key="ni_pm_int_unit", label_visibility="hidden")
        p["pm_interval_h"] = int(pm_int_val * _unit_to_h[pm_int_unit])
        c1.caption(f"= {p['pm_interval_h']:,} h")

        c3, c4 = st.columns(2)
        pm_ez_val  = c3.number_input("EZ PM duration", min_value=1, value=3, step=1, key="ni_pm_ez_val")
        pm_ez_unit = c4.selectbox("Unit##ez", ["hours","days","weeks"],
                                   index=1, key="ni_pm_ez_unit", label_visibility="hidden")
        p["pm_ez_dur"] = int(pm_ez_val * _unit_to_h[pm_ez_unit])
        c3.caption(f"= {p['pm_ez_dur']} h")

        c5, c6 = st.columns(2)
        pm_comp_val  = c5.number_input("Comp PM duration", min_value=1, value=4, step=1, key="ni_pm_comp_val")
        pm_comp_unit = c6.selectbox("Unit##comp", ["hours","days","weeks"],
                                     index=1, key="ni_pm_comp_unit", label_visibility="hidden")
        p["pm_comp_dur"] = int(pm_comp_val * _unit_to_h[pm_comp_unit])
        c5.caption(f"= {p['pm_comp_dur']} h")

        # Write Weibull + PM back to session_state
        st.session_state["ram_params"] = p

        st.divider()

        # ── Node BOM ───────────────────────────────────────────
        st.subheader("Node BOM — auxiliary exponential components")
        st.caption(
            "Each auxiliary node is a series system of components from the Equipment Library. "
            "Edit quantities or add/remove. Equivalent MTBF and MTTR computed automatically."
        )
        bom = {n: dict(c) for n, c in st.session_state["bom"].items()}
        eq  = st.session_state["eq_lib"]

        _NODE_LABELS = {
            "ez_aux":        "EZ Aux — instrumentation & valves on each electrolyzer",
            "comp_aux":      "Comp Aux — instrumentation & rotating equipment on each compressor",
            "fill_line_aux": "Fill Line Aux — valves & instrumentation per fill line",
        }
        bom_dirty = False
        for node_key, node_label in _NODE_LABELS.items():
            with st.expander(f"📦 {node_label}", expanded=True):
                node_bom = bom[node_key]
                h0, h1, h2, h3 = st.columns([3, 1, 1, 1])
                h0.markdown("**Component**"); h1.markdown("**Qty**")
                h2.markdown("**Eff. MTBF (h)**"); h3.markdown("**Remove**")

                to_remove = []
                for comp, qty in list(node_bom.items()):
                    r0, r1, r2, r3 = st.columns([3, 1, 1, 1])
                    r0.markdown(f"`{comp}`")
                    new_qty = r1.number_input("qty", min_value=1, value=int(qty), step=1,
                        key=f"bom_{node_key}_{comp}", label_visibility="collapsed")
                    if new_qty != qty:
                        node_bom[comp] = new_qty
                        bom_dirty = True
                    if comp in eq and eq[comp]["mtbf"] > 0:
                        r2.caption(f"{eq[comp]['mtbf']/node_bom[comp]:,.0f}")
                    else:
                        r2.caption("—")
                    if r3.button("✕", key=f"rm_{node_key}_{comp}"):
                        to_remove.append(comp)

                for c in to_remove:
                    del node_bom[c]
                if to_remove:
                    bom_dirty = True

                available = [k for k in eq if k not in node_bom]
                if available:
                    ca, cb_ = st.columns([3, 1])
                    add_comp = ca.selectbox("Add from library", ["— select —"] + available,
                                            key=f"add_sel_{node_key}")
                    if cb_.button("Add", key=f"add_btn_{node_key}") and add_comp != "— select —":
                        node_bom[add_comp] = 1
                        bom_dirty = True

                # Rollup
                lam, wm, valid = 0.0, 0.0, True
                for comp, qty in node_bom.items():
                    if comp not in eq:
                        valid = False; break
                    lam += qty / eq[comp]["mtbf"]
                    wm  += (qty / eq[comp]["mtbf"]) * eq[comp]["mttr"]
                if valid and lam > 0:
                    st.info(f"**Equivalent node:** MTBF = {1/lam:,.0f} h  ·  MTTR = {wm/lam:.0f} h")
                elif not valid:
                    st.warning("Some BOM components missing from Equipment Library.")

        if bom_dirty:
            st.session_state["bom"] = bom
            st.rerun()

        st.divider()

        # ── Equipment Library ──────────────────────────────────
        st.subheader("Equipment Library — exponential component data")
        st.caption("Shared by all nodes. Built-in entries cannot be deleted.")
        eq = {k: dict(v) for k, v in st.session_state["eq_lib"].items()}

        h0, h1, h2, h3 = st.columns([3, 2, 2, 1])
        h0.markdown("**Component**"); h1.markdown("**MTBF (h)**")
        h2.markdown("**MTTR (h)**"); h3.markdown("")
        st.divider()

        eq_dirty = False
        to_delete = []
        for eq_name, vals in list(eq.items()):
            c0, c1, c2, c3 = st.columns([3, 2, 2, 1])
            c0.markdown(f"`{eq_name}`")
            new_mtbf = c1.number_input("MTBF", min_value=1000, value=int(vals["mtbf"]), step=1000,
                key=f"eq_mtbf_{eq_name}", label_visibility="collapsed")
            new_mttr = c2.number_input("MTTR", min_value=1, value=int(vals["mttr"]), step=8,
                key=f"eq_mttr_{eq_name}", label_visibility="collapsed")
            if new_mtbf != vals["mtbf"] or new_mttr != vals["mttr"]:
                eq[eq_name] = {"mtbf": new_mtbf, "mttr": new_mttr}
                eq_dirty = True
            is_builtin = eq_name in _EQ_LIB_DEFAULTS
            if c3.button("🗑", key=f"del_{eq_name}", disabled=is_builtin,
                         help="Cannot delete built-in components" if is_builtin else "Remove"):
                to_delete.append(eq_name)

        for d in to_delete:
            del eq[d]
        if to_delete:
            eq_dirty = True

        with st.expander("➕ Add component to library"):
            nc0, nc1, nc2, nc3 = st.columns([3, 2, 2, 1])
            new_name = nc0.text_input("Name", key="new_eq_name", placeholder="e.g. flow_meter")
            new_mtbf = nc1.number_input("MTBF (h)", min_value=1000, value=100_000, step=1000, key="new_mtbf")
            new_mttr = nc2.number_input("MTTR (h)", min_value=1, value=48, step=8, key="new_mttr")
            if nc3.button("Add", key="btn_add_eq"):
                if new_name and new_name not in eq:
                    eq[new_name] = {"mtbf": int(new_mtbf), "mttr": int(new_mttr)}
                    eq_dirty = True
                elif new_name in eq:
                    st.warning("Already exists.")

        if eq_dirty:
            st.session_state["eq_lib"] = eq
            st.rerun()

    # ── Reliability Timeline ───────────────────────────────────
    with sub_tl:
        st.header("Standalone Availability Timeline")
        c1, c2, c3 = st.columns(3)
        tl_years  = c1.slider("Years to simulate", 1, 20, 10, key="tl_years")
        tl_seed   = c2.number_input("Random seed", min_value=0, value=99, step=1, key="tl_seed")
        roll_days = c3.slider("Rolling avg (days)", 1, 90, 30, key="tl_roll")

        if st.button("▶ Run timeline", type="primary", key="btn_tl"):
            # Capture all config at button-press time — not at render time
            _params, _pm, _eq_lib, _bom = _build_ram_dicts()
            with st.spinner("Running..."):
                tl_result = rel.run_reliability_timeline(
                    TOPOLOGY, years=int(tl_years), random_seed=int(tl_seed),
                    reliability_params=_params, pm_config=_pm,
                    eq_lib=_eq_lib, bom=_bom,
                )
            st.session_state["tl_result"]    = tl_result
            st.session_state["tl_seed_used"] = int(tl_seed)

        if st.session_state["tl_result"] is not None:
            import numpy as np
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            tl       = st.session_state["tl_result"]
            cap      = np.array(tl["capacity_history"])
            pm_hist  = np.array(tl["pm_history"], dtype=bool)
            n_years  = len(cap) // 8760
            years_ax = np.arange(len(cap)) / 8760

            BAND_COLORS = {
                "100%":   "#1D9E75",
                "75-99%": "#639922",
                "50-74%": "#EF9F27",
                "25-49%": "#D85A30",
                "0-24%":  "#E24B4A",
            }
            BANDS = [
                (1.00, 1.01, "100%"),
                (0.75, 1.00, "75-99%"),
                (0.50, 0.75, "50-74%"),
                (0.25, 0.50, "25-49%"),
                (0.00, 0.25, "0-24%"),
            ]

            overall = cap.mean() * 100
            n_zero  = int((cap == 0).sum())
            n_deg   = int(((cap > 0) & (cap < 1)).sum())
            n_pm    = int(pm_hist.sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Mean availability", f"{overall:.2f}%")
            k2.metric("Zero-output hours", f"{n_zero:,} h")
            k3.metric("Degraded hours",    f"{n_deg:,} h")
            k4.metric("Hours in PM",       f"{n_pm:,} h")

            with st.expander("Failure summary"):
                s = tl["model"].summary()
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("EZ failures",       s["electrolyzer_failures"])
                sc2.metric("Stack failures",     s["stack_failures"])
                sc3.metric("Comp failures",      s["compressor_failures"])
                sc4.metric("Fill line failures", s["fill_line_failures"])
                sc5.metric("PM events",          s["pm_events"])

            tl_hourly, tl_monthly, tl_yearly, tl_exceed = st.tabs([
                "📈 Hourly", "📅 Monthly avg", "📊 Yearly bars", "📉 Exceedance"
            ])

            with tl_hourly:
                roll_w = roll_days * 24
                roll = np.convolve(cap, np.ones(roll_w)/roll_w, mode="same") if len(cap) >= roll_w else cap
                fig = go.Figure()
                for lo, hi, label in BANDS:
                    mask   = (cap >= lo) & (cap < hi + 0.001)
                    filled = np.where(mask, cap * 100, np.nan)
                    fig.add_trace(go.Scatter(
                        x=years_ax, y=filled, fill="tozeroy", mode="none",
                        fillcolor=BAND_COLORS[label], opacity=0.55, name=label,
                        hovertemplate=f"{label}<br>Year: %{{x:.2f}}<br>Capacity: %{{y:.1f}}%<extra></extra>",
                    ))
                pm_starts = np.where(np.diff(pm_hist.astype(int)) == 1)[0]
                pm_ends   = np.where(np.diff(pm_hist.astype(int)) == -1)[0]
                if pm_hist[0]:  pm_starts = np.concatenate([[0], pm_starts])
                if pm_hist[-1]: pm_ends   = np.concatenate([pm_ends, [len(pm_hist)-1]])
                for s_i, e_i in zip(pm_starts[:50], pm_ends[:50]):
                    fig.add_vrect(x0=years_ax[s_i], x1=years_ax[min(e_i, len(years_ax)-1)],
                                  fillcolor="#2196F3", opacity=0.10, layer="below", line_width=0)
                fig.add_trace(go.Scatter(
                    x=years_ax, y=roll * 100, mode="lines",
                    line=dict(color="#1a1a2e", width=1.5), name=f"{roll_days}d rolling mean",
                    hovertemplate="Year: %{x:.2f}<br>Rolling avg: %{y:.1f}%<extra></extra>",
                ))
                for y in range(1, n_years + 1):
                    fig.add_vline(x=y, line_color="#aaaaaa", line_width=0.5, line_dash="dash")
                fig.update_layout(
                    height=420,
                    xaxis=dict(title="Year", tickmode="linear", dtick=1,
                               rangeslider=dict(visible=True, thickness=0.06)),
                    yaxis=dict(title="Plant capacity (%)", range=[0, 108]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    margin=dict(l=50, r=20, t=40, b=60),
                    hovermode="x unified", plot_bgcolor="#F8F7F4", paper_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)

            with tl_monthly:
                n_months  = len(cap) // 730
                mo_avail  = [cap[i*730:(i+1)*730].mean()*100 for i in range(n_months)]
                mo_x      = [(i + 0.5) / 12 for i in range(n_months)]
                mo_colors = [
                    BAND_COLORS["100%"]   if v >= 100 else
                    BAND_COLORS["75-99%"] if v >= 75  else
                    BAND_COLORS["50-74%"] if v >= 50  else
                    BAND_COLORS["25-49%"] if v >= 25  else
                    BAND_COLORS["0-24%"]  for v in mo_avail
                ]
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=mo_x, y=mo_avail, marker_color=mo_colors, name="Monthly avg",
                    hovertemplate="Month %{x:.1f}<br>Avg availability: %{y:.1f}%<extra></extra>",
                ))
                fig2.add_hline(y=overall, line_dash="dash", line_color="#333",
                               annotation_text=f"Mean {overall:.1f}%", annotation_position="top right")
                for y in range(1, n_years + 1):
                    fig2.add_vline(x=y, line_color="#aaaaaa", line_width=0.5, line_dash="dash")
                fig2.update_layout(
                    height=380,
                    xaxis=dict(title="Year", tickmode="linear", dtick=1),
                    yaxis=dict(title="Avg availability (%)", range=[0, 110]),
                    plot_bgcolor="#F8F7F4", paper_bgcolor="white",
                    margin=dict(l=50, r=20, t=30, b=50),
                )
                st.plotly_chart(fig2, use_container_width=True)

            with tl_yearly:
                yr_avail = [cap[y*8760:(y+1)*8760].mean()*100 for y in range(n_years)]
                yr_x     = list(range(1, n_years + 1))
                yr_colors = [
                    BAND_COLORS["100%"]   if v >= 95 else
                    BAND_COLORS["75-99%"] if v >= 85 else
                    BAND_COLORS["50-74%"] if v >= 70 else
                    BAND_COLORS["25-49%"] if v >= 50 else
                    BAND_COLORS["0-24%"]  for v in yr_avail
                ]
                bottleneck_kg_hr = min(
                    TOPOLOGY.theoretical_capacity_kg_per_hr(),
                    TOPOLOGY.theoretical_compressor_capacity_kg_per_hr()
                )
                yr_lost = [(1 - v/100) * bottleneck_kg_hr * 8760 / 1000 for v in yr_avail]
                fig3 = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=("Yearly availability", "Lost production (t H₂)"),
                    vertical_spacing=0.12, row_heights=[0.6, 0.4],
                )
                fig3.add_trace(go.Bar(
                    x=yr_x, y=yr_avail, marker_color=yr_colors, name="Availability",
                    text=[f"{v:.1f}%" for v in yr_avail], textposition="outside",
                    hovertemplate="Year %{x}<br>Availability: %{y:.1f}%<extra></extra>",
                ), row=1, col=1)
                fig3.add_hline(y=overall, line_dash="dash", line_color="#333",
                               annotation_text=f"Mean {overall:.1f}%", row=1, col=1)
                fig3.add_trace(go.Bar(
                    x=yr_x, y=yr_lost, marker_color="#D85A30", name="Lost production",
                    hovertemplate="Year %{x}<br>Lost: %{y:.1f} t H₂<extra></extra>",
                ), row=2, col=1)
                fig3.update_layout(
                    height=480, showlegend=False,
                    plot_bgcolor="#F8F7F4", paper_bgcolor="white",
                    xaxis2=dict(title="Year", tickmode="linear", dtick=1),
                    yaxis=dict(title="Availability (%)", range=[0, 115]),
                    yaxis2=dict(title="Lost prod. (t H₂)"),
                    margin=dict(l=60, r=20, t=50, b=50),
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.caption(
                    f"Total lost: **{sum(yr_lost):,.1f} t H₂** over {n_years} years "
                    f"at {bottleneck_kg_hr:.2f} kg/hr plant capacity"
                )

            with tl_exceed:
                st.caption(
                    "For a given capacity level X, what fraction of hours is plant capacity ≥ X? "
                    "A steep drop near 100% = frequent partial outages."
                )
                levels = np.linspace(0, 1, 500)
                exceed = np.array([(cap >= lv).mean() * 100 for lv in levels])
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=levels * 100, y=exceed, mode="lines",
                    line=dict(color="#533483", width=2),
                    fill="tozeroy", fillcolor="rgba(83,52,131,0.12)",
                    hovertemplate="Capacity ≥ %{x:.1f}%<br>%{y:.1f}% of hours<extra></extra>",
                    name="Exceedance",
                ))
                for pct, label, color in [(50,"P50","#1D9E75"),(90,"P90","#EF9F27"),(99,"P99","#E24B4A")]:
                    idx = np.searchsorted(-exceed, -pct)
                    if 0 < idx < len(levels):
                        fig4.add_vline(x=levels[idx]*100, line_dash="dot", line_color=color,
                                       annotation_text=f"{label}: {levels[idx]*100:.1f}%",
                                       annotation_position="top right")
                fig4.update_layout(
                    height=380,
                    xaxis=dict(title="Plant capacity (%)"),
                    yaxis=dict(title="% of hours capacity ≥ X", range=[0, 105]),
                    plot_bgcolor="#F8F7F4", paper_bgcolor="white",
                    margin=dict(l=60, r=20, t=30, b=50), hovermode="x",
                )
                st.plotly_chart(fig4, use_container_width=True)

    # ── FMEA ──────────────────────────────────────────────────
    with sub_fmea:
        st.header("FMEA — Failure Mode & Effects Analysis")
        if st.button("▶ Generate FMEA", type="primary", key="btn_fmea"):
            st.session_state["fmea_df"] = generate_fmea(TOPOLOGY)

        if st.session_state["fmea_df"] is not None:
            fmea_df = st.session_state["fmea_df"]
            edited = st.data_editor(
                fmea_df,
                column_config={
                    "Severity":   st.column_config.NumberColumn("Severity (1-10)",   min_value=1, max_value=10),
                    "Occurrence": st.column_config.NumberColumn("Occurrence (1-10)", min_value=1, max_value=10),
                    "Detection":  st.column_config.NumberColumn("Detection (1-10)",  min_value=1, max_value=10),
                    "RPN":        st.column_config.NumberColumn(disabled=True),
                },
                hide_index=True, use_container_width=True, key="fmea_editor",
            )
            edited["RPN"] = edited["Severity"] * edited["Occurrence"] * edited["Detection"]
            edited = edited.sort_values("RPN", ascending=False).reset_index(drop=True)
            high = edited[edited["RPN"] >= 200]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", len(edited))
            m2.metric("🔴 High (>=200)", len(high))
            m3.metric("🟠 Medium (100-199)", len(edited[(edited["RPN"]>=100)&(edited["RPN"]<200)]))
            m4.metric("🟢 Low (<100)", len(edited[edited["RPN"]<100]))
            if not high.empty:
                st.subheader("🔴 High priority")
                st.dataframe(high[["ID","Node","Failure Mode","System Effect","RPN","Recommended Action"]],
                              hide_index=True, use_container_width=True)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, max(4, len(edited)*0.32)))
            colors = ["#E24B4A" if r>=200 else "#EF9F27" if r>=100 else "#1D9E75" for r in edited["RPN"]]
            ax.barh(range(len(edited)), edited["RPN"], color=colors)
            ax.set_yticks(range(len(edited)))
            ax.set_yticklabels(edited["ID"]+"  "+edited["Failure Mode"], fontsize=7)
            ax.set_xlabel("RPN"); ax.set_title("FMEA Risk Priority")
            ax.axvline(200, color="#E24B4A", linestyle="--", lw=0.8, label="High")
            ax.axvline(100, color="#EF9F27", linestyle="--", lw=0.8, label="Medium")
            ax.legend(fontsize=8); ax.invert_yaxis()
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.download_button("⬇ Export FMEA (CSV)",
                                data=edited.to_csv(index=False).encode("utf-8"),
                                file_name="fmea.csv", mime="text/csv")


# ────────────────────────────────────────────────────────────
# TAB 2 — Operations
# ────────────────────────────────────────────────────────────
with tab_ops:
    sub_single, sub_schedule = st.tabs(["🔬 Single run", "📋 Schedule comparison"])

    with sub_single:
        st.header("Single Simulation Run")
        schedule_label = st.selectbox("Staff schedule", ALL_SCHEDULES, index=0, key="ops_sched")
        if st.button("▶ Run simulation", type="primary", key="btn_single"):
            # Capture all state at press time
            _rel_model = make_rel_model(TOPOLOGY, RELIABILITY_SEED, RELIABILITY_ON)
            _plant     = po.HydrogenPlant(topology=TOPOLOGY, step_minutes=1)
            _schedule  = make_schedule(schedule_label)
            _containers = make_container_types(frac_a, frac_b, frac_c)
            _pattern    = make_arrival_pattern(pattern_type, peak_hour, peak_hour_2, peak_width, peak_weight)
            with st.spinner("Simulating..."):
                result = po.run_simulation(
    container_types=_containers,
    plant=_plant,
    avg_arrivals_per_day=float(avg_arrivals),
    days=int(sim_days),
    step_minutes=1,
    schedule=_schedule,
    arrival_pattern=_pattern,
    reliability_model=_rel_model,
)
                kpis = rpo.compute_kpis(result, _plant, _containers,
                         float(avg_arrivals), int(sim_days))
                econ = eco.run_economics(
                    result, kpis, schedule_label, int(sim_days),
                    margin_kr_per_kg=float(st.session_state["margin_kr"]),
                    queue_cost_kr_per_hr=float(st.session_state["queue_cost_kr"]),
                    staff_cost_overrides=st.session_state["staff_costs"],
                )
            st.session_state["single_result"] = (result, kpis, econ)

        if st.session_state["single_result"] is not None:
            result, kpis, econ = st.session_state["single_result"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("H2 dispensed",      f"{kpis['total_dispensed_kg']:.0f} kg")
            c2.metric("Plant utilization",  f"{kpis['plant_utilization']*100:.1f}%")
            c3.metric("Avg ext. queue",     f"{kpis['avg_external_queue']:.2f} trailers")
            c4.metric("Net result",         f"{econ['net_kr']:,.0f} kr")
            with st.expander("Full KPIs"):
                import pandas as pd
                st.dataframe(pd.DataFrame([kpis]).T.rename(columns={0: "value"}), use_container_width=True)
            with st.expander("Economics"):
                import pandas as pd
                st.dataframe(pd.DataFrame([econ]).T.rename(columns={0: "value"}), use_container_width=True)
            with st.expander("Operations plots"):
                with silence_show():
                    rpo.plot_results(result)
                show_figs()

    with sub_schedule:
        st.header("Schedule Comparison")
        compare_scheds = st.multiselect("Schedules to compare", ALL_SCHEDULES,
                                         default=["unmanned","8-16_closed","8-20_open","24_7"],
                                         key="cmp_scheds")
        if st.button("▶ Compare schedules", type="primary", key="btn_cmp") and compare_scheds:
            _containers = make_container_types(frac_a, frac_b, frac_c)
            _pattern    = make_arrival_pattern(pattern_type, peak_hour, peak_hour_2, peak_width, peak_weight)
            rows = []
            with st.spinner("Running..."):
                for lbl in compare_scheds:
                    _plant  = po.HydrogenPlant(topology=TOPOLOGY, step_minutes=1)
                    _rel    = make_rel_model(TOPOLOGY, RELIABILITY_SEED, RELIABILITY_ON)
                    result  = po.run_simulation(
                        plant=_plant, days=int(sim_days), schedule=make_schedule(lbl),
                        container_types=_containers, avg_arrivals_per_day=float(avg_arrivals),
                        arrival_pattern=_pattern, reliability_model=_rel,
                    )
                    kpis = rpo.compute_kpis(result)
                    econ = eco.run_economics(
                        result, kpis, lbl, int(sim_days),
                        margin_kr_per_kg=float(st.session_state["margin_kr"]),
                        queue_cost_kr_per_hr=float(st.session_state["queue_cost_kr"]),
                        staff_cost_overrides=st.session_state["staff_costs"],
                    )
                    rows.append({"schedule": lbl, **kpis, **{k: v for k, v in econ.items() if k != "schedule_label"}})
            import pandas as pd
            df = pd.DataFrame(rows).set_index("schedule")
            st.dataframe(df, use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 3 — Economics (Monte Carlo)
# ────────────────────────────────────────────────────────────
with tab_econ:
    sub_sweep, sub_charts = st.tabs(["▶ Run sweep", "📊 Charts"])

    with sub_sweep:
        st.header("Monte Carlo Sweep")
        c1, c2 = st.columns(2)
        with c1:
            rate_text = st.text_input("Arrival rates (comma-separated)", value="0.1, 0.3, 0.5, 1.0, 2.0, 3.0")
            sw_scheds = st.multiselect("Schedules", ALL_SCHEDULES,
                                        default=["unmanned","8-16_closed","8-20_closed","8-24_open","24_7"],
                                        key="sw_scheds")
        with c2:
            n_runs     = st.number_input("Seeds per scenario", min_value=1, value=10, step=1)
            sweep_days = st.number_input("Days per run", min_value=1, value=31, step=1)
        rel_opt = st.radio("Reliability dimension",
                            ["Off only","On only","Both (off + on)"], index=0, horizontal=True)
        rel_settings = (
            [("off", False)] if rel_opt == "Off only" else
            [("on",  True)]  if rel_opt == "On only"  else
            mc.default_reliability_settings()
        )
        sw_pat = st.selectbox("Arrival timing", ["uniform","single_peak","double_peak","All three"], key="sw_pat")
        sw_patterns = (
            mc.default_arrival_patterns() if sw_pat == "All three"
            else [(p[0], p[1]) for p in mc.default_arrival_patterns() if p[0] == sw_pat]
        )
        try:
            arrival_rates = [float(x.strip()) for x in rate_text.split(",") if x.strip()]
        except ValueError:
            arrival_rates = []
            st.error("Invalid arrival rates.")
        if arrival_rates and sw_scheds:
            n_scen = len(arrival_rates)*len(sw_scheds)*len(rel_settings)*len(sw_patterns)
            st.info(f"**{n_scen} scenarios × {int(n_runs)} seeds = {n_scen*int(n_runs)} runs**")
        if st.button("▶ Run sweep", type="primary", key="btn_sweep") and arrival_rates and sw_scheds:
            # Capture state at press time
            _db = st.session_state["db_path"]
            t0 = time.time()
            with st.spinner("Running sweep..."):
                mc.run_sweep(
                    arrival_rates        = arrival_rates,
                    schedules            = [(l, make_schedule(l)) for l in sw_scheds],
                    plant_configs        = [{"label": topology_mode+"_plant", "topology": TOPOLOGY}],
                    container_mixes      = [{"label": "current_mix",
                                             "types": make_container_types(frac_a, frac_b, frac_c)}],
                    arrival_patterns     = sw_patterns,
                    reliability_settings = rel_settings,
                    n_runs               = int(n_runs),
                    days                 = int(sweep_days),
                    step_minutes         = 1,
                    db_path              = _db,
                    verbose              = False,
                )
            st.success(f"Done in {time.time()-t0:.1f}s — {safe_count(_db):,} runs in DB.")

    with sub_charts:
        st.header("Economics Charts")
        _db = st.session_state["db_path"]
        try:
            import pandas as pd
            raw = mc.query(_db)
        except Exception:
            import pandas as pd
            raw = pd.DataFrame()
        if raw.empty:
            st.info("No sweep data yet — run a Monte Carlo sweep first.")
        else:
            econ_df = eco.add_economics_columns(
                raw,
                margin_kr_per_kg=float(st.session_state["margin_kr"]),
                queue_cost_kr_per_hr=float(st.session_state["queue_cost_kr"]),
                staff_cost_overrides=st.session_state["staff_costs"],
            )
            f1, f2, f3, f4 = st.columns(4)
            mix_f = f1.selectbox("Mix",   sorted(econ_df["mix_label"].unique()),  key="ef_mix")
            pcol  = "plant_label" if "plant_label" in econ_df.columns else "topology_mode"
            pf    = f2.selectbox("Plant", sorted(econ_df[pcol].unique()),          key="ef_plant")
            hr    = f3.selectbox("Headline rate", sorted(econ_df["arrival_rate"].unique()), key="ef_rate")
            rel_f = None
            if "reliability_label" in econ_df.columns:
                rel_f = f4.selectbox("Reliability",
                                      sorted(econ_df["reliability_label"].dropna().unique()),
                                      key="ef_rel")
            filtered = econ_df[(econ_df["mix_label"]==mix_f) & (econ_df[pcol]==pf)]
            if rel_f:
                filtered = filtered[filtered["reliability_label"]==rel_f]
            if filtered.empty:
                st.warning("No rows match.")
            else:
                headline    = filtered[filtered["arrival_rate"]==hr]
                sched_order = [s for s in ALL_SCHEDULES if s in headline["schedule_label"].unique()]
                summary     = eco.economics_summary(headline, group_by=["schedule_label"])
                summary     = (summary.set_index("schedule_label")
                               .reindex(sched_order).dropna(how="all").reset_index())
                n_seeds = (headline.groupby("schedule_label")["seed"].nunique().max()
                           if "seed" in headline.columns else len(headline))
                st.subheader(f"Net result spread — {hr} arrivals/day")
                fig, _ = ep.plot_net_result_spread(headline, sched_order, hr, n_seeds)
                st.pyplot(fig)
                st.subheader("Revenue vs. cost")
                fig2, _ = ep.plot_stacked_bar(summary)
                st.pyplot(fig2)
                st.subheader("Sensitivity vs. arrival rate")
                all_s   = [s for s in ALL_SCHEDULES if s in filtered["schedule_label"].unique()]
                filters = {"reliability_label": rel_f} if rel_f else None
                fig3, _ = ep.plot_sensitivity(filtered, all_s, filters=filters)
                st.pyplot(fig3)


# ────────────────────────────────────────────────────────────
# TAB 4 — Data
# ────────────────────────────────────────────────────────────
with tab_data:
    st.header("Data & Export")
    _db = st.session_state["db_path"]
    try:
        import pandas as pd
        full_df = mc.query(_db)
    except Exception:
        import pandas as pd
        full_df = pd.DataFrame()
    if full_df.empty:
        st.info("No sweep data yet.")
    else:
        st.subheader("Custom SQL")
        default_sql = (
            "SELECT arrival_rate, schedule_label, reliability_label,\n"
            "       COUNT(*) AS n_runs,\n"
            "       ROUND(AVG(ext_wait_avg),1) AS ext_wait_avg,\n"
            "       ROUND(AVG(plant_utilization)*100,1) AS util_pct,\n"
            "       ROUND(AVG(total_dispensed_kg),0) AS avg_dispensed_kg\n"
            "FROM runs\n"
            "GROUP BY arrival_rate, schedule_label, reliability_label\n"
            "ORDER BY arrival_rate, schedule_label LIMIT 50"
        )
        sql = st.text_area("SQL", value=default_sql, height=160)
        if st.button("▶ Run query", key="btn_sql"):
            try:
                st.dataframe(mc.query(_db, sql), hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"Query failed: {e}")
        st.subheader("Export")
        c1, c2 = st.columns(2)
        with c1:
            grp = [c for c in ["arrival_rate","schedule_label","mix_label",
                                "topology_mode","n_fill_lines","reliability_label"]
                   if c in full_df.columns]
            st.download_button("⬇ KPI summary (CSV)",
                                data=mc.summary_by(_db, grp).to_csv(index=False).encode("utf-8"),
                                file_name="mc_summary.csv", mime="text/csv")
        with c2:
            edf  = eco.add_economics_columns(
                full_df,
                margin_kr_per_kg=float(st.session_state["margin_kr"]),
                queue_cost_kr_per_hr=float(st.session_state["queue_cost_kr"]),
                staff_cost_overrides=st.session_state["staff_costs"],
            )
            cols = [c for c in ["arrival_rate","schedule_label","reliability_label",
                                  "total_dispensed_kg","revenue_kr_annual",
                                  "staff_cost_kr_annual","queue_cost_kr_annual","net_kr_annual"]
                    if c in edf.columns]
            st.download_button("⬇ Economics (CSV)",
                                data=edf[cols].to_csv(index=False).encode("utf-8"),
                                file_name="economics_summary.csv", mime="text/csv")
        st.subheader("Raw data (first 50 rows)")
        st.dataframe(full_df.head(50), hide_index=True, use_container_width=True)
