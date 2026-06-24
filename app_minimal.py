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
# Helpers
# ============================================================

ALL_SCHEDULES = [
    "unmanned", "8-16_closed", "8-20_closed", "8-24_closed",
    "8-16_open", "8-20_open", "8-24_open", "24_7",
]

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
        return po.ArrivalPattern(pattern_type="single_peak",
                                  peak_hour=ph, peak_width_hours=pw)
    return po.ArrivalPattern(pattern_type="double_peak",
                              peak_hour=ph, peak_hour_2=ph2,
                              peak_width_hours=pw, peak_weight=pwt)

def make_plant():
    return po.HydrogenPlant(topology=TOPOLOGY, step_minutes=1)

def make_rel_model(seed=None):
    if not RELIABILITY_ON:
        return None
    return rel.ReliabilityModel(
        TOPOLOGY,
        random_seed=int(seed or RELIABILITY_SEED),
        reliability_params=RAM_PARAMS,
        pm_config=RAM_PM,
        eq_lib=RAM_EQ_LIB,
        bom=RAM_BOM,
    )

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

def safe_count(path):
    try:
        return mc.run_count(path)
    except Exception:
        return 0

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
            "Failure Mode": "Body / housing failure",
            "Cause": "Wear, corrosion (Weibull b=2.5)",
            "Local Effect": "Full EZ offline",
            "System Effect": f"~{100/n_ez:.0f}% capacity loss",
            "Safeguard": "Isolation valves, pressure relief",
            "Severity": 8, "Occurrence": 3, "Detection": 4,
            "Recommended Action": "Annual inspection",
        })
        for s in range(stacks):
            rows.append({
                "ID": f"EZ-{i+1}-STK{s+1}", "Node": f"EZ {i+1} Stack {s+1}", "Type": "Stack",
                "Failure Mode": "Stack degradation",
                "Cause": "Membrane wear (Weibull b=2.5)",
                "Local Effect": f"EZ {i+1} output -{100//stacks:.0f}%",
                "System Effect": f"~{100/n_ez/stacks:.0f}% loss",
                "Safeguard": "H2 purity monitor",
                "Severity": 6, "Occurrence": 5, "Detection": 3,
                "Recommended Action": "Quarterly test",
            })
    for j in range(n_comp):
        sys_eff = "Full plant offline" if n_comp == 1 else f"~{100/n_comp:.0f}% loss"
        for fm, sev, occ, det in [
            ("Seal failure", 9, 4, 3),
            ("Motor failure", 8, 3, 4),
            ("Block/valve failure", 8, 3, 5),
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
# Sidebar
# ============================================================

st.sidebar.title("⚙️ HyOps Configuration")

with st.sidebar.expander("🏗️ Plant & Topology", expanded=True):
    topology_mode = st.selectbox(
        "Wiring mode", ["common", "pooled_ez_dedicated_comp", "trains"],
    )
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
    st.caption(f"**{TOPOLOGY.total_electrolyzers()} EZ · {TOPOLOGY.total_compressors()} comp · "
               f"{TOPOLOGY.total_fill_lines()} fill lines**  \n"
               f"Theoretical: **{theo:.1f} kg/day** ({bneck} limited)")

with st.sidebar.expander("⚡ Reliability (RAM)", expanded=False):
    RELIABILITY_ON   = st.toggle("Simulate with live availability", value=False)
    RELIABILITY_SEED = 42
    if RELIABILITY_ON:
        RELIABILITY_SEED = st.number_input("Reliability seed", min_value=0, value=42, step=1)

# ── RAM state defaults (edited in Nodes tab) ───────────────────────────────
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
    "ez_aux":       {"control_valve": 2, "pressure_tx": 2, "temp_tx": 2},
    "comp_aux":     {"control_valve": 2, "lube_oil_pump": 1, "fan": 2,
                     "pressure_tx": 2, "temp_tx": 2, "vibration_sensor": 1},
    "fill_line_aux":{"manual_valve": 2, "control_valve": 1,
                     "pressure_tx": 1, "flow_tx": 1, "check_valve": 1},
}
if "eq_lib" not in st.session_state:
    st.session_state["eq_lib"] = {k: {"mtbf": v[0], "mttr": v[1]} for k, v in _EQ_LIB_DEFAULTS.items()}
if "bom" not in st.session_state:
    st.session_state["bom"] = {node: dict(comps) for node, comps in _BOM_DEFAULTS.items()}
if "ram_params" not in st.session_state:
    st.session_state["ram_params"] = {
        "ez_beta": 2.5, "ez_eta": 50000, "ez_mttr": 72,
        "stk_beta": 2.5, "stk_eta": 55000, "stk_mttr": 168,
        "cb_beta": 3.0, "cb_eta": 40000, "cb_mttr": 240,
        "cm_beta": 3.0, "cm_eta": 40000, "cm_mttr": 168,
        "cs_beta": 3.0, "cs_eta": 20000, "cs_mttr": 48,
        "pm_interval_h": 8760, "pm_ez_dur": 72, "pm_comp_dur": 96,
    }

def _build_ram_dicts():
    p = st.session_state["ram_params"]
    return (
        {
            "electrolyzer_body": dict(beta=p["ez_beta"],  eta=p["ez_eta"],  mttr_corrective=p["ez_mttr"]),
            "stack":             dict(beta=p["stk_beta"], eta=p["stk_eta"], mttr_corrective=p["stk_mttr"]),
            "compressor_block":  dict(beta=p["cb_beta"],  eta=p["cb_eta"],  mttr_corrective=p["cb_mttr"]),
            "compressor_motor":  dict(beta=p["cm_beta"],  eta=p["cm_eta"],  mttr_corrective=p["cm_mttr"]),
            "compressor_seals":  dict(beta=p["cs_beta"],  eta=p["cs_eta"],  mttr_corrective=p["cs_mttr"]),
        },
        {
            "interval_h":              int(p["pm_interval_h"]),
            "electrolyzer_duration_h": int(p["pm_ez_dur"]),
            "compressor_duration_h":   int(p["pm_comp_dur"]),
        },
        dict(st.session_state["eq_lib"]),
        dict(st.session_state["bom"]),
    )

RAM_PARAMS, RAM_PM, RAM_EQ_LIB, RAM_BOM = _build_ram_dicts()

with st.sidebar.expander("🚛 Container fleet", expanded=False):
    frac_a = st.slider("Type-A  (1000 kg)", 0.0, 1.0, 0.3, 0.05, key="fa")
    frac_b = st.slider("Type-B  (600 kg)",  0.0, 1.0, 0.5, 0.05, key="fb")
    frac_c = st.slider("Type-C  (300 kg)",  0.0, 1.0, 0.2, 0.05, key="fc")

with st.sidebar.expander("📦 Arrivals", expanded=False):
    avg_arrivals = st.number_input("Avg containers / day", min_value=0.01, value=3.0, step=0.5, format="%.1f")
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

with st.sidebar.expander("💰 Cost & Revenue", expanded=False):
    margin_kr_per_kg     = st.number_input("Revenue margin (kr/kg)", 0.0, value=30.0, step=1.0)
    queue_cost_kr_per_hr = st.number_input("Queue cost (kr/trailer-hour)", 0.0, value=1500.0, step=50.0)
    st.caption("Annual staff cost per schedule (kr/year)")
    default_staff = {
        "unmanned": 0, "8-16_closed": 650_000, "8-20_closed": 950_000,
        "8-24_closed": 1_550_000, "8-16_open": 850_000,
        "8-20_open": 1_200_000, "8-24_open": 1_750_000, "24_7": 2_800_000,
    }
    staff_costs = {
        lbl: st.number_input(lbl, min_value=0, value=v, step=50_000, key=f"sc_{lbl}")
        for lbl, v in default_staff.items()
    }
    eco.MARGIN_KR_PER_KG     = margin_kr_per_kg
    eco.QUEUE_COST_KR_PER_HR = queue_cost_kr_per_hr
    eco.STAFF_ANNUAL_COST    = dict(staff_costs)

with st.sidebar.expander("🗄️ Database", expanded=False):
    db_path = st.text_input("DuckDB file", value="hydrogen_mc.duckdb")
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

    with sub_arch:
        st.header("Reliability Architecture")
        if st.button("▶ Draw architecture", type="primary", key="btn_arch"):
            with st.spinner("Drawing..."):
                with silence_show():
                    rpo.plot_plant_architecture(TOPOLOGY)
                show_figs()
        with st.expander("Topology summary"):
            st.code(TOPOLOGY.summary(), language=None)

    # ── Nodes tab ─────────────────────────────────────────────
    with sub_nodes:
        st.header("Node RAM Parameters")
        st.caption(
            "Each **node** is a group of components modelled together. "
            "Weibull nodes use wear-out failure distributions. "
            "Auxiliary nodes are BOM rollups of exponential components from the Equipment Library."
        )

        p = st.session_state["ram_params"]

        # ── Weibull nodes ──────────────────────────────────────
        st.subheader("Weibull nodes")
        st.caption("β = shape (>1 → wear-out),  η = characteristic life (h),  MTTR = corrective repair time (h)")

        _WB_NODES = [
            ("Electrolyzer body", "ez",  "EZ body — housing, membrane assembly. Series with stacks and aux."),
            ("Stack",             "stk", "One stack per electrolyzer. 1-of-N needed → proportional derate."),
            ("Compressor block",  "cb",  "Compressor main block. All of block+motor+seals must be up."),
            ("Compressor motor",  "cm",  "Compressor drive motor."),
            ("Compressor seals",  "cs",  "Compressor seal system (faster wear — lower η)."),
        ]
        hdr = st.columns([3, 1, 1, 1])
        hdr[0].markdown("**Node**"); hdr[1].markdown("**β**"); hdr[2].markdown("**η (h)**"); hdr[3].markdown("**MTTR (h)**")
        st.divider()
        for label, key, tooltip in _WB_NODES:
            c0, c1, c2, c3 = st.columns([3, 1, 1, 1])
            c0.markdown(f"**{label}**")
            c0.caption(tooltip)
            p[f"{key}_beta"] = c1.number_input("β",    min_value=0.5, max_value=10.0,
                value=float(p[f"{key}_beta"]), step=0.1, format="%.1f", key=f"ni_{key}_beta",
                label_visibility="collapsed")
            p[f"{key}_eta"]  = c2.number_input("η",    min_value=1000,
                value=int(p[f"{key}_eta"]),  step=1000, key=f"ni_{key}_eta",
                label_visibility="collapsed")
            p[f"{key}_mttr"] = c3.number_input("MTTR", min_value=1,
                value=int(p[f"{key}_mttr"]), step=8,    key=f"ni_{key}_mttr",
                label_visibility="collapsed")
            st.divider()

        # ── Planned Maintenance ────────────────────────────────
        st.subheader("Planned Maintenance")
        st.caption("Staggered across units: group A offset 0, group B offset interval/2.")
        c1, c2, c3 = st.columns(3)
        p["pm_interval_h"] = c1.number_input("Interval (h)",       min_value=720,  value=int(p["pm_interval_h"]), step=720,  key="ni_pm_int")
        p["pm_ez_dur"]     = c2.number_input("EZ duration (h)",    min_value=1,    value=int(p["pm_ez_dur"]),     step=8,    key="ni_pm_ez")
        p["pm_comp_dur"]   = c3.number_input("Comp duration (h)",  min_value=1,    value=int(p["pm_comp_dur"]),   step=8,    key="ni_pm_comp")
        st.session_state["ram_params"] = p

        st.divider()

        # ── Node BOM ───────────────────────────────────────────
        st.subheader("Node BOM — auxiliary exponential components")
        st.caption(
            "Each auxiliary node is a series system of components from the Equipment Library below. "
            "Edit quantities or add/remove components. The equivalent MTBF and MTTR are computed automatically."
        )
        bom = st.session_state["bom"]
        eq  = st.session_state["eq_lib"]

        _NODE_LABELS = {
            "ez_aux":       "EZ Aux — instrumentation & valves on each electrolyzer",
            "comp_aux":     "Comp Aux — instrumentation & rotating equipment on each compressor",
            "fill_line_aux":"Fill Line Aux — valves & instrumentation per fill line",
        }
        for node_key, node_label in _NODE_LABELS.items():
            with st.expander(f"📦 {node_label}", expanded=True):
                node_bom = bom[node_key]

                # Header row
                h0, h1, h2, h3 = st.columns([3, 1, 1, 1])
                h0.markdown("**Component**"); h1.markdown("**Qty**")
                h2.markdown("**Eff. MTBF (h)**"); h3.markdown("**Remove**")

                to_remove = []
                for comp, qty in list(node_bom.items()):
                    r0, r1, r2, r3 = st.columns([3, 1, 1, 1])
                    r0.markdown(f"`{comp}`")
                    node_bom[comp] = r1.number_input(
                        "qty", min_value=1, value=int(qty), step=1,
                        key=f"bom_{node_key}_{comp}", label_visibility="collapsed")
                    if comp in eq and eq[comp]["mtbf"] > 0:
                        eff_mtbf = eq[comp]["mtbf"] / node_bom[comp]
                        r2.caption(f"{eff_mtbf:,.0f}")
                    else:
                        r2.caption("—")
                    if r3.button("✕", key=f"rm_{node_key}_{comp}"):
                        to_remove.append(comp)
                for c in to_remove:
                    del node_bom[c]

                # Add component from library
                available = [k for k in eq if k not in node_bom]
                if available:
                    ca, cb_ = st.columns([3, 1])
                    add_comp = ca.selectbox("Add from library", ["— select —"] + available,
                                            key=f"add_sel_{node_key}")
                    if cb_.button("Add", key=f"add_btn_{node_key}") and add_comp != "— select —":
                        node_bom[add_comp] = 1
                        st.rerun()

                # Rollup summary
                lam, wm = 0.0, 0.0
                valid = True
                for comp, qty in node_bom.items():
                    if comp not in eq:
                        valid = False; break
                    lam += qty / eq[comp]["mtbf"]
                    wm  += (qty / eq[comp]["mtbf"]) * eq[comp]["mttr"]
                if valid and lam > 0:
                    st.info(f"**Equivalent node:** MTBF = {1/lam:,.0f} h  ·  MTTR = {wm/lam:.0f} h")
                elif not valid:
                    st.warning("Some components in BOM are not in the Equipment Library.")

        bom_changed = any(bom[n] != st.session_state["bom"][n] for n in bom)
        st.session_state["bom"] = bom

        st.divider()

        # ── Equipment Library ──────────────────────────────────
        st.subheader("Equipment Library — exponential component data")
        st.caption("Shared by all nodes. MTBF and MTTR in hours. Built-in entries cannot be deleted.")
        eq = st.session_state["eq_lib"]

        # Column headers
        h0, h1, h2, h3 = st.columns([3, 2, 2, 1])
        h0.markdown("**Component**"); h1.markdown("**MTBF (h)**"); h2.markdown("**MTTR (h)**"); h3.markdown("")
        st.divider()

        to_delete = []
        for eq_name, vals in list(eq.items()):
            c0, c1, c2, c3 = st.columns([3, 2, 2, 1])
            c0.markdown(f"`{eq_name}`")
            eq[eq_name]["mtbf"] = c1.number_input(
                "MTBF", min_value=1000, value=int(vals["mtbf"]), step=1000,
                key=f"eq_mtbf_{eq_name}", label_visibility="collapsed")
            eq[eq_name]["mttr"] = c2.number_input(
                "MTTR", min_value=1, value=int(vals["mttr"]), step=8,
                key=f"eq_mttr_{eq_name}", label_visibility="collapsed")
            is_builtin = eq_name in _EQ_LIB_DEFAULTS
            if c3.button("🗑", key=f"del_{eq_name}", disabled=is_builtin,
                         help="Cannot delete built-in components" if is_builtin else "Remove"):
                to_delete.append(eq_name)

        for d in to_delete:
            del eq[d]
        if to_delete:
            st.session_state["eq_lib"] = eq
            st.rerun()

        # Add new component
        with st.expander("➕ Add component to library"):
            nc0, nc1, nc2, nc3 = st.columns([3, 2, 2, 1])
            new_name = nc0.text_input("Name", key="new_eq_name", placeholder="e.g. flow_meter")
            new_mtbf = nc1.number_input("MTBF (h)", min_value=1000, value=100_000, step=1000, key="new_mtbf")
            new_mttr = nc2.number_input("MTTR (h)", min_value=1,    value=48,      step=8,    key="new_mttr")
            if nc3.button("Add", key="btn_add_eq"):
                if new_name and new_name not in eq:
                    eq[new_name] = {"mtbf": int(new_mtbf), "mttr": int(new_mttr)}
                    st.session_state["eq_lib"] = eq
                    st.rerun()
                elif new_name in eq:
                    st.warning("Already exists.")

        st.session_state["eq_lib"] = eq

        # Rebuild RAM dicts after any edits in this tab
        RAM_PARAMS, RAM_PM, RAM_EQ_LIB, RAM_BOM = _build_ram_dicts()

    with sub_tl:
        st.header("Standalone Availability Timeline")
        c1, c2 = st.columns(2)
        with c1:
            tl_years = st.slider("Years to simulate", 1, 20, 10, key="tl_years")
        with c2:
            tl_seed  = st.number_input("Random seed", min_value=0, value=99, step=1, key="tl_seed")
        if st.button("▶ Run timeline", type="primary", key="btn_tl"):
            with st.spinner("Running..."):
                tl_result = rel.run_reliability_timeline(
                    TOPOLOGY, years=int(tl_years), random_seed=int(tl_seed),
                    reliability_params=RAM_PARAMS, pm_config=RAM_PM,
                    eq_lib=RAM_EQ_LIB, bom=RAM_BOM,
                )
            st.session_state["tl_result"]    = tl_result
            st.session_state["tl_seed_used"] = tl_seed
        if "tl_result" in st.session_state:
            with silence_show():
                rpo.plot_reliability_timeline(
                    st.session_state["tl_result"], TOPOLOGY,
                    title_suffix=f"seed={st.session_state['tl_seed_used']}",
                )
            show_figs()
            st.caption(f"Summary: {st.session_state['tl_result']['model'].summary()}")

    with sub_fmea:
        st.header("FMEA — Failure Mode & Effects Analysis")
        if st.button("▶ Generate FMEA", type="primary", key="btn_fmea"):
            st.session_state["fmea_df"] = generate_fmea(TOPOLOGY)
        if "fmea_df" in st.session_state:
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
    sub_single, sub_cmp = st.tabs(["▶️ Single Run", "⚖️ Schedule Comparison"])

    with sub_single:
        st.header("Single Run")
        c1, c2 = st.columns(2)
        with c1:
            single_sched = st.selectbox("Staff schedule", ALL_SCHEDULES, index=7, key="ss")
        with c2:
            single_seed  = st.number_input("Random seed", min_value=0, value=17, step=1, key="sr_seed")
        if st.button("▶ Run simulation", type="primary", key="btn_single"):
            with st.spinner("Running..."):
                ct = make_container_types(frac_a, frac_b, frac_c)
                ap = make_arrival_pattern(pattern_type, peak_hour, peak_hour_2, peak_width, peak_weight)
                plant   = make_plant()
                results = po.run_simulation(
                    container_types=ct, plant=plant, avg_arrivals_per_day=avg_arrivals,
                    days=int(sim_days), step_minutes=1, random_seed=int(single_seed),
                    schedule=make_schedule(single_sched), arrival_pattern=ap,
                    reliability_model=make_rel_model(),
                )
                kpis = rpo.compute_kpis(results, plant, ct,
                                         avg_arrivals_per_day=avg_arrivals, days=int(sim_days))
                econ = eco.run_economics(results, kpis, schedule_label=single_sched, days=int(sim_days))
            st.session_state.update({"sr_res": results, "sr_kpis": kpis, "sr_econ": econ})
        if "sr_kpis" in st.session_state:
            import numpy as np
            kpis    = st.session_state["sr_kpis"]
            econ    = st.session_state["sr_econ"]
            results = st.session_state["sr_res"]
            st.subheader("Operations KPIs")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Containers filled", kpis["containers_filled"])
            c2.metric("Dispensed (kg)", f"{kpis['total_dispensed_kg']:.0f}")
            c3.metric("Avg ext wait (min)", f"{kpis['ext_avg']:.0f}")
            c4.metric("Plant utilization", f"{kpis['plant_utilization']*100:.1f}%")
            c5,c6,c7,c8 = st.columns(4)
            c5.metric("Avg docked wait (min)", f"{kpis['doc_avg']:.0f}")
            c6.metric("Max external queue", f"{kpis['max_external_queue']:.0f}")
            c7.metric("Avg fill time (min)", f"{kpis['fill_avg']:.0f}")
            c8.metric("Avg fill lines in use", f"{kpis['avg_filling']:.2f}")
            if RELIABILITY_ON and results.get("reliability_summary"):
                st.subheader("Availability (this run)")
                r1,r2,r3 = st.columns(3)
                r1.metric("Avg EZ availability", f"{float(np.mean(results['ez_availability_log']))*100:.2f}%")
                r2.metric("Avg fill lines up", f"{float(np.mean(results['fill_lines_up_log'])):.2f}/{TOPOLOGY.total_fill_lines()}")
                r3.metric("Time in PM", f"{float(np.mean(results['pm_active_log']))*100:.2f}%")
            st.subheader("Economics")
            e1,e2,e3,e4 = st.columns(4)
            e1.metric("Revenue",    f"{econ['revenue_kr_annual']:,.0f} kr/yr")
            e2.metric("Staff cost", f"{econ['staff_cost_kr_annual']:,.0f} kr/yr")
            e3.metric("Queue cost", f"{econ['queue_cost_kr_annual']:,.0f} kr/yr")
            e4.metric("Net result", f"{econ['net_kr_annual']:,.0f} kr/yr")
            st.subheader("Charts")
            with silence_show():
                rpo.plot_results(results)
            show_figs()

    with sub_cmp:
        st.header("Schedule Comparison")
        cmp_scheds = st.multiselect("Schedules", ALL_SCHEDULES,
                                     default=["unmanned","8-16_closed","8-20_closed","8-24_open","24_7"])
        cmp_seed = st.number_input("Random seed", min_value=0, value=17, step=1, key="cmp_seed")
        if st.button("▶ Run comparison", type="primary", key="btn_cmp") and cmp_scheds:
            import pandas as pd
            prog = st.progress(0.0)
            ct   = make_container_types(frac_a, frac_b, frac_c)
            ap   = make_arrival_pattern(pattern_type, peak_hour, peak_hour_2, peak_width, peak_weight)
            rows = []
            for i, lbl in enumerate(cmp_scheds):
                plant   = make_plant()
                results = po.run_simulation(
                    container_types=ct, plant=plant, avg_arrivals_per_day=avg_arrivals,
                    days=int(sim_days), step_minutes=1, random_seed=int(cmp_seed),
                    schedule=make_schedule(lbl), arrival_pattern=ap,
                    reliability_model=make_rel_model(seed=int(cmp_seed)),
                )
                kpis = rpo.compute_kpis(results, plant, ct,
                                         avg_arrivals_per_day=avg_arrivals, days=int(sim_days))
                rows.append(eco.run_economics(results, kpis, schedule_label=lbl, days=int(sim_days)))
                prog.progress((i+1)/len(cmp_scheds))
            prog.empty()
            st.session_state["cmp_df"] = pd.DataFrame(rows)
        if "cmp_df" in st.session_state:
            import pandas as pd
            df = st.session_state["cmp_df"]
            show_cols = ["schedule_label","dispensed_kg_annual","revenue_kr_annual",
                          "staff_cost_kr_annual","queue_cost_kr_annual","net_kr_annual"]
            st.dataframe(df[show_cols].round(0), hide_index=True, use_container_width=True)
            fig, _ = ep.plot_stacked_bar(df)
            st.pyplot(fig)


# ────────────────────────────────────────────────────────────
# TAB 3 — Economics
# ────────────────────────────────────────────────────────────
with tab_econ:
    sub_sweep, sub_charts = st.tabs(["🎲 Monte Carlo Sweep", "📊 Charts"])

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
            st.info(f"**{n_scen} scenarios x {int(n_runs)} seeds = {n_scen*int(n_runs)} runs**")
        if st.button("▶ Run sweep", type="primary", key="btn_sweep") and arrival_rates and sw_scheds:
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
                    db_path              = db_path,
                    verbose              = False,
                )
            st.success(f"Done in {time.time()-t0:.1f}s — {safe_count(db_path):,} runs in DB.")

    with sub_charts:
        st.header("Economics Charts")
        try:
            import pandas as pd
            raw = mc.query(db_path)
        except Exception:
            import pandas as pd
            raw = pd.DataFrame()
        if raw.empty:
            st.info("No sweep data yet — run a Monte Carlo sweep first.")
        else:
            econ_df = eco.add_economics_columns(raw)
            f1,f2,f3,f4 = st.columns(4)
            with f1:
                mix_f = st.selectbox("Mix", sorted(econ_df["mix_label"].unique()), key="ef_mix")
            with f2:
                pcol = "plant_label" if "plant_label" in econ_df.columns else "topology_mode"
                pf   = st.selectbox("Plant", sorted(econ_df[pcol].unique()), key="ef_plant")
            with f3:
                hr = st.selectbox("Headline rate", sorted(econ_df["arrival_rate"].unique()), key="ef_rate")
            with f4:
                rel_f = None
                if "reliability_label" in econ_df.columns:
                    rel_f = st.selectbox("Reliability",
                                          sorted(econ_df["reliability_label"].dropna().unique()),
                                          key="ef_rel")
            filtered = econ_df[(econ_df["mix_label"]==mix_f)&(econ_df[pcol]==pf)]
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
    try:
        import pandas as pd
        full_df = mc.query(db_path)
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
                st.dataframe(mc.query(db_path, sql), hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"Query failed: {e}")
        st.subheader("Export")
        c1, c2 = st.columns(2)
        with c1:
            grp = [c for c in ["arrival_rate","schedule_label","mix_label",
                                "topology_mode","n_fill_lines","reliability_label"]
                   if c in full_df.columns]
            st.download_button("⬇ KPI summary (CSV)",
                                data=mc.summary_by(db_path, grp).to_csv(index=False).encode("utf-8"),
                                file_name="mc_summary.csv", mime="text/csv")
        with c2:
            edf  = eco.add_economics_columns(full_df)
            cols = [c for c in ["arrival_rate","schedule_label","reliability_label",
                                  "total_dispensed_kg","revenue_kr_annual",
                                  "staff_cost_kr_annual","queue_cost_kr_annual","net_kr_annual"]
                    if c in edf.columns]
            st.download_button("⬇ Economics (CSV)",
                                data=edf[cols].to_csv(index=False).encode("utf-8"),
                                file_name="economics_summary.csv", mime="text/csv")
        st.subheader("Raw data (first 50 rows)")
        st.dataframe(full_df.head(50), hide_index=True, use_container_width=True)
