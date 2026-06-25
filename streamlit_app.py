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
}
_PM_DEFAULTS = {
    "ez":  {"enabled": True, "interval_h": 8760, "duration_h": 72,  "resets_age": True},
    "stk": {"enabled": True},
    "cb":  {"enabled": True, "interval_h": 8760, "duration_h": 96,  "resets_age": True},
    "cm":  {"enabled": True, "interval_h": 8760, "duration_h": 96,  "resets_age": True},
    "cs":  {"enabled": True, "interval_h": 8760, "duration_h": 96,  "resets_age": True},
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
        "pm_config":     {k: dict(v) for k, v in _PM_DEFAULTS.items()},
        "pm_offsets":    {"ez": [], "comp": []},
        "staff_costs":   dict(_STAFF_DEFAULTS),
        "margin_kr":     30.0,
        "queue_cost_kr": 1500.0,
        "db_path":       "hydrogen_mc.duckdb",
        "tl_result":     None,
        "tl_seed_used":  None,
        "single_result": None,
        "fmea_df":       None,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Migrate old flat PM keys to per-node format
    rp = st.session_state.get("ram_params", {})
    if "pm_interval_h" in rp:
        interval = rp["pm_interval_h"]
        ez_dur   = rp.get("pm_ez_dur", 72)
        comp_dur = rp.get("pm_comp_dur", 96)
        st.session_state["pm_config"] = {
            "ez":  {"enabled": True, "interval_h": interval, "duration_h": ez_dur,   "resets_age": True},
            "stk": {"enabled": True},
            "cb":  {"enabled": True, "interval_h": interval, "duration_h": comp_dur, "resets_age": True},
            "cm":  {"enabled": True, "interval_h": interval, "duration_h": comp_dur, "resets_age": True},
            "cs":  {"enabled": True, "interval_h": interval, "duration_h": comp_dur, "resets_age": True},
        }
        for k in ["pm_interval_h", "pm_ez_dur", "pm_comp_dur"]:
            rp.pop(k, None)

    # Ensure "stk" key exists in pm_config (added after initial release)
    if "pm_config" in st.session_state and "stk" not in st.session_state["pm_config"]:
        st.session_state["pm_config"]["stk"] = {"enabled": True}

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
    pm_ss = st.session_state["pm_config"]
    stk_pm = {**pm_ss["ez"], "enabled": pm_ss.get("stk", {}).get("enabled", True)}
    pm = {
        "electrolyzer_body": dict(pm_ss["ez"]),
        "stack":             stk_pm,
        "compressor_block":  dict(pm_ss["cb"]),
        "compressor_motor":  dict(pm_ss["cm"]),
        "compressor_seals":  dict(pm_ss["cs"]),
    }
    offsets = st.session_state.get("pm_offsets", {"ez": [], "comp": []})
    pm_offsets = offsets if (offsets.get("ez") or offsets.get("comp")) else None
    return params, pm, dict(st.session_state["eq_lib"]), dict(st.session_state["bom"]), pm_offsets


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


def make_arrival_pattern(ptype, ph, ph2, pw, pw2, pwt):
    if ptype == "uniform":
        return po.ArrivalPattern(pattern_type="uniform")
    if ptype == "single_peak":
        return po.ArrivalPattern(pattern_type="single_peak", peak_hour=ph, peak_width_hours=pw)
    return po.ArrivalPattern(pattern_type="double_peak",
                              peak_hour=ph, peak_hour_2=ph2,
                              peak_width_hours=pw, peak_width_hours_2=pw2,
                              peak_weight=pwt)


def make_rel_model(topology, seed, reliability_on):
    if not reliability_on:
        return None
    params, pm, eq_lib, bom, pm_offsets = _build_ram_dicts()
    return rel.ReliabilityModel(
        topology, random_seed=int(seed),
        reliability_params=params, pm_config=pm,
        eq_lib=eq_lib, bom=bom, pm_offsets=pm_offsets,
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
#1

# ── Topology ─────────────────────────────────────────────────
with st.sidebar.expander("🏗️ Plant & Topology", expanded=True):
    topology_mode = st.selectbox("Topology lauout mode", ["Common Header at low and high pressure", "Common Header at low pressure, dedicated fill lines per compressor", "Seperate trains"])
    if topology_mode == "Common Header at low and high pressure":
        n_ez            = st.slider("Electrolyzers", 1, 8, 3)
        stacks          = st.slider("Stacks per electrolyzer", 1, 4, 2)
        ez_kg_hr_each   = st.number_input("Capacity per electrolyzer (kg/hr)", 1.0, value=44.0, step=1.0)
        n_comp          = st.slider("Compressors", 1, 6, 2)
        comp_kg_hr_each = st.number_input("Flow per compressor (kg/hr)", 1.0, value=round(ez_kg_hr_each * n_ez / n_comp, 1), step=1.0)
        n_fill          = st.slider("Shared fill lines", 1, 12, 4)
        TOPOLOGY = pt.PlantTopology(
            mode="common", n_electrolyzers=n_ez, stacks_per_electrolyzer=stacks,
            electrolyzer_kg_per_hr_each=ez_kg_hr_each,
            n_compressors=n_comp,
            compressor_flow_kg_per_hr_each=comp_kg_hr_each,
            n_fill_lines=n_fill,
        )
    elif topology_mode == "Common Header at low pressure, dedicated fill lines per compressor":
        n_ez            = st.slider("Electrolyzers", 1, 8, 3)
        stacks          = st.slider("Stacks per electrolyzer", 1, 4, 2)
        ez_kg_hr_each   = st.number_input("Capacity per electrolyzer (kg/hr)", 1.0, value=44.0, step=1.0)
        n_comp          = st.slider("Compressors", 1, 6, 2)
        comp_kg_hr_each = st.number_input("Flow per compressor (kg/hr)", 1.0, value=round(ez_kg_hr_each * n_ez / n_comp, 1), step=1.0)
        lines_per_comp  = st.slider("Fill lines per compressor", 1, 6, 2)
        TOPOLOGY = pt.PlantTopology(
            mode="pooled_ez_dedicated_comp",
            n_electrolyzers=n_ez, stacks_per_electrolyzer=stacks,
            electrolyzer_kg_per_hr_each=ez_kg_hr_each,
            n_compressors=n_comp,
            compressor_flow_kg_per_hr_each=comp_kg_hr_each,
            n_fill_lines_per_compressor=lines_per_comp,
        )
    else:
        n_trains              = st.slider("Number of trains", 2, 4, 2)
        ez_per_train          = st.slider("Electrolyzers per train", 1, 4, 2)
        stacks                = st.slider("Stacks per electrolyzer", 1, 4, 2)
        ez_kg_hr_each_train   = st.number_input("Capacity per electrolyzer (kg/hr)", 1.0, value=22.0, step=1.0)
        comp_per_train        = st.slider("Compressors per train", 1, 3, 1)
        comp_kg_hr_each_train = st.number_input("Flow per compressor (kg/hr)", 1.0, value=round(ez_kg_hr_each_train * ez_per_train / comp_per_train, 1), step=1.0)
        lines_per_train       = st.slider("Fill lines per train", 1, 6, 2)
        TOPOLOGY = pt.PlantTopology(
            mode="trains",
            trains=[
                pt.Train(
                    label=f"Train {i+1}", n_electrolyzers=ez_per_train,
                    electrolyzer_kg_per_hr_each=ez_kg_hr_each_train,
                    n_compressors=comp_per_train,
                    compressor_flow_kg_per_hr_each=comp_kg_hr_each_train,
                    n_fill_lines=lines_per_train, stacks_per_electrolyzer=stacks,
                )
                for i in range(n_trains)
            ],
        )

    ez_cap   = TOPOLOGY.theoretical_capacity_kg_per_hr()
    comp_cap = TOPOLOGY.theoretical_compressor_capacity_kg_per_hr()
    theo_hr  = min(ez_cap, comp_cap)
    theo_day = theo_hr * 24
    bneck    = "compressor" if TOPOLOGY.compressor_is_bottleneck() else "electrolyzer"

    # Overcapacity / balance info
    if ez_cap > comp_cap:
        overcap = f"⚠️ Compressor undersized — EZ can produce {ez_cap:.1f} kg/hr but comp handles {comp_cap:.1f} kg/hr"
    elif comp_cap > ez_cap * 1.05:
        overcap = f"ℹ️ Compressor has spare capacity — {comp_cap:.1f} kg/hr vs {ez_cap:.1f} kg/hr EZ output"
    else:
        overcap = f"✅ Balanced — EZ {ez_cap:.1f} kg/hr · Comp {comp_cap:.1f} kg/hr"

    st.caption(
        f"**{TOPOLOGY.total_electrolyzers()} EZ · {TOPOLOGY.total_compressors()} comp · "
        f"{TOPOLOGY.total_fill_lines()} fill lines**  \n"
        f"Bottleneck: **{bneck}** · {theo_hr:.1f} kg/hr · {theo_day:.0f} kg/day  \n"
        f"{overcap}"
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
    container_seed = st.number_input("Container seed", min_value=0, value=7, step=1, key="container_seed")

# ── Arrivals ─────────────────────────────────────────────────
with st.sidebar.expander("📦 Arrivals", expanded=False):
    avg_arrivals = st.number_input("Avg containers / day", min_value=0.01, value=3.0,
                                    step=0.5, format="%.1f")
    pattern_type = st.selectbox("Timing pattern", ["uniform", "single_peak", "double_peak"])
    peak_hour = peak_hour_2 = None
    peak_width = 3.0
    peak_width_2 = 3.0
    peak_weight = 0.5
    if pattern_type == "single_peak":
        peak_hour  = st.slider("Peak hour", 0.0, 24.0, 8.0, 0.5)
        peak_width = st.slider("Peak width (hours)", 0.5, 8.0, 3.0, 0.5)
    elif pattern_type == "double_peak":
        peak_hour   = st.slider("First peak hour",  0.0, 24.0,  8.0, 0.5)
        peak_width  = st.slider("First peak width (hours)", 0.5, 8.0, 2.5, 0.5)
        peak_hour_2 = st.slider("Second peak hour", 0.0, 24.0, 16.0, 0.5)
        peak_width_2 = st.slider("Second peak width (hours)", 0.5, 8.0, 2.5, 0.5)
        peak_weight = st.slider("Weight on first peak", 0.1, 0.9, 0.5, 0.05)
    sim_days = st.number_input("Simulated days", min_value=1, value=31, step=1)
    arrival_seed = st.number_input("Arrival seed", min_value=0, value=42, step=1, key="arrival_seed")

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
    f"{theo_day:.0f} kg/day · RAM {'🟢 ON' if RELIABILITY_ON else '⚪ OFF'} · "
    f"{avg_arrivals}/day ({pattern_type}) · {int(sim_days)} days"
)

tab_plant, tab_ops, tab_econ, tab_data = st.tabs([
    "🏗️ Plant & RAM", "🚛 Operations", "💰 Economics", "🗄️ Data",
])


# ────────────────────────────────────────────────────────────
# TAB 1 — Plant & RAM
# ────────────────────────────────────────────────────────────
with tab_plant:
    sub_arch, sub_nodes, sub_pm_sched, sub_tl, sub_fmea = st.tabs([
        "🏗️ Architecture", "🔩 Nodes", "🗓️ PM Scheduling",
        "📉 Reliability Data & Timeline", "📋 FMEA",
    ])

    # ── Architecture ──────────────────────────────────────────
    with sub_arch:
        st.header("Reliability Architecture")
        st.plotly_chart(
            rpo.draw_plant_architecture_plotly(TOPOLOGY, ram_params=st.session_state["ram_params"]),
            use_container_width=True,
        )
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

        p = dict(st.session_state["ram_params"])
        pm_cfg = {k: dict(v) for k, v in st.session_state["pm_config"].items()}

        # ── Weibull nodes + per-node PM ────────────────────────
        st.subheader("Weibull nodes")
        st.caption(
            "β = shape (>1 → wear-out),  η = characteristic life (h),  "
            "MTTR = corrective repair (h).  PM columns control planned maintenance per node type."
        )

        _WB_NODES = [
            ("Electrolyzer body", "ez",  "Housing, membrane assembly. Series with stacks and aux.", "ez"),
            ("Stack",             "stk", "One stack per EZ. 1-of-N needed → proportional derate.",  "stk"),
            ("Compressor block",  "cb",  "Compressor main block. All of block + motor + seals must be up.", "cb"),
            ("Compressor motor",  "cm",  "Compressor drive motor.", "cm"),
            ("Compressor seals",  "cs",  "Seal system — faster wear, lower η.", "cs"),
        ]

        hdr = st.columns([2.5, 0.8, 0.8, 0.8, 0.7, 1.0, 0.8, 0.7])
        hdr[0].markdown("**Node**")
        hdr[1].markdown("**β**")
        hdr[2].markdown("**η (h)**")
        hdr[3].markdown("**MTTR (h)**")
        hdr[4].markdown("**PM**")
        hdr[5].markdown("**Interval (h)**")
        hdr[6].markdown("**Duration (h)**")
        hdr[7].markdown("**Reset age**")
        st.divider()

        for label, key, tooltip, pm_key in _WB_NODES:
            cols = st.columns([2.5, 0.8, 0.8, 0.8, 0.7, 1.0, 0.8, 0.7])
            cols[0].markdown(f"**{label}**")
            cols[0].caption(tooltip)
            p[f"{key}_beta"] = cols[1].number_input("β", min_value=0.5, max_value=10.0,
                value=float(p[f"{key}_beta"]), step=0.1, format="%.1f",
                key=f"ni_{key}_beta", label_visibility="collapsed")
            p[f"{key}_eta"]  = cols[2].number_input("η", min_value=1000,
                value=int(p[f"{key}_eta"]),  step=1000,
                key=f"ni_{key}_eta", label_visibility="collapsed")
            p[f"{key}_mttr"] = cols[3].number_input("MTTR", min_value=1,
                value=int(p[f"{key}_mttr"]), step=8,
                key=f"ni_{key}_mttr", label_visibility="collapsed")

            if pm_key == "stk":
                node_pm = pm_cfg[pm_key]
                node_pm["enabled"] = cols[4].toggle("On", value=node_pm["enabled"],
                                                     key=f"pm_en_{pm_key}")
                cols[5].caption("Inherits")
                cols[6].caption("EZ body")
                cols[7].caption("settings")
            elif pm_key is not None:
                node_pm = pm_cfg[pm_key]
                node_pm["enabled"] = cols[4].toggle("On", value=node_pm["enabled"],
                                                     key=f"pm_en_{pm_key}")
                if node_pm["enabled"]:
                    node_pm["interval_h"] = cols[5].number_input("Int", min_value=168,
                        value=int(node_pm["interval_h"]), step=730,
                        key=f"pm_int_{pm_key}", label_visibility="collapsed")
                    node_pm["duration_h"] = cols[6].number_input("Dur", min_value=1,
                        value=int(node_pm["duration_h"]), step=8,
                        key=f"pm_dur_{pm_key}", label_visibility="collapsed")
                    node_pm["resets_age"] = cols[7].toggle("Reset", value=node_pm["resets_age"],
                                                            key=f"pm_reset_{pm_key}")
                else:
                    cols[5].caption("—")
                    cols[6].caption("—")
                    cols[7].caption("—")
            st.divider()

        st.session_state["ram_params"] = p
        st.session_state["pm_config"] = pm_cfg

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

    # ── PM Scheduling ─────────────────────────────────────────
    with sub_pm_sched:
        st.header("PM Scheduling")
        import numpy as np
        import plotly.graph_objects as go

        n_ez   = TOPOLOGY.total_electrolyzers()
        n_comp = TOPOLOGY.total_compressors()
        pm_cfg = st.session_state["pm_config"]
        ez_pm  = pm_cfg["ez"]
        cb_pm  = pm_cfg["cb"]
        cm_pm  = pm_cfg["cm"]
        cs_pm  = pm_cfg["cs"]

        # Collect all enabled compressor sub-component PM configs
        _comp_pms = []
        if cb_pm["enabled"]: _comp_pms.append(("block", cb_pm))
        if cm_pm["enabled"]: _comp_pms.append(("motor", cm_pm))
        if cs_pm["enabled"]: _comp_pms.append(("seals", cs_pm))
        # Unique intervals across enabled sub-components
        _comp_intervals = sorted({p["interval_h"] for _, p in _comp_pms})

        st.caption(
            "Electrolyzers and compressors are taken down **sequentially** — "
            "one at a time, so capacity is reduced by 1/N during each PM window. "
            "Compressor sub-components with different intervals get separate PM windows."
        )

        c1, c2 = st.columns(2)
        gap_days = c1.number_input("Gap between sequential PMs (days)",
                                    min_value=0, value=2, step=1, key="pm_gap_days")
        gap_h = gap_days * 24

        pair_comp = c2.toggle("Pair compressor PM with electrolyzer PM",
                               value=True, key="pm_pair_comp")

        # Compute sequential offsets
        ez_dur  = ez_pm["duration_h"]
        ez_int  = ez_pm["interval_h"]
        ez_step = ez_dur + gap_h

        offsets = {"ez": [], "comp": []}
        for i in range(n_ez):
            offsets["ez"].append(float(ez_int + i * ez_step))

        if pair_comp:
            for j in range(n_comp):
                offsets["comp"].append(offsets["ez"][j % n_ez])
        else:
            longest_dur = max((p["duration_h"] for _, p in _comp_pms), default=96)
            comp_start  = ez_int + n_ez * ez_step + gap_h
            comp_step   = longest_dur + gap_h
            for j in range(n_comp):
                offsets["comp"].append(float(comp_start + j * comp_step))

        st.session_state["pm_offsets"] = offsets

        # Summary
        total_seq = n_ez * ez_step - gap_h
        comp_info = ", ".join(f"{name} every {p['interval_h']:,} h" for name, p in _comp_pms)
        st.info(
            f"**EZ sequence:** {n_ez} × {ez_dur:.0f} h PM + {gap_h:.0f} h gap = "
            f"**{total_seq:.0f} h** total ({total_seq/24:.0f} days).  \n"
            f"Capacity during each EZ PM: **{(1 - 1/max(n_ez,1))*100:.0f}%**  \n"
            f"**Compressor PM:** {comp_info or 'all disabled'}"
            + (f"  — paired with EZ" if pair_comp else f"  — separate sequence")
        )

        # ── PM Timeline Preview ───────────────────────────────
        st.subheader("PM Timeline Preview")
        preview_years = st.slider("Preview years", 1, 5, 2, key="pm_preview_years")
        total_h = preview_years * 8760

        fig_pm = go.Figure()
        unit_labels = []
        y_idx = 0

        def _add_windows(interval_h, duration_h, offset_h, y_pos, color, opacity=0.6):
            t = offset_h
            while t < total_h:
                fig_pm.add_shape(type="rect",
                    x0=t, x1=min(t + duration_h, total_h),
                    y0=y_pos - 0.35, y1=y_pos + 0.35,
                    fillcolor=color, opacity=opacity, line_width=0)
                t += interval_h

        for i in range(n_ez):
            y_idx += 1
            unit_labels.append(f"EZ {i+1}")
            if ez_pm["enabled"]:
                _add_windows(ez_int, ez_dur, offsets["ez"][i], y_idx, "#2196F3")

        _COMP_COLORS = {"block": "#9C27B0", "motor": "#E91E63", "seals": "#FF5722"}
        for j in range(n_comp):
            y_idx += 1
            unit_labels.append(f"Comp {j+1}")
            off = offsets["comp"][j]
            for name, cpm in _comp_pms:
                _add_windows(cpm["interval_h"], cpm["duration_h"], off, y_idx,
                            _COMP_COLORS.get(name, "#9C27B0"))

        # Capacity overlay — any sub-component in PM takes the compressor offline
        cap_ez   = np.ones(total_h)
        cap_comp = np.ones(total_h)
        for i in range(n_ez):
            if ez_pm["enabled"]:
                t = offsets["ez"][i]
                while t < total_h:
                    s, e = int(t), min(int(t + ez_dur), total_h)
                    cap_ez[s:e] -= 1.0 / max(n_ez, 1)
                    t += ez_int

        comp_down = np.zeros(total_h, dtype=bool)
        for j in range(n_comp):
            unit_down = np.zeros(total_h, dtype=bool)
            off = offsets["comp"][j]
            for _, cpm in _comp_pms:
                t = off
                while t < total_h:
                    s, e = int(t), min(int(t + cpm["duration_h"]), total_h)
                    unit_down[s:e] = True
                    t += cpm["interval_h"]
            cap_comp[unit_down] -= 1.0 / max(n_comp, 1)

        cap_plant = np.clip(np.minimum(cap_ez, cap_comp), 0, 1)

        ds = max(1, total_h // 2000)
        x_ds = np.arange(0, total_h, ds)
        fig_pm.add_trace(go.Scatter(
            x=x_ds, y=cap_ez[::ds] * 100, mode="lines",
            line=dict(color="rgba(33,150,243,0.5)", width=1, dash="dot"),
            name="EZ capacity (%)", yaxis="y2",
        ))
        fig_pm.add_trace(go.Scatter(
            x=x_ds, y=cap_plant[::ds] * 100, mode="lines",
            line=dict(color="rgba(255,152,0,0.9)", width=2),
            name="Plant capacity (%)", yaxis="y2",
            hovertemplate="Hour %{x:,}<br>Plant capacity: %{y:.0f}%<extra></extra>",
        ))

        # Legend for comp sub-components
        for name, color in _COMP_COLORS.items():
            if any(n == name for n, _ in _comp_pms):
                fig_pm.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=color), name=f"Comp {name}",
                    yaxis="y2", showlegend=True,
                ))

        for y in range(1, preview_years + 1):
            fig_pm.add_vline(x=y * 8760, line_color="#aaa", line_width=0.5, line_dash="dash",
                            annotation_text=f"Year {y}", annotation_position="top")

        fig_pm.update_layout(
            height=max(350, y_idx * 50 + 140),
            xaxis=dict(title="Hours", range=[0, total_h]),
            yaxis=dict(tickvals=list(range(1, y_idx + 1)), ticktext=unit_labels,
                      range=[0.3, y_idx + 0.7]),
            yaxis2=dict(title="Capacity (%)", overlaying="y", side="right",
                       range=[0, 110]),
            plot_bgcolor="#F8F7F4", paper_bgcolor="white",
            margin=dict(l=80, r=60, t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x",
        )
        st.plotly_chart(fig_pm, use_container_width=True)

    # ── Reliability Timeline ───────────────────────────────────
    with sub_tl:
        st.header("Standalone Availability Timeline")
        c1, c2, c3 = st.columns(3)
        tl_years  = c1.slider("Years to simulate", 1, 20, 10, key="tl_years")
        tl_seed   = c2.number_input("Random seed", min_value=0, value=99, step=1, key="tl_seed")
        roll_days = c3.slider("Rolling avg (days)", 1, 90, 30, key="tl_roll")

        if st.button("▶ Run timeline", type="primary", key="btn_tl"):
            # Capture all config at button-press time — not at render time
            _params, _pm, _eq_lib, _bom, _pm_offsets = _build_ram_dicts()
            with st.spinner("Running..."):
                tl_result = rel.run_reliability_timeline(
                    TOPOLOGY, years=int(tl_years), random_seed=int(tl_seed),
                    reliability_params=_params, pm_config=_pm,
                    eq_lib=_eq_lib, bom=_bom, pm_offsets=_pm_offsets,
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

            tl_hourly, tl_monthly, tl_yearly, tl_exceed, tl_bands, tl_pmvsfail, tl_compfail = st.tabs([
                "📈 Hourly", "📅 Monthly avg", "📊 Yearly bars", "📉 Exceedance",
                "🟩 Capacity Bands", "🔧 PM vs Failures", "⚙️ Component Failures",
            ])

            with tl_hourly:
                roll_w = roll_days * 24
                roll = np.convolve(cap, np.ones(roll_w)/roll_w, mode="same") if len(cap) >= roll_w else cap

                # Separate PM vs failure downtime per hour
                down = cap < 1.0
                pm_down   = down & pm_hist
                fail_down = down & ~pm_hist

                _n_ez_tl   = TOPOLOGY.total_electrolyzers()
                _n_comp_tl = TOPOLOGY.total_compressors()
                n_pm_rows  = _n_ez_tl + _n_comp_tl

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.55, 0.15, 0.30],
                    vertical_spacing=0.04,
                    subplot_titles=("Plant Capacity", "PM Schedule (per unit)", "Downtime Classification"),
                )

                # ── Row 1: Capacity timeline ──────────────────
                fig.add_trace(go.Scatter(
                    x=years_ax, y=cap * 100, mode="lines",
                    line=dict(color="rgba(30,30,30,0.15)", width=0.5),
                    name="Hourly capacity", showlegend=False,
                    hovertemplate="Year %{x:.2f}<br>Capacity: %{y:.1f}%<extra></extra>",
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=years_ax, y=roll * 100, mode="lines",
                    line=dict(color="#1a1a2e", width=2), name=f"{roll_days}d rolling mean",
                    hovertemplate="Year %{x:.2f}<br>Rolling avg: %{y:.1f}%<extra></extra>",
                ), row=1, col=1)
                fig.add_hline(y=overall, line_dash="dot", line_color="#999",
                              annotation_text=f"Mean {overall:.1f}%",
                              annotation_position="top right", row=1, col=1)

                # ── Row 2: PM schedule strips ─────────────────
                _pm_offsets_tl = st.session_state.get("pm_offsets", {"ez": [], "comp": []})
                _pm_cfg_tl = st.session_state["pm_config"]
                _strip_h = 0.8 / max(n_pm_rows, 1)

                for i in range(_n_ez_tl):
                    if _pm_cfg_tl["ez"]["enabled"]:
                        off = _pm_offsets_tl["ez"][i] if i < len(_pm_offsets_tl.get("ez", [])) else 0
                        interval = _pm_cfg_tl["ez"]["interval_h"]
                        dur = _pm_cfg_tl["ez"]["duration_h"]
                        t = off if off > 0 else interval
                        y_pos = 1.0 - i * _strip_h - _strip_h / 2
                        while t < len(cap):
                            fig.add_shape(type="rect",
                                x0=t/8760, x1=min(t+dur, len(cap))/8760,
                                y0=y_pos - _strip_h*0.4, y1=y_pos + _strip_h*0.4,
                                fillcolor="#2196F3", opacity=0.7, line_width=0,
                                xref="x2", yref="y2")
                            t += interval

                for j in range(_n_comp_tl):
                    off = _pm_offsets_tl["comp"][j] if j < len(_pm_offsets_tl.get("comp", [])) else 0
                    y_pos = 1.0 - (_n_ez_tl + j) * _strip_h - _strip_h / 2
                    for ckey in ["cb", "cm", "cs"]:
                        cpm = _pm_cfg_tl.get(ckey, {})
                        if cpm.get("enabled", False):
                            interval = cpm["interval_h"]
                            dur = cpm["duration_h"]
                            t = off if off > 0 else interval
                            _ccol = {"cb": "#9C27B0", "cm": "#E91E63", "cs": "#FF5722"}[ckey]
                            while t < len(cap):
                                fig.add_shape(type="rect",
                                    x0=t/8760, x1=min(t+dur, len(cap))/8760,
                                    y0=y_pos - _strip_h*0.4, y1=y_pos + _strip_h*0.4,
                                    fillcolor=_ccol, opacity=0.7, line_width=0,
                                    xref="x2", yref="y2")
                                t += interval

                pm_labels = [f"EZ{i+1}" for i in range(_n_ez_tl)] + [f"C{j+1}" for j in range(_n_comp_tl)]
                pm_ticks  = [1.0 - k * _strip_h - _strip_h/2 for k in range(n_pm_rows)]
                fig.update_yaxes(
                    tickvals=pm_ticks, ticktext=pm_labels, range=[0, 1.1],
                    tickfont=dict(size=9), row=2, col=1,
                )

                # ── Row 3: Capacity loss (PM vs failure) ──────
                loss = (1.0 - cap) * 100
                pm_loss   = loss * pm_hist.astype(float)
                fail_loss = loss * (~pm_hist).astype(float)

                ds = max(1, len(cap) // 4000)
                x_ds = years_ax[::ds]
                pm_ds   = np.array([pm_loss[i*ds:min((i+1)*ds, len(cap))].mean()
                                    for i in range(len(x_ds))])
                fail_ds = np.array([fail_loss[i*ds:min((i+1)*ds, len(cap))].mean()
                                    for i in range(len(x_ds))])

                fig.add_trace(go.Scatter(
                    x=x_ds, y=pm_ds, mode="lines", fill="tozeroy",
                    line=dict(color="#2196F3", width=0), fillcolor="rgba(33,150,243,0.5)",
                    name="PM capacity loss",
                    hovertemplate="Year %{x:.2f}<br>PM loss: %{y:.1f}%<extra></extra>",
                ), row=3, col=1)
                fig.add_trace(go.Scatter(
                    x=x_ds, y=pm_ds + fail_ds, mode="lines", fill="tonexty",
                    line=dict(color="#E24B4A", width=0), fillcolor="rgba(226,75,74,0.5)",
                    name="Failure capacity loss",
                    hovertemplate="Year %{x:.2f}<br>Total loss: %{y:.1f}%<extra></extra>",
                ), row=3, col=1)

                for y in range(1, n_years + 1):
                    fig.add_vline(x=y, line_color="#ccc", line_width=0.5, line_dash="dash")

                fig.update_layout(
                    height=650,
                    xaxis3=dict(title="Year", tickmode="linear", dtick=1,
                                rangeslider=dict(visible=True, thickness=0.04)),
                    yaxis=dict(title="Capacity (%)", range=[0, 108]),
                    yaxis3=dict(title="Capacity loss (%)", range=[0, max(15, (pm_ds+fail_ds).max()*1.3)]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    margin=dict(l=55, r=20, t=50, b=60),
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

            with tl_bands:
                st.caption(
                    "Total hours the plant spent at each capacity level. "
                    "A healthy plant concentrates hours in the 100% band."
                )
                BAND_DEFS = [
                    (1.00, 1.01, "100%",   "#1D9E75"),
                    (0.76, 1.00, "76-99%", "#639922"),
                    (0.51, 0.75, "51-75%", "#EF9F27"),
                    (0.26, 0.50, "26-50%", "#D85A30"),
                    (0.01, 0.25, "1-25%",  "#E24B4A"),
                    (0.00, 0.01, "0%",     "#8B0000"),
                ]
                band_labels, band_hours, band_colors = [], [], []
                for lo, hi, lbl, col in BAND_DEFS:
                    h = int(((cap >= lo) & (cap < hi + 0.001)).sum())
                    band_labels.append(lbl)
                    band_hours.append(h)
                    band_colors.append(col)

                fig_bands = go.Figure()
                fig_bands.add_trace(go.Bar(
                    x=band_labels, y=band_hours, marker_color=band_colors,
                    text=[f"{h:,} h<br>({h/len(cap)*100:.1f}%)" for h in band_hours],
                    textposition="outside",
                    hovertemplate="Band: %{x}<br>Hours: %{y:,}<br>"
                                 f"of {len(cap):,} total<extra></extra>",
                ))
                fig_bands.update_layout(
                    height=420, showlegend=False,
                    title="Capacity Band Distribution",
                    yaxis=dict(title="Hours"),
                    plot_bgcolor="#F8F7F4", paper_bgcolor="white",
                    margin=dict(l=60, r=20, t=50, b=50),
                )
                st.plotly_chart(fig_bands, use_container_width=True)

            with tl_pmvsfail:
                st.caption(
                    "Planned maintenance vs unplanned failure downtime per year. "
                    "High unplanned bars suggest the PM interval is too long or component reliability is low."
                )
                yr_pm_h, yr_fail_h = [], []
                for y in range(n_years):
                    sl = slice(y * 8760, (y + 1) * 8760)
                    yr_cap = cap[sl]
                    yr_pm  = pm_hist[sl]
                    down = yr_cap < 1.0
                    pm_down   = int((down & yr_pm).sum())
                    fail_down = int((down & ~yr_pm).sum())
                    yr_pm_h.append(pm_down)
                    yr_fail_h.append(fail_down)

                yr_x = list(range(1, n_years + 1))
                fig_pmf = go.Figure()
                fig_pmf.add_trace(go.Bar(
                    x=yr_x, y=yr_pm_h, name="Planned (PM)",
                    marker_color="#2196F3",
                    hovertemplate="Year %{x}<br>PM downtime: %{y:,} h<extra></extra>",
                ))
                fig_pmf.add_trace(go.Bar(
                    x=yr_x, y=yr_fail_h, name="Unplanned (failure)",
                    marker_color="#E24B4A",
                    hovertemplate="Year %{x}<br>Failure downtime: %{y:,} h<extra></extra>",
                ))
                fig_pmf.update_layout(
                    height=420, barmode="stack",
                    title="Downtime: Planned Maintenance vs Unplanned Failures",
                    xaxis=dict(title="Year", tickmode="linear", dtick=1),
                    yaxis=dict(title="Hours of reduced capacity"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    plot_bgcolor="#F8F7F4", paper_bgcolor="white",
                    margin=dict(l=60, r=20, t=50, b=50),
                )
                st.plotly_chart(fig_pmf, use_container_width=True)

                total_pm = sum(yr_pm_h)
                total_fail = sum(yr_fail_h)
                total_down = total_pm + total_fail
                st.caption(
                    f"**Total:** {total_down:,} h downtime over {n_years} years — "
                    f"{total_pm:,} h planned ({total_pm/max(total_down,1)*100:.0f}%), "
                    f"{total_fail:,} h unplanned ({total_fail/max(total_down,1)*100:.0f}%)"
                )

            with tl_compfail:
                st.caption(
                    "Failure count by component type over the full simulation. "
                    "Shows where maintenance effort should be focused."
                )
                model = tl["model"]
                comp_data = {
                    "EZ body":       sum(b.failures for b in model.ez_bodies),
                    "Stack":         sum(s.failures for stks in model.ez_stacks for s in stks),
                    "Comp block":    sum(u.failures for u in model.comp_block),
                    "Comp motor":    sum(u.failures for u in model.comp_motor),
                    "Comp seals":    sum(u.failures for u in model.comp_seals),
                    "EZ aux":        sum(u.failures for u in model.ez_aux),
                    "Comp aux":      sum(u.failures for u in model.comp_aux),
                    "Fill line aux": sum(u.failures for u in model.fill_lines),
                }
                comp_names  = list(comp_data.keys())
                comp_counts = list(comp_data.values())
                comp_colors = ["#16213e", "#0f3460", "#533483", "#7b2d8e",
                               "#a569bd", "#2196F3", "#607D8B", "#e94560"]

                fig_cf = go.Figure()
                fig_cf.add_trace(go.Bar(
                    x=comp_names, y=comp_counts,
                    marker_color=comp_colors[:len(comp_names)],
                    text=comp_counts, textposition="outside",
                    hovertemplate="%{x}<br>Failures: %{y}<extra></extra>",
                ))
                fig_cf.update_layout(
                    height=420, showlegend=False,
                    title=f"Failure Count by Component Type ({n_years} years)",
                    yaxis=dict(title="Number of failures"),
                    plot_bgcolor="#F8F7F4", paper_bgcolor="white",
                    margin=dict(l=60, r=20, t=50, b=50),
                )
                st.plotly_chart(fig_cf, use_container_width=True)

                # MTBF realized vs theoretical
                st.subheader("MTBF: Realised vs Theoretical")
                st.caption(
                    "Compares actual mean time between failures from the simulation "
                    "against the Weibull η (characteristic life) parameter. "
                    "Realised MTBF below η suggests PM resets or clustering effects."
                )
                sim_hours = n_years * 8760
                p = st.session_state["ram_params"]
                mtbf_rows = []
                def _mtbf_row(label, units, eta):
                    n_units = len(units)
                    total_f = sum(u.failures for u in units)
                    realised = (n_units * sim_hours) / total_f if total_f > 0 else float("inf")
                    return {"Component": label, "Units": n_units, "Failures": total_f,
                            "Realised MTBF (h)": f"{realised:,.0f}" if total_f > 0 else "—",
                            "Theoretical η (h)": f"{eta:,}"}

                mtbf_rows.append(_mtbf_row("EZ body",    model.ez_bodies, p["ez_eta"]))
                mtbf_rows.append(_mtbf_row("Stack",      [s for stks in model.ez_stacks for s in stks], p["stk_eta"]))
                mtbf_rows.append(_mtbf_row("Comp block", model.comp_block, p["cb_eta"]))
                mtbf_rows.append(_mtbf_row("Comp motor", model.comp_motor, p["cm_eta"]))
                mtbf_rows.append(_mtbf_row("Comp seals", model.comp_seals, p["cs_eta"]))

                import pandas as pd
                st.dataframe(pd.DataFrame(mtbf_rows), hide_index=True, use_container_width=True)

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
    sub_fleet, sub_single, sub_schedule = st.tabs([
        "📊 Fleet & Arrivals", "🔬 Single run", "📋 Schedule comparison",
    ])

    with sub_fleet:
        import plotly.graph_objects as go
        import numpy as np

        _containers = make_container_types(frac_a, frac_b, frac_c)
        _pattern    = make_arrival_pattern(pattern_type, peak_hour, peak_hour_2, peak_width, peak_width_2, peak_weight)
        _type_colors = {"Type-A": "#2196F3", "Type-B": "#FF9800", "Type-C": "#4CAF50"}

        # ── 1. Container fleet composition ──────────────────────
        st.subheader("Container Fleet")
        fig_fleet = go.Figure()
        for ct in _containers:
            fig_fleet.add_trace(go.Bar(
                x=[ct.name], y=[ct.fleet_fraction * 100],
                name=f"{ct.name} ({ct.max_capacity_kg:.0f} kg)",
                marker_color=_type_colors[ct.name],
                text=f"{ct.fleet_fraction*100:.0f}%", textposition="outside",
                hovertemplate=f"{ct.name}<br>{ct.max_capacity_kg:.0f} kg capacity<br>"
                              f"Fill rate: {ct.max_fill_rate_kg_per_hr:.0f} kg/hr<br>"
                              f"Fleet share: %{{y:.1f}}%<extra></extra>",
            ))
        fig_fleet.update_layout(
            height=320, showlegend=True, barmode="group",
            yaxis=dict(title="Fleet share (%)", range=[0, 110]),
            plot_bgcolor="#F8F7F4", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_fleet, use_container_width=True)

        # ── 2. Arrival probability distribution ─────────────────
        st.subheader("Arrival Pattern")
        hours = np.linspace(0, 24, 500)
        rates = np.array([_pattern.rate_at_hour(h) for h in hours])
        rate_sum = np.trapezoid(rates, hours)
        pdf = rates / rate_sum * float(avg_arrivals)

        fig_pdf = go.Figure()
        fig_pdf.add_trace(go.Scatter(
            x=hours, y=pdf, mode="lines",
            line=dict(color="#333", width=2),
            fill="tozeroy", fillcolor="rgba(33,150,243,0.12)",
            name="Arrival rate",
            hovertemplate="Hour: %{x:.1f}<br>Rate: %{y:.2f} /hr<extra></extra>",
        ))
        fig_pdf.update_layout(
            height=320,
            xaxis=dict(title="Hour of day", dtick=2, range=[0, 24]),
            yaxis=dict(title="Expected arrivals / hr"),
            title=f"Arrival distribution — {pattern_type} ({avg_arrivals:.1f} / day)",
            plot_bgcolor="#F8F7F4", paper_bgcolor="white",
            margin=dict(l=50, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_pdf, use_container_width=True)

        # ── 3. Animated arrival scatter on the PDF ──────────────
        st.subheader("Simulated Arrivals Preview")
        anim_speed = st.slider("Animation speed (ms/frame)", 50, 2000, 400, 50, key="anim_speed")

        n_preview_days = 365
        n_frames = 50
        days_per_frame = max(n_preview_days // n_frames, 1)
        rng_arrivals = np.random.default_rng(int(arrival_seed))
        rng_containers = np.random.default_rng(int(container_seed))

        # Fast generation: draw daily counts then sample hours from PDF
        fracs = np.array([ct.fleet_fraction for ct in _containers])
        cum_fracs = np.cumsum(fracs)
        hourly_rates = np.array([_pattern.rate_at_hour(h) for h in np.linspace(0, 24, 1440)])
        hourly_cdf = np.cumsum(hourly_rates)
        hourly_cdf /= hourly_cdf[-1]

        daily_counts = rng_arrivals.poisson(float(avg_arrivals), size=n_preview_days)
        total_n = int(daily_counts.sum())
        u_hours = rng_arrivals.random(total_n)
        arrival_hours_all = np.interp(u_hours, hourly_cdf, np.linspace(0, 24, 1440))
        u_types = rng_containers.random(total_n)
        type_idx = np.searchsorted(cum_fracs, u_types, side="right").clip(0, len(_containers) - 1)
        type_names = np.array([_containers[i].name for i in type_idx])
        arrival_days_all = np.repeat(np.arange(1, n_preview_days + 1), daily_counts)

        pdf_x, pdf_y = hours, pdf
        y_max = float(max(pdf_y))
        jitter_all = rng_arrivals.uniform(0, y_max * 0.15, size=total_n)

        base_traces = [go.Scatter(
            x=pdf_x, y=pdf_y, mode="lines",
            line=dict(color="#333", width=2),
            fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
            name="Arrival rate", showlegend=True, hoverinfo="skip",
        )]
        for ct in _containers:
            base_traces.append(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=_type_colors[ct.name], size=8),
                name=f"{ct.name} ({ct.max_capacity_kg:.0f} kg)", showlegend=True,
            ))

        frames = []
        frame_days = list(range(days_per_frame, n_preview_days + 1, days_per_frame))
        if frame_days[-1] != n_preview_days:
            frame_days.append(n_preview_days)
        for fd in frame_days:
            mask = arrival_days_all <= fd
            scatter_traces = []
            for ct in _containers:
                ct_mask = mask & (type_names == ct.name)
                scatter_traces.append(go.Scatter(
                    x=arrival_hours_all[ct_mask], y=-jitter_all[ct_mask],
                    mode="markers",
                    marker=dict(color=_type_colors[ct.name], size=4, opacity=0.5,
                                line=dict(width=0)),
                    showlegend=False,
                    hovertemplate=f"{ct.name}<br>Hour: %{{x:.1f}}<extra></extra>",
                ))
            n_so_far = int(mask.sum())
            frames.append(go.Frame(
                data=base_traces + scatter_traces,
                name=f"Day {fd}",
                layout=go.Layout(title=f"Arrivals — Days 1–{fd}  ({n_so_far} containers)"),
            ))

        initial_scatters = [go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=_type_colors[ct.name], size=4, opacity=0.5),
            showlegend=False,
        ) for ct in _containers]

        fig_anim = go.Figure(data=base_traces + initial_scatters, frames=frames)
        fig_anim.update_layout(
            height=480,
            xaxis=dict(title="Hour of day", dtick=2, range=[0, 24]),
            yaxis=dict(title="Arrivals / hr", range=[-(y_max * 0.25), y_max * 1.15]),
            title=f"Arrivals — press ▶ to animate ({n_preview_days} days)",
            plot_bgcolor="#F8F7F4", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=50, r=20, t=60, b=50),
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.0, xanchor="left", y=-0.15, yanchor="top",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=anim_speed, redraw=True),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0, steps=[
                    dict(args=[[f.name], dict(frame=dict(duration=anim_speed, redraw=True),
                                              mode="immediate")],
                         label=f"Day {fd}", method="animate")
                    for fd, f in zip(frame_days, frames)
                ],
                x=0.15, len=0.85, xanchor="left",
                y=-0.10, yanchor="top",
                currentvalue=dict(prefix="", visible=True),
                transition=dict(duration=min(anim_speed, 300)),
            )],
        )
        st.plotly_chart(fig_anim, use_container_width=True)

    with sub_single:
        st.header("Single Simulation Run")
        schedule_label = st.selectbox("Staff schedule", ALL_SCHEDULES, index=0, key="ops_sched")
        if st.button("▶ Run simulation", type="primary", key="btn_single"):
            # Capture all state at press time
            _rel_model = make_rel_model(TOPOLOGY, RELIABILITY_SEED, RELIABILITY_ON)
            _plant     = po.HydrogenPlant(topology=TOPOLOGY, step_minutes=1)
            _schedule  = make_schedule(schedule_label)
            _containers = make_container_types(frac_a, frac_b, frac_c)
            _pattern    = make_arrival_pattern(pattern_type, peak_hour, peak_hour_2, peak_width, peak_width_2, peak_weight)
            with st.spinner("Simulating..."):
                result = po.run_simulation(
                    container_types=_containers,
                    plant=_plant, days=int(sim_days), schedule=_schedule,
                    avg_arrivals_per_day=float(avg_arrivals), step_minutes=1,
                    random_seed=int(arrival_seed),
                    container_seed=int(container_seed),
                    arrival_pattern=_pattern, reliability_model=_rel_model,
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
                st.plotly_chart(ep.plot_waterfall(econ), use_container_width=True)

            plots = rpo.plot_results(result)
            with st.expander("Operational Plots"):
                st.plotly_chart(plots["pressure_production"], use_container_width=True)
                st.plotly_chart(plots["queue_zones"], use_container_width=True)
            with st.expander("Container Queue"):
                st.plotly_chart(plots["time_histograms"], use_container_width=True)
                if "type_boxplots" in plots:
                    st.plotly_chart(plots["type_boxplots"], use_container_width=True)
            with st.expander("Compressor Data"):
                if "comp_filling_activity" in plots:
                    st.plotly_chart(plots["comp_filling_activity"], use_container_width=True)
                if "comp_utilisation_bar" in plots:
                    st.plotly_chart(plots["comp_utilisation_bar"], use_container_width=True)
                st.plotly_chart(plots["pressure_distribution"], use_container_width=True)
            with st.expander("Power Use"):
                st.plotly_chart(plots["power"], use_container_width=True)

    with sub_schedule:
        st.header("Schedule Comparison")
        compare_scheds = st.multiselect("Schedules to compare", ALL_SCHEDULES,
                                         default=["unmanned","8-16_closed","8-20_open","24_7"],
                                         key="cmp_scheds")
        if st.button("▶ Compare schedules", type="primary", key="btn_cmp") and compare_scheds:
            _containers = make_container_types(frac_a, frac_b, frac_c)
            _pattern    = make_arrival_pattern(pattern_type, peak_hour, peak_hour_2, peak_width, peak_width_2, peak_weight)
            rows = []
            with st.spinner("Running..."):
                for lbl in compare_scheds:
                    _plant  = po.HydrogenPlant(topology=TOPOLOGY, step_minutes=1)
                    _rel    = make_rel_model(TOPOLOGY, RELIABILITY_SEED, RELIABILITY_ON)
                    result  = po.run_simulation(
                        container_types=_containers,
                        plant=_plant, days=int(sim_days), schedule=make_schedule(lbl),
                        avg_arrivals_per_day=float(avg_arrivals), step_minutes=1,
                        arrival_pattern=_pattern, reliability_model=_rel,
                    )
                    kpis = rpo.compute_kpis(result, _plant, _containers,
                                             float(avg_arrivals), int(sim_days))
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

            # Economics plots
            econ_rows = []
            for r in rows:
                econ_rows.append({
                    "schedule_label": r["schedule"],
                    **{k: v for k, v in r.items() if "kr" in k}
                })
            econ_df_cmp = pd.DataFrame(econ_rows)
            if not econ_df_cmp.empty and "revenue_kr_annual" in econ_df_cmp.columns:
                sched_order = [s for s in ALL_SCHEDULES if s in econ_df_cmp["schedule_label"].values]
                summary = eco.economics_summary(econ_df_cmp, group_by=["schedule_label"])
                summary = (summary.set_index("schedule_label")
                           .reindex(sched_order).dropna(how="all").reset_index())
                st.subheader("Revenue vs. cost breakdown")
                st.plotly_chart(ep.plot_stacked_bar(summary), use_container_width=True)
                st.subheader("Net result spread")
                st.plotly_chart(ep.plot_net_result_spread(econ_df_cmp, sched_order, float(avg_arrivals), 1), use_container_width=True)


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
            [("off", False), ("on", True)]
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
                st.plotly_chart(ep.plot_net_result_spread(headline, sched_order, hr, n_seeds), use_container_width=True)
                st.subheader("Revenue vs. cost")
                st.plotly_chart(ep.plot_stacked_bar(summary), use_container_width=True)
                st.subheader("Sensitivity vs. arrival rate")
                all_s   = [s for s in ALL_SCHEDULES if s in filtered["schedule_label"].unique()]
                filters = {"reliability_label": rel_f} if rel_f else None
                st.plotly_chart(ep.plot_sensitivity(filtered, all_s, filters=filters), use_container_width=True)


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
