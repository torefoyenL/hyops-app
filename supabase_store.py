# -*- coding: utf-8 -*-
"""
supabase_store.py
=================
Save / load named plant configuration snapshots to Supabase.

Each config is a full JSON snapshot of all plant parameters (topology,
RAM, PM, BOM, equipment library, container fleet, arrivals, costs),
stored in a single `configs` table with a JSONB `config_data` column.

On load, the user picks which sections to apply — the rest of the
current session state is left untouched.
"""

import streamlit as st

CONFIG_VERSION = 1

SECTION_KEYS = [
    "topology", "ram", "pm", "bom", "eq_lib",
    "container_fleet", "arrivals", "cost_revenue",
]

SECTION_LABELS = {
    "topology":        "Topology",
    "ram":             "RAM Parameters",
    "pm":              "Planned Maintenance",
    "bom":             "Bill of Materials",
    "eq_lib":          "Equipment Library",
    "container_fleet": "Container Fleet",
    "arrivals":        "Arrivals",
    "cost_revenue":    "Cost & Revenue",
}


# ============================================================
# Connection
# ============================================================

@st.cache_resource
def get_supabase_client():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    except (KeyError, FileNotFoundError):
        return None
    from supabase import create_client
    return create_client(url, key)


# ============================================================
# Snapshot capture
# ============================================================

def capture_snapshot(session_state, sidebar_values: dict) -> dict:
    sv = sidebar_values
    return {
        "version": CONFIG_VERSION,
        "sections": {
            "topology": {
                "topology_mode":         sv.get("topology_mode"),
                "n_ez":                  sv.get("n_ez"),
                "stacks":               sv.get("stacks"),
                "ez_kg_hr_each":        sv.get("ez_kg_hr_each"),
                "n_comp":               sv.get("n_comp"),
                "comp_kg_hr_each":      sv.get("comp_kg_hr_each"),
                "n_fill":               sv.get("n_fill"),
                "lines_per_comp":       sv.get("lines_per_comp"),
                "n_trains":             sv.get("n_trains"),
                "ez_per_train":         sv.get("ez_per_train"),
                "comp_per_train":       sv.get("comp_per_train"),
                "ez_kg_hr_each_train":  sv.get("ez_kg_hr_each_train"),
                "comp_kg_hr_each_train": sv.get("comp_kg_hr_each_train"),
                "lines_per_train":      sv.get("lines_per_train"),
            },
            "ram": {
                "ram_params":       dict(session_state.get("ram_params", {})),
                "reliability_on":   sv.get("reliability_on", False),
                "reliability_seed": sv.get("reliability_seed", 42),
            },
            "pm": {
                "pm_config":  {k: dict(v) for k, v in session_state.get("pm_config", {}).items()},
                "pm_offsets": dict(session_state.get("pm_offsets", {"ez": [], "comp": []})),
            },
            "bom": {
                "bom": {n: dict(c) for n, c in session_state.get("bom", {}).items()},
            },
            "eq_lib": {
                "eq_lib": {k: dict(v) for k, v in session_state.get("eq_lib", {}).items()},
            },
            "container_fleet": {
                "frac_a":                  sv.get("frac_a", 0.3),
                "frac_b":                  sv.get("frac_b", 0.5),
                "frac_c":                  sv.get("frac_c", 0.2),
                "container_seed":          sv.get("container_seed", 7),
                "container_costs_monthly": dict(session_state.get("container_costs_monthly",
                                                {"Type-A": 0, "Type-B": 0, "Type-C": 0})),
            },
            "arrivals": {
                "avg_arrivals":  sv.get("avg_arrivals", 3.0),
                "pattern_type":  sv.get("pattern_type", "uniform"),
                "peak_hour":     sv.get("peak_hour"),
                "peak_hour_2":   sv.get("peak_hour_2"),
                "peak_width":    sv.get("peak_width", 3.0),
                "peak_width_2":  sv.get("peak_width_2", 3.0),
                "peak_weight":   sv.get("peak_weight", 0.5),
                "sim_days":      sv.get("sim_days", 31),
                "arrival_seed":  sv.get("arrival_seed", 42),
            },
            "cost_revenue": {
                "margin_kr":    session_state.get("margin_kr", 30.0),
                "queue_cost_kr": session_state.get("queue_cost_kr", 1500.0),
                "staff_costs":  dict(session_state.get("staff_costs", {})),
            },
        },
    }


# ============================================================
# Section apply (selective load)
# ============================================================

def apply_sections(config_data: dict, sections: list, session_state):
    sec = config_data.get("sections", {})
    defaults = {}

    if "topology" in sections and "topology" in sec:
        defaults.update(sec["topology"])

    if "ram" in sections and "ram" in sec:
        ram = sec["ram"]
        if "ram_params" in ram:
            session_state["ram_params"] = dict(ram["ram_params"])
        defaults["reliability_on"]   = ram.get("reliability_on", False)
        defaults["reliability_seed"] = ram.get("reliability_seed", 42)

    if "pm" in sections and "pm" in sec:
        pm = sec["pm"]
        if "pm_config" in pm:
            session_state["pm_config"] = {k: dict(v) for k, v in pm["pm_config"].items()}
        if "pm_offsets" in pm:
            session_state["pm_offsets"] = dict(pm["pm_offsets"])

    if "bom" in sections and "bom" in sec:
        if "bom" in sec["bom"]:
            session_state["bom"] = {n: dict(c) for n, c in sec["bom"]["bom"].items()}

    if "eq_lib" in sections and "eq_lib" in sec:
        if "eq_lib" in sec["eq_lib"]:
            session_state["eq_lib"] = {k: dict(v) for k, v in sec["eq_lib"]["eq_lib"].items()}

    if "container_fleet" in sections and "container_fleet" in sec:
        cf = sec["container_fleet"]
        session_state["fa"] = cf.get("frac_a", 0.3)
        session_state["fb"] = cf.get("frac_b", 0.5)
        session_state["fc"] = cf.get("frac_c", 0.2)
        session_state["container_seed"] = cf.get("container_seed", 7)
        if "container_costs_monthly" in cf:
            session_state["container_costs_monthly"] = dict(cf["container_costs_monthly"])

    if "arrivals" in sections and "arrivals" in sec:
        arr = sec["arrivals"]
        defaults.update({k: arr[k] for k in arr if k != "arrival_seed"})
        session_state["arrival_seed"] = arr.get("arrival_seed", 42)

    if "cost_revenue" in sections and "cost_revenue" in sec:
        cr = sec["cost_revenue"]
        session_state["margin_kr"]   = cr.get("margin_kr", 30.0)
        session_state["queue_cost_kr"] = cr.get("queue_cost_kr", 1500.0)
        if "staff_costs" in cr:
            session_state["staff_costs"] = dict(cr["staff_costs"])

    if defaults:
        session_state["_cfg_defaults"] = defaults


# ============================================================
# CRUD
# ============================================================

def save_config(name: str, description: str, config_data: dict):
    client = get_supabase_client()
    if client is None:
        return None, "Supabase not configured"
    try:
        result = client.table("configs").insert({
            "name": name,
            "description": description,
            "config_data": config_data,
        }).execute()
        return result.data[0] if result.data else None, None
    except Exception as e:
        return None, str(e)


def list_configs():
    client = get_supabase_client()
    if client is None:
        return []
    try:
        result = (client.table("configs")
                  .select("id, name, description, created_at")
                  .order("created_at", desc=True)
                  .execute())
        return result.data or []
    except Exception:
        return []


def load_config(config_id: str):
    client = get_supabase_client()
    if client is None:
        return None, "Supabase not configured"
    try:
        result = (client.table("configs")
                  .select("*")
                  .eq("id", config_id)
                  .single()
                  .execute())
        return result.data, None
    except Exception as e:
        return None, str(e)


def delete_config(config_id: str):
    client = get_supabase_client()
    if client is None:
        return False, "Supabase not configured"
    try:
        client.table("configs").delete().eq("id", config_id).execute()
        return True, None
    except Exception as e:
        return False, str(e)
