"""Headless engineering core for the PySide6 spread footing app.

This module contains all ACI 318-25 calculations, soil bearing distribution,
drawing, and report logic for isolated spread footings.
"""
from __future__ import annotations

import io
import math
import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

APP_TITLE = "Spread Footing Designer"
APP_SUBTITLE = "ACI 318-25 isolated footing design aid | Soil bearing & RC design"


# =============================================================================
# CONSTANTS & DATABASES
# =============================================================================

BAR_DATABASE_MM = {
    "DB10": {"diameter": 10.0, "area": 78.5},
    "DB12": {"diameter": 12.0, "area": 113.1},
    "DB16": {"diameter": 16.0, "area": 201.1},
    "DB20": {"diameter": 20.0, "area": 314.2},
    "DB25": {"diameter": 25.0, "area": 490.9},
    "DB28": {"diameter": 28.0, "area": 615.8},
    "DB32": {"diameter": 32.0, "area": 804.2},
    "DB36": {"diameter": 36.0, "area": 1017.9},
    "DB40": {"diameter": 40.0, "area": 1256.6},
}

ACI_REFERENCES = {
    "general": "ACI CODE-318-25 should be used directly for final design and detailing.",
    "foundations": "ACI 318-25 Chapter 13 — Foundations.",
    "flexure": "ACI 318-25 §22.2 — Flexural strength. Rectangular section, singly-reinforced.",
    "one_way_shear": "ACI 318-25 §22.5 — One-way shear. Members bearing directly on soil: λs = 1.0.",
    "two_way_shear": "ACI 318-25 §22.6 — Two-way punching shear. Members bearing directly on soil: λs = 1.0.",
    "bearing": "ACI 318-25 §22.8 — Bearing strength with area ratio confinement effect.",
    "development": "ACI 318-25 §25.4 — Development and anchorage of reinforcement.",
    "dowels": "ACI 318-25 §16.3 — Column-to-footing interface: dowels or extended column bars.",
    "cover": "ACI 318-25 Table 20.6.1.3.1 — Cast against earth: 75 mm minimum.",
    "shrinkage_temp": "ACI 318-25 §13.3.3.3 — Minimum shrinkage/temperature reinforcement in footings.",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Material:
    fc_MPa: float = 28.0
    fy_MPa: float = 420.0
    lambda_c: float = 1.0
    gamma_conc_kN_m3: float = 24.0
    phi_flexure: float = 0.90
    phi_shear: float = 0.75
    phi_bearing: float = 0.65


@dataclass
class FootingGeometry:
    # Footing dimensions
    footing_length_x_mm: float = 2400.0
    footing_width_y_mm: float = 2400.0
    footing_thickness_mm: float = 500.0
    bottom_cover_mm: float = 75.0
    side_cover_mm: float = 75.0
    top_cover_mm: float = 50.0

    # Column/pedestal dimensions
    column_bx_mm: float = 400.0
    column_by_mm: float = 400.0
    column_shape: str = "rectangular"
    column_location: str = "Interior"

    # Pedestal (optional)
    use_pedestal: bool = False
    pedestal_bx_mm: float = 600.0
    pedestal_by_mm: float = 600.0
    pedestal_height_mm: float = 500.0

    # Embedment
    footing_embedment_depth_mm: float = 1500.0
    soil_above_footing_mm: float = 1000.0
    gamma_soil_kN_m3: float = 18.0

    # Soil bearing
    allowable_bearing_kPa: float = 150.0
    friction_coefficient: float = 0.40
    passive_pressure_kPa: float = 0.0


@dataclass
class Reinforcement:
    main_bar_x: str = "DB16"
    main_bar_y: str = "DB16"
    spacing_x_mm: float = 200.0
    spacing_y_mm: float = 200.0
    dowel_bar: str = "DB20"
    dowel_count: int = 8
    preferred_spacing_step_mm: float = 25.0
    hook_extension_mm: float = 150.0


@dataclass
class LoadCase:
    name: str = "ULS-1.2D+1.6L"
    case_type: str = "Ultimate"
    Pu_kN: float = 1000.0
    Mux_kNm: float = 0.0
    Muy_kNm: float = 0.0
    Vux_kN: float = 0.0
    Vuy_kN: float = 0.0


@dataclass
class CheckResult:
    name: str
    demand: float
    capacity: float
    ratio: float
    unit: str
    status: str
    note: str = ""


@dataclass
class DesignState:
    material: Material
    geometry: FootingGeometry
    reinforcement: Reinforcement
    loadcase: LoadCase
    effective_depth_x_mm: float = 0.0
    effective_depth_y_mm: float = 0.0
    self_weight_kN: float = 0.0
    overburden_weight_kN: float = 0.0
    total_axial_kN: float = 0.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return float(value)
    except Exception:
        return default

def fmt(value: float, nd: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:,.{nd}f}"

def status_from_ratio(ratio: float, warn_limit: float = 0.95) -> str:
    if ratio <= warn_limit:
        return "PASS"
    if ratio <= 1.0:
        return "NEAR"
    return "FAIL"

def bar_area(bar_name: str) -> float:
    return BAR_DATABASE_MM.get(bar_name, BAR_DATABASE_MM["DB16"])["area"]

def bar_diameter(bar_name: str) -> float:
    return BAR_DATABASE_MM.get(bar_name, BAR_DATABASE_MM["DB16"])["diameter"]

def round_down_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step

def beta1_aci(fc_MPa: float) -> float:
    if fc_MPa <= 28.0:
        return 0.85
    return max(0.65, 0.85 - 0.05 * ((fc_MPa - 28.0) / 7.0))

def mm_to_m(value_mm: float) -> float:
    return value_mm / 1000.0

def kNm_to_Nmm(value_kNm: float) -> float:
    return value_kNm * 1_000_000.0

def Nmm_to_kNm(value_Nmm: float) -> float:
    return value_Nmm / 1_000_000.0

def kN_from_N(value_N: float) -> float:
    return value_N / 1000.0

# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def get_loaded_area_dims(geom: FootingGeometry) -> Tuple[float, float]:
    """Return effective dimensions of the loaded area on the footing (pedestal or column)."""
    if geom.use_pedestal:
        return geom.pedestal_bx_mm, geom.pedestal_by_mm
    
    if geom.column_shape.lower() == "circular":
        # Equivalent square for circular column: b = sqrt(pi/4) * D
        b_eq = math.sqrt(math.pi / 4) * geom.column_bx_mm
        return b_eq, b_eq
    
    return geom.column_bx_mm, geom.column_by_mm

def get_column_dims(geom: FootingGeometry) -> Tuple[float, float]:
    """Return effective dimensions of the column."""
    if geom.column_shape.lower() == "circular":
        b_eq = math.sqrt(math.pi / 4) * geom.column_bx_mm
        return b_eq, b_eq
    return geom.column_bx_mm, geom.column_by_mm

# =============================================================================
# GRAVITY LOADS
# =============================================================================

def self_weight_kN(geom: FootingGeometry, mat: Material) -> float:
    vol = mm_to_m(geom.footing_length_x_mm) * mm_to_m(geom.footing_width_y_mm) * mm_to_m(geom.footing_thickness_mm)
    return vol * mat.gamma_conc_kN_m3

def overburden_weight_kN(geom: FootingGeometry) -> float:
    footing_area = mm_to_m(geom.footing_length_x_mm) * mm_to_m(geom.footing_width_y_mm)
    loaded_bx, loaded_by = get_loaded_area_dims(geom)
    loaded_area = mm_to_m(loaded_bx) * mm_to_m(loaded_by)
    return max(0.0, footing_area - loaded_area) * mm_to_m(geom.soil_above_footing_mm) * geom.gamma_soil_kN_m3

# =============================================================================
# SOIL MECHANICS & STABILITY
# =============================================================================

def bearing_pressure_distribution(
    P_total_kN: float,
    Mx_kNm: float,
    My_kNm: float,
    Lx_mm: float,
    Ly_mm: float,
) -> Dict[str, Any]:
    """
    Computes soil bearing pressure using Kern/Meyerhof method.
    Positive P is compression.
    """
    if P_total_kN <= 0:
        ex_m = abs(My_kNm / P_total_kN) if abs(P_total_kN) > 1e-9 else 999.9
        ey_m = abs(Mx_kNm / P_total_kN) if abs(P_total_kN) > 1e-9 else 999.9
        return {
            "q_max_kPa": 0.0, "q_min_kPa": 0.0, "q_avg_kPa": 0.0,
            "ex_mm": ex_m * 1000.0, "ey_mm": ey_m * 1000.0, "contact_type": "uplift",
            "kern_check": False, "effective_area_m2": 0.0
        }
    
    Lx_m = mm_to_m(Lx_mm)
    Ly_m = mm_to_m(Ly_mm)
    A_m2 = Lx_m * Ly_m
    
    # Eccentricities
    # Note: Mx is moment about X-axis, causes eccentricity in Y direction.
    # My is moment about Y-axis, causes eccentricity in X direction.
    ey_m = abs(Mx_kNm / P_total_kN)
    ex_m = abs(My_kNm / P_total_kN)
    
    ey_mm = ey_m * 1000
    ex_mm = ex_m * 1000
    
    # Kern limits
    in_kern_x = ex_m <= (Lx_m / 6.0)
    in_kern_y = ey_m <= (Ly_m / 6.0)
    kern_check = in_kern_x and in_kern_y
    
    if kern_check:
        # Full contact
        term_axial = P_total_kN / A_m2
        term_mx = (6 * abs(Mx_kNm)) / (Lx_m * Ly_m**2) if Ly_m > 0 else 0
        term_my = (6 * abs(My_kNm)) / (Lx_m**2 * Ly_m) if Lx_m > 0 else 0
        
        q_max = term_axial + term_mx + term_my
        q_min = term_axial - term_mx - term_my
        
        contact_type = "full"
        eff_area = A_m2
    else:
        # Partial contact (Meyerhof approximation)
        Lx_eff = Lx_m - 2 * ex_m
        Ly_eff = Ly_m - 2 * ey_m
        
        # Prevent negative effective dimensions
        Lx_eff = max(0.001, Lx_eff)
        Ly_eff = max(0.001, Ly_eff)
        
        eff_area = Lx_eff * Ly_eff
        q_max = P_total_kN / eff_area
        q_min = 0.0
        
        if not in_kern_x and not in_kern_y:
            contact_type = "partial_xy"
        elif not in_kern_x:
            contact_type = "partial_x"
        else:
            contact_type = "partial_y"
            
    return {
        "q_max_kPa": q_max,
        "q_min_kPa": max(0.0, q_min),
        "q_avg_kPa": (q_max + max(0.0, q_min)) / 2.0,
        "ex_mm": ex_mm,
        "ey_mm": ey_mm,
        "contact_type": contact_type,
        "kern_check": kern_check,
        "effective_area_m2": eff_area
    }


def overturning_check(
    P_service_kN: float,
    M_service_kNm: float,
    footing_dim_mm: float,
    V_service_kN: float,
    depth_mm: float,
) -> Dict[str, Any]:
    """Overturning stability check for one direction."""
    # Resisting moment from gravity
    M_resist_kNm = P_service_kN * mm_to_m(footing_dim_mm) / 2.0
    
    # Overturning moment
    M_overturn_kNm = abs(M_service_kNm) + abs(V_service_kN) * mm_to_m(depth_mm)
    
    if M_overturn_kNm <= 1e-9:
        fs = 999.9
    else:
        fs = M_resist_kNm / M_overturn_kNm
        
    return {
        "fs_overturning": fs,
        "M_resist_kNm": M_resist_kNm,
        "M_overturn_kNm": M_overturn_kNm,
        "status": "PASS" if fs >= 1.5 else "FAIL"
    }


def sliding_check(
    P_service_kN: float,
    V_service_kN: float,
    friction_coeff: float,
    passive_kPa: float,
    depth_mm: float,
    footing_width_mm: float,
) -> Dict[str, Any]:
    """Sliding stability check for one direction."""
    # Friction resistance
    F_friction_kN = max(0.0, P_service_kN) * friction_coeff
    
    # Passive resistance
    F_passive_kN = passive_kPa * mm_to_m(depth_mm) * mm_to_m(footing_width_mm)
    
    F_resist_kN = F_friction_kN + F_passive_kN
    V_demand_kN = abs(V_service_kN)
    
    if V_demand_kN <= 1e-9:
        fs = 999.9
    else:
        fs = F_resist_kN / V_demand_kN
        
    return {
        "fs_sliding": fs,
        "F_resist_kN": F_resist_kN,
        "F_friction_kN": F_friction_kN,
        "F_passive_kN": F_passive_kN,
        "V_demand_kN": V_demand_kN,
        "status": "PASS" if fs >= 1.5 else "FAIL"
    }

# =============================================================================
# RC STRUCTURAL DESIGN (FACTORED LOADS)
# =============================================================================

def effective_depths(geom: FootingGeometry, reinf: Reinforcement) -> Tuple[float, float]:
    db_x = bar_diameter(reinf.main_bar_x)
    db_y = bar_diameter(reinf.main_bar_y)
    
    # Assume X bars are bottom-most layer
    d_x = geom.footing_thickness_mm - geom.bottom_cover_mm - db_x / 2.0
    d_y = geom.footing_thickness_mm - geom.bottom_cover_mm - db_x - db_y / 2.0
    
    return max(0.0, d_x), max(0.0, d_y)

def flexural_demand(
    q_u_max_kPa: float,
    q_u_min_kPa: float,
    footing_dim_mm: float,
    column_dim_mm: float,
    strip_width_mm: float,
) -> Dict[str, float]:
    """
    Computes cantilever factored moment at face of column.
    Uses a simplified trapezoidal pressure distribution approach.
    """
    overhang_m = mm_to_m(footing_dim_mm - column_dim_mm) / 2.0
    if overhang_m <= 0:
        return {"Mu_kNm": 0.0, "q_at_face_kPa": q_u_max_kPa, "overhang_mm": 0.0}
        
    footing_L_m = mm_to_m(footing_dim_mm)
    strip_W_m = mm_to_m(strip_width_mm)
    
    # Interpolate pressure at column face
    # Assuming q_max is at the tip of the overhang
    slope = (q_u_max_kPa - q_u_min_kPa) / footing_L_m if footing_L_m > 0 else 0
    q_at_face = q_u_max_kPa - slope * overhang_m
    
    # Moment = (rectangle part) + (triangle part)
    M_rect = q_at_face * (overhang_m**2 / 2.0) * strip_W_m
    M_tri = (q_u_max_kPa - q_at_face) * 0.5 * overhang_m * (2.0/3.0 * overhang_m) * strip_W_m
    
    Mu = M_rect + M_tri
    
    return {
        "Mu_kNm": Mu,
        "q_at_face_kPa": q_at_face,
        "overhang_mm": overhang_m * 1000
    }

def flexural_As_required(
    Mu_kNm: float,
    b_mm: float,
    d_mm: float,
    fc_MPa: float,
    fy_MPa: float,
    phi: float = 0.90,
) -> Dict[str, float]:
    """Singly-reinforced rectangular section flexural design."""
    Mu_Nmm = abs(kNm_to_Nmm(Mu_kNm))
    
    # Min As for footings (shrinkage & temp per ACI 318-25)
    # Using the 0.0018 gross area rule for fy >= 420
    h_mm = d_mm + 75.0 + 10.0 # Approx gross thickness
    rho_min = 0.0018 if fy_MPa >= 420 else 0.0020
    As_min = rho_min * b_mm * h_mm
    
    if Mu_Nmm <= 1e-9:
        return {
            "As_req_mm2": As_min,
            "As_strength_mm2": 0.0,
            "As_min_mm2": As_min,
            "rho": As_min / (b_mm * d_mm) if d_mm > 0 else 0,
            "a_mm": 0.0,
            "phiMn_kNm": 0.0,
        }

    A = phi * fy_MPa * fy_MPa / (2.0 * 0.85 * fc_MPa * b_mm)
    B = -phi * fy_MPa * d_mm
    C = Mu_Nmm

    disc = B * B - 4.0 * A * C
    if disc < 0:
        As_strength = np.nan # Needs compression steel or deeper section
    else:
        root1 = (-B - math.sqrt(disc)) / (2.0 * A)
        root2 = (-B + math.sqrt(disc)) / (2.0 * A)
        candidates = [r for r in (root1, root2) if r > 0]
        As_strength = min(candidates) if candidates else np.nan

    As_req = max(As_min, As_strength if np.isfinite(As_strength) else 1e99)
    a = As_req * fy_MPa / (0.85 * fc_MPa * b_mm)
    phiMn = phi * As_req * fy_MPa * (d_mm - a / 2.0)
    
    return {
        "As_req_mm2": As_req,
        "As_strength_mm2": As_strength,
        "As_min_mm2": As_min,
        "rho": As_req / (b_mm * d_mm) if d_mm > 0 else 0,
        "a_mm": a,
        "phiMn_kNm": Nmm_to_kNm(phiMn),
    }

def flexural_capacity(
    As_mm2: float,
    b_mm: float,
    d_mm: float,
    fc_MPa: float,
    fy_MPa: float,
    phi: float = 0.90,
) -> Dict[str, float]:
    a = As_mm2 * fy_MPa / (0.85 * fc_MPa * b_mm)
    Mn_Nmm = As_mm2 * fy_MPa * max(d_mm - a / 2.0, 0.0)
    phiMn_Nmm = phi * Mn_Nmm
    c = a / max(beta1_aci(fc_MPa), 1e-9)
    eps_t = 0.003 * (d_mm - c) / max(c, 1e-9) if c > 0 else 0.01
    return {
        "a_mm": a,
        "c_mm": c,
        "eps_t": eps_t,
        "Mn_kNm": Nmm_to_kNm(Mn_Nmm),
        "phiMn_kNm": Nmm_to_kNm(phiMn_Nmm),
    }

def spacing_for_As(
    As_req_mm2: float,
    bar: str,
    strip_width_mm: float,
    preferred_step_mm: float = 25.0,
    s_min_mm: float = 75.0,
    s_max_mm: float = 450.0,
) -> Dict[str, float]:
    Ab = bar_area(bar)
    if As_req_mm2 <= 0:
        return {"spacing_req_mm": s_max_mm, "spacing_use_mm": s_max_mm, "n_bars": 0, "As_prov_mm2": 0.0}
    s_req = Ab * strip_width_mm / As_req_mm2
    s_use = min(s_max_mm, max(s_min_mm, round_down_to_step(s_req, preferred_step_mm)))
    n_bars = int(math.floor(strip_width_mm / s_use)) + 1
    As_prov = n_bars * Ab
    return {"spacing_req_mm": s_req, "spacing_use_mm": s_use, "n_bars": n_bars, "As_prov_mm2": As_prov}

def one_way_shear_demand(
    q_u_max_kPa: float,
    q_u_min_kPa: float,
    footing_dim_mm: float,
    column_dim_mm: float,
    d_mm: float,
    strip_width_mm: float,
) -> Dict[str, float]:
    """Demand at distance d from face."""
    overhang_m = mm_to_m(footing_dim_mm - column_dim_mm) / 2.0
    critical_length_m = overhang_m - mm_to_m(d_mm)
    
    if critical_length_m <= 0:
        return {"Vu_kN": 0.0, "critical_length_mm": 0.0}
        
    footing_L_m = mm_to_m(footing_dim_mm)
    strip_W_m = mm_to_m(strip_width_mm)
    
    slope = (q_u_max_kPa - q_u_min_kPa) / footing_L_m if footing_L_m > 0 else 0
    q_at_d = q_u_max_kPa - slope * critical_length_m
    
    # Average pressure over critical length
    q_avg = (q_u_max_kPa + q_at_d) / 2.0
    Vu = q_avg * critical_length_m * strip_W_m
    
    return {"Vu_kN": Vu, "critical_length_mm": critical_length_m * 1000}

def one_way_shear_capacity(
    b_mm: float,
    d_mm: float,
    fc_MPa: float,
    lambda_c: float = 1.0,
    phi: float = 0.75,
) -> Dict[str, float]:
    """ACI 318-25: lambda_s = 1.0 for members bearing directly on soil."""
    # vc = 0.17 * lambda * sqrt(fc')
    vc_MPa = 0.17 * lambda_c * math.sqrt(fc_MPa)
    Vc_N = vc_MPa * b_mm * d_mm
    
    return {
        "Vc_kN": kN_from_N(Vc_N),
        "phiVc_kN": kN_from_N(phi * Vc_N),
        "vc_MPa": vc_MPa,
        "lambda_s": 1.0
    }

def two_way_shear_demand(
    Pu_kN: float,
    q_u_avg_kPa: float,
    loaded_bx_mm: float,
    loaded_by_mm: float,
    d_mm: float,
) -> Dict[str, float]:
    """Demand at d/2 from face."""
    bo_x = loaded_bx_mm + d_mm
    bo_y = loaded_by_mm + d_mm
    
    # Area inside critical perimeter
    area_inside_m2 = mm_to_m(bo_x) * mm_to_m(bo_y)
    
    # Net upward force causing punching
    Vu = Pu_kN - (q_u_avg_kPa * area_inside_m2)
    
    # Perimeter
    bo_mm = 2 * (bo_x + bo_y)
    
    return {
        "Vu_kN": max(0.0, Vu),
        "bo_mm": bo_mm,
        "area_inside_m2": area_inside_m2
    }

def two_way_shear_capacity(
    bo_mm: float,
    d_mm: float,
    fc_MPa: float,
    lambda_c: float = 1.0,
    phi: float = 0.75,
    loaded_bx_mm: float = 400.0,
    loaded_by_mm: float = 400.0,
    column_location: str = "Interior",
) -> Dict[str, float]:
    """ACI 318-25 punching shear for footings."""
    loaded_short = max(min(loaded_bx_mm, loaded_by_mm), 1e-9)
    beta = max(max(loaded_bx_mm, loaded_by_mm) / loaded_short, 1.0)
    
    loc = str(column_location or "Interior").strip().lower()
    alpha_s = 40.0 if loc.startswith("interior") else 30.0 if loc.startswith("edge") else 20.0
    
    root_fc = math.sqrt(fc_MPa)
    
    vc_a = 0.33 * lambda_c * root_fc
    vc_b = (0.17 + 0.33 / beta) * lambda_c * root_fc
    vc_c = (0.17 + 0.083 * alpha_s * d_mm / max(bo_mm, 1e-9)) * lambda_c * root_fc
    
    options = {"upper_limit": vc_a, "beta_limit": vc_b, "alpha_s_limit": vc_c}
    governing = min(options, key=options.get)
    vc_use = options[governing]
    
    Vc_N = vc_use * bo_mm * d_mm
    
    return {
        "Vc_kN": kN_from_N(Vc_N),
        "phiVc_kN": kN_from_N(phi * Vc_N),
        "beta": beta,
        "alpha_s": alpha_s,
        "vc_MPa": vc_use,
        "governing_equation": governing
    }

def bearing_capacity(
    loaded_area_mm2: float,
    footing_area_mm2: float,
    fc_MPa: float,
    phi: float = 0.65,
) -> Dict[str, float]:
    """Bearing on concrete (ACI 318-25 §22.8)."""
    area_ratio = math.sqrt(footing_area_mm2 / max(loaded_area_mm2, 1e-9))
    area_ratio = min(area_ratio, 2.0)
    
    Pn_N = 0.85 * fc_MPa * loaded_area_mm2 * area_ratio
    
    return {
        "Pn_kN": kN_from_N(Pn_N),
        "phiPn_kN": kN_from_N(phi * Pn_N),
        "area_ratio": area_ratio
    }

def dowel_interface_check(
    Pu_kN: float,
    loaded_area_mm2: float,
    dowel_bar: str,
    n_dowels: int,
    fc_MPa: float,
    fy_MPa: float,
    phi_bearing: float = 0.65,
) -> Dict[str, float]:
    """Dowel design at column-footing interface."""
    # Max pure concrete bearing (without A2/A1 enhancement for simplicity at interface)
    pure_bearing_kN = kN_from_N(phi_bearing * 0.85 * fc_MPa * loaded_area_mm2)
    
    excess_kN = max(0.0, Pu_kN - pure_bearing_kN)
    
    # As required for excess load (phi = 0.65 or 0.75 or 0.9 depending on tie condition, we use 0.65 for compression)
    phi_dowel = 0.65
    As_req_excess = (excess_kN * 1000.0) / (phi_dowel * fy_MPa)
    
    # Minimum dowels (0.005 Ag)
    As_min = 0.005 * loaded_area_mm2
    As_req = max(As_req_excess, As_min)
    
    As_prov = n_dowels * bar_area(dowel_bar)
    
    return {
        "pure_bearing_kN": pure_bearing_kN,
        "excess_kN": excess_kN,
        "As_dowel_req_mm2": As_req,
        "As_dowel_prov_mm2": As_prov,
        "As_dowel_min_mm2": As_min
    }

def development_length_tension(
    db_mm: float,
    fc_MPa: float,
    fy_MPa: float,
    lambda_c: float = 1.0,
    clear_cover_mm: float = 75.0,
    clear_spacing_mm: Optional[float] = None,
    min_ld_mm: float = 300.0,
) -> Dict[str, float]:
    """ACI-style development length estimate."""
    if clear_spacing_mm is None:
        clear_spacing_mm = db_mm * 2
        
    cb = min(clear_cover_mm + db_mm/2, clear_spacing_mm/2)
    cb = max(cb, db_mm/2)
    
    # Simplified without Ktr
    confinement = min(max(cb / db_mm, 1.5), 2.5)
    
    # psi_t=1, psi_e=1, psi_s=1 for simplicity
    ld_calc = (fy_MPa / (1.1 * lambda_c * math.sqrt(fc_MPa))) * db_mm / confinement
    ld = max(min_ld_mm, ld_calc)
    
    return {
        "ld_mm": ld,
        "ld_calc_mm": ld_calc,
        "cb_mm": cb,
        "confinement_factor": confinement
    }

# =============================================================================
# MASTER DESIGN & ORCHESTRATION
# =============================================================================

def design_all(state: DesignState, is_service: bool = False) -> Dict[str, Any]:
    """
    Run all checks for a single state. 
    Service loads check soil mechanics. Ultimate loads check RC.
    """
    mat = state.material
    geom = state.geometry
    reinf = state.reinforcement
    load = state.loadcase
    
    # 1. Geometry & properties
    foot_Lx = geom.footing_length_x_mm
    foot_Ly = geom.footing_width_y_mm
    footing_area_mm2 = foot_Lx * foot_Ly
    loaded_bx, loaded_by = get_loaded_area_dims(geom)
    loaded_area_mm2 = loaded_bx * loaded_by
    
    d_x, d_y = effective_depths(geom, reinf)
    state.effective_depth_x_mm = d_x
    state.effective_depth_y_mm = d_y
    d_avg = (d_x + d_y) / 2.0
    
    # 2. Gravity loads
    self_weight = self_weight_kN(geom, mat)
    overburden = overburden_weight_kN(geom)
    state.self_weight_kN = self_weight
    state.overburden_weight_kN = overburden
    
    total_axial_kN = load.Pu_kN + self_weight + overburden
    state.total_axial_kN = total_axial_kN
    
    results = {}
    checks = []
    
    # ---------------------------------------------------------
    # SERVICE DESIGN (Geotechnical)
    # ---------------------------------------------------------
    if is_service:
        # Soil Bearing
        soil_res = bearing_pressure_distribution(
            total_axial_kN, load.Mux_kNm, load.Muy_kNm, foot_Lx, foot_Ly
        )
        results["soil_pressure"] = soil_res
        
        q_max = soil_res["q_max_kPa"]
        q_ratio = q_max / max(geom.allowable_bearing_kPa, 1e-9)
        checks.append(CheckResult("Soil Bearing Pressure", q_max, geom.allowable_bearing_kPa, q_ratio, "kPa", status_from_ratio(q_ratio)))
        
        # Overturning X and Y
        ot_x = overturning_check(total_axial_kN, load.Mux_kNm, foot_Ly, load.Vuy_kN, geom.footing_embedment_depth_mm)
        ot_y = overturning_check(total_axial_kN, load.Muy_kNm, foot_Lx, load.Vux_kN, geom.footing_embedment_depth_mm)
        results["overturning_x"] = ot_x
        results["overturning_y"] = ot_y
        
        checks.append(CheckResult("Overturning Stability X", ot_x["M_overturn_kNm"], ot_x["M_resist_kNm"], 1/max(ot_x["fs_overturning"], 1e-9), "kN-m", ot_x["status"]))
        checks.append(CheckResult("Overturning Stability Y", ot_y["M_overturn_kNm"], ot_y["M_resist_kNm"], 1/max(ot_y["fs_overturning"], 1e-9), "kN-m", ot_y["status"]))
        
        # Sliding X and Y
        sl_x = sliding_check(total_axial_kN, load.Vux_kN, geom.friction_coefficient, geom.passive_pressure_kPa, geom.footing_embedment_depth_mm, foot_Ly)
        sl_y = sliding_check(total_axial_kN, load.Vuy_kN, geom.friction_coefficient, geom.passive_pressure_kPa, geom.footing_embedment_depth_mm, foot_Lx)
        results["sliding_x"] = sl_x
        results["sliding_y"] = sl_y
        
        checks.append(CheckResult("Sliding Stability X", sl_x["V_demand_kN"], sl_x["F_resist_kN"], 1/max(sl_x["fs_sliding"], 1e-9), "kN", sl_x["status"]))
        checks.append(CheckResult("Sliding Stability Y", sl_y["V_demand_kN"], sl_y["F_resist_kN"], 1/max(sl_y["fs_sliding"], 1e-9), "kN", sl_y["status"]))
        
    # ---------------------------------------------------------
    # ULTIMATE DESIGN (RC Structural)
    # ---------------------------------------------------------
    else:
        # Ultimate soil pressure distribution
        soil_res = bearing_pressure_distribution(
            total_axial_kN, load.Mux_kNm, load.Muy_kNm, foot_Lx, foot_Ly
        )
        q_u_max = soil_res["q_max_kPa"]
        q_u_min = soil_res["q_min_kPa"]
        q_u_avg = soil_res["q_avg_kPa"]
        
        # 1. Flexural Design
        # X-direction bars (bending about Y axis)
        mom_dem_x = flexural_demand(q_u_max, q_u_min, foot_Lx, loaded_bx, foot_Ly)
        flex_x = flexural_As_required(mom_dem_x["Mu_kNm"], foot_Ly, d_x, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)
        spac_x = provided_spacing_As(reinf.main_bar_x, reinf.spacing_x_mm, foot_Ly)
        cap_x = flexural_capacity(spac_x["As_prov_mm2"], foot_Ly, d_x, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)
        
        ratio_flex_x = mom_dem_x["Mu_kNm"] / max(cap_x["phiMn_kNm"], 1e-9)
        checks.append(CheckResult("Flexure - Bottom X Bars", mom_dem_x["Mu_kNm"], cap_x["phiMn_kNm"], ratio_flex_x, "kN-m", status_from_ratio(ratio_flex_x)))
        
        # Y-direction bars (bending about X axis)
        mom_dem_y = flexural_demand(q_u_max, q_u_min, foot_Ly, loaded_by, foot_Lx)
        flex_y = flexural_As_required(mom_dem_y["Mu_kNm"], foot_Lx, d_y, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)
        spac_y = provided_spacing_As(reinf.main_bar_y, reinf.spacing_y_mm, foot_Lx)
        cap_y = flexural_capacity(spac_y["As_prov_mm2"], foot_Lx, d_y, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)
        
        ratio_flex_y = mom_dem_y["Mu_kNm"] / max(cap_y["phiMn_kNm"], 1e-9)
        checks.append(CheckResult("Flexure - Bottom Y Bars", mom_dem_y["Mu_kNm"], cap_y["phiMn_kNm"], ratio_flex_y, "kN-m", status_from_ratio(ratio_flex_y)))
        
        results["moment_demand_x"] = mom_dem_x
        results["moment_demand_y"] = mom_dem_y
        results["flex_x"] = flex_x
        results["flex_y"] = flex_y
        results["spacing_x"] = spac_x
        results["spacing_y"] = spac_y
        results["cap_xbars"] = cap_x
        results["cap_ybars"] = cap_y
        
        # 2. One-Way Shear
        # Section normal to X axis (resisting shear along X)
        v_dem_x = one_way_shear_demand(q_u_max, q_u_min, foot_Lx, loaded_bx, d_x, foot_Ly)
        v_cap_x = one_way_shear_capacity(foot_Ly, d_x, mat.fc_MPa, mat.lambda_c, mat.phi_shear)
        ratio_vx = v_dem_x["Vu_kN"] / max(v_cap_x["phiVc_kN"], 1e-9)
        checks.append(CheckResult("One-Way Shear X", v_dem_x["Vu_kN"], v_cap_x["phiVc_kN"], ratio_vx, "kN", status_from_ratio(ratio_vx)))
        results["one_way_x"] = {**v_dem_x, **v_cap_x}
        
        # Section normal to Y axis
        v_dem_y = one_way_shear_demand(q_u_max, q_u_min, foot_Ly, loaded_by, d_y, foot_Lx)
        v_cap_y = one_way_shear_capacity(foot_Lx, d_y, mat.fc_MPa, mat.lambda_c, mat.phi_shear)
        ratio_vy = v_dem_y["Vu_kN"] / max(v_cap_y["phiVc_kN"], 1e-9)
        checks.append(CheckResult("One-Way Shear Y", v_dem_y["Vu_kN"], v_cap_y["phiVc_kN"], ratio_vy, "kN", status_from_ratio(ratio_vy)))
        results["one_way_y"] = {**v_dem_y, **v_cap_y}
        
        # 3. Two-Way Punching Shear
        punch_dem = two_way_shear_demand(load.Pu_kN, q_u_avg, loaded_bx, loaded_by, d_avg)
        punch_cap = two_way_shear_capacity(punch_dem["bo_mm"], d_avg, mat.fc_MPa, mat.lambda_c, mat.phi_shear, loaded_bx, loaded_by, geom.column_location)
        ratio_punch = punch_dem["Vu_kN"] / max(punch_cap["phiVc_kN"], 1e-9)
        checks.append(CheckResult("Two-Way Punching Shear", punch_dem["Vu_kN"], punch_cap["phiVc_kN"], ratio_punch, "kN", status_from_ratio(ratio_punch)))
        results["punch"] = {**punch_dem, **punch_cap}
        
        # 4. Bearing
        bearing = bearing_capacity(loaded_area_mm2, footing_area_mm2, mat.fc_MPa, mat.phi_bearing)
        ratio_bearing = load.Pu_kN / max(bearing["phiPn_kN"], 1e-9)
        checks.append(CheckResult("Concrete Bearing", load.Pu_kN, bearing["phiPn_kN"], ratio_bearing, "kN", status_from_ratio(ratio_bearing)))
        results["bearing"] = bearing
        
        # 5. Dowel Interface
        dowels = dowel_interface_check(load.Pu_kN, loaded_area_mm2, reinf.dowel_bar, reinf.dowel_count, mat.fc_MPa, mat.fy_MPa, mat.phi_bearing)
        ratio_dowel = dowels["As_dowel_req_mm2"] / max(dowels["As_dowel_prov_mm2"], 1e-9)
        checks.append(CheckResult("Dowel Area", dowels["As_dowel_req_mm2"], dowels["As_dowel_prov_mm2"], ratio_dowel, "mm2", status_from_ratio(ratio_dowel)))
        results["dowels"] = dowels
        
        # 6. Development Length
        # X bars
        spacing_clear_x = reinf.spacing_x_mm - bar_diameter(reinf.main_bar_x)
        ld_x = development_length_tension(bar_diameter(reinf.main_bar_x), mat.fc_MPa, mat.fy_MPa, mat.lambda_c, geom.side_cover_mm, spacing_clear_x)
        l_avail_x = (foot_Lx - loaded_bx)/2.0 - geom.side_cover_mm
        ratio_ldx = ld_x["ld_mm"] / max(l_avail_x, 1e-9)
        checks.append(CheckResult("Development Length X Bars", ld_x["ld_mm"], l_avail_x, ratio_ldx, "mm", status_from_ratio(ratio_ldx)))
        results["ld_x"] = ld_x
        
        # Y bars
        spacing_clear_y = reinf.spacing_y_mm - bar_diameter(reinf.main_bar_y)
        ld_y = development_length_tension(bar_diameter(reinf.main_bar_y), mat.fc_MPa, mat.fy_MPa, mat.lambda_c, geom.side_cover_mm, spacing_clear_y)
        l_avail_y = (foot_Ly - loaded_by)/2.0 - geom.side_cover_mm
        ratio_ldy = ld_y["ld_mm"] / max(l_avail_y, 1e-9)
        checks.append(CheckResult("Development Length Y Bars", ld_y["ld_mm"], l_avail_y, ratio_ldy, "mm", status_from_ratio(ratio_ldy)))
        results["ld_y"] = ld_y

    results["checks"] = checks
    return results

def provided_spacing_As(bar: str, spacing_mm: float, strip_width_mm: float) -> Dict[str, float]:
    spacing_use = max(safe_float(spacing_mm, 150.0), 1.0)
    n_bars = int(math.floor(strip_width_mm / spacing_use)) + 1
    As_prov = n_bars * bar_area(bar)
    return {
        "spacing_req_mm": spacing_use,
        "spacing_use_mm": spacing_use,
        "n_bars": n_bars,
        "As_prov_mm2": As_prov,
    }

# =============================================================================
# ENVELOPE BATCH
# =============================================================================

def envelope_group_design(
    group_name: str,
    service_cases: List[LoadCase],
    ultimate_cases: List[LoadCase],
    material: Material,
    geometry: FootingGeometry,
    reinforcement: Reinforcement,
) -> Dict[str, Any]:
    """
    Run all service + ultimate load cases and envelope to governing values.
    """
    # 1. Run service cases
    service_runs = []
    for lc in service_cases:
        state = DesignState(material, geometry, reinforcement, lc)
        res = design_all(state, is_service=True)
        service_runs.append((lc, state, res))
        
    # 2. Run ultimate cases
    ultimate_runs = []
    for lc in ultimate_cases:
        state = DesignState(material, geometry, reinforcement, lc)
        res = design_all(state, is_service=False)
        ultimate_runs.append((lc, state, res))
        
    # Mocking envelope aggregation for now
    # We would pick the max D/C ratio for each check type
    
    if ultimate_runs:
        gov_ult_lc, gov_ult_state, gov_ult_res = max(ultimate_runs, key=lambda x: max(c.ratio for c in x[2]["checks"]))
    else:
        gov_ult_lc = LoadCase()
        gov_ult_state = DesignState(material, geometry, reinforcement, gov_ult_lc)
        gov_ult_res = design_all(gov_ult_state, is_service=False)
        
    if service_runs:
        gov_srv_lc, gov_srv_state, gov_srv_res = max(service_runs, key=lambda x: max(c.ratio for c in x[2]["checks"]))
    else:
        gov_srv_lc = LoadCase(case_type="Service")
        gov_srv_state = DesignState(material, geometry, reinforcement, gov_srv_lc)
        gov_srv_res = design_all(gov_srv_state, is_service=True)

    # Combine checks
    combined_checks = gov_srv_res.get("checks", []) + gov_ult_res.get("checks", [])
    
    display_results = {**gov_ult_res, "service_soil": gov_srv_res.get("soil_pressure", {})}
    display_results["checks"] = combined_checks
    display_results["governing_combos"] = {
        "Service Checks": gov_srv_lc.name,
        "Structural Checks": gov_ult_lc.name
    }
    
    return {
        "group": group_name,
        "state": gov_ult_state,          # governing ultimate state (for RC design)
        "srv_state": gov_srv_state,       # governing service state (for soil pressure calcs)
        "results": display_results,
        "service_runs": service_runs,
        "ultimate_runs": ultimate_runs
    }

def normalize_manual_service_load_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cmap = {}
    for col in df.columns:
        c = str(col).lower().replace(" ", "").replace("_", "")
        if "group" in c: cmap[col] = "Group Name"
        elif "dps" in c or "deadps" in c: cmap[col] = "D Ps"
        elif "dmsx" in c or "deadmsx" in c: cmap[col] = "D Msx"
        elif "dmsy" in c or "deadmsy" in c: cmap[col] = "D Msy"
        elif "lps" in c or "liveps" in c: cmap[col] = "L Ps"
        elif "lmsx" in c or "livemsx" in c: cmap[col] = "L Msx"
        elif "lmsy" in c or "livemsy" in c: cmap[col] = "L Msy"
        elif c == "dfactor" or "deadfactor" in c: cmap[col] = "D Factor"
        elif c == "lfactor" or "livefactor" in c: cmap[col] = "L Factor"
        elif "footinglx" in c: cmap[col] = "Footing Lx"
        elif "footingly" in c: cmap[col] = "Footing Ly"
        elif "thickness" in c: cmap[col] = "Thickness"
        elif "jointid" in c: cmap[col] = "Joint IDs"
    return df.rename(columns=cmap)

def default_manual_foundations(geom: FootingGeometry) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Group Name": "F1", "D Ps": 800, "D Msx": 20, "D Msy": 0, "L Ps": 400, "L Msx": 10, "L Msy": 0,
            "D Factor": 1.2, "L Factor": 1.6, "Footing Lx": geom.footing_length_x_mm, "Footing Ly": geom.footing_width_y_mm, "Thickness": geom.footing_thickness_mm
        }
    ])

def default_sap_groups(df: pd.DataFrame, geom: FootingGeometry) -> pd.DataFrame:
    groups = []
    if "Joint" in df.columns:
        joints = sorted(df["Joint"].unique())
        for i, j in enumerate(joints):
            groups.append({
                "Group Name": f"F{i+1}", "Joint IDs": str(j), "Footing Lx": geom.footing_length_x_mm, "Footing Ly": geom.footing_width_y_mm, "Thickness": geom.footing_thickness_mm
            })
    if not groups:
        groups.append({
            "Group Name": "F1", "Joint IDs": "1, 2", "Footing Lx": geom.footing_length_x_mm, "Footing Ly": geom.footing_width_y_mm, "Thickness": geom.footing_thickness_mm
        })
    return pd.DataFrame(groups)

def parse_joint_list(text: str) -> List[str]:
    return [j.strip() for j in str(text).replace(";", ",").split(",") if j.strip()]

def design_batch_from_manual_table(
    groups_df: pd.DataFrame,
    material: Material,
    geometry: FootingGeometry,
    reinforcement: Reinforcement,
) -> Dict[str, Any]:
    items = []
    df = normalize_manual_service_load_columns(groups_df)
    for i, row in df.iterrows():
        name = str(row.get("Group Name", f"F{i+1}"))
        
        # Build Service Case (D+L)
        p_d = safe_float(row.get("D Ps", 0.0))
        p_l = safe_float(row.get("L Ps", 0.0))
        mx_d = safe_float(row.get("D Msx", 0.0))
        mx_l = safe_float(row.get("L Msx", 0.0))
        my_d = safe_float(row.get("D Msy", 0.0))
        my_l = safe_float(row.get("L Msy", 0.0))
        
        lc_serv = LoadCase("Service", "Service", p_d+p_l, mx_d+mx_l, my_d+my_l, 0, 0)
        
        # Build Ultimate Case
        fd = safe_float(row.get("D Factor", 1.2))
        fl = safe_float(row.get("L Factor", 1.6))
        
        lc_ult = LoadCase(f"{fd}D + {fl}L", "Ultimate", p_d*fd + p_l*fl, mx_d*fd + mx_l*fl, my_d*fd + my_l*fl, 0, 0)
        
        # Geometry override
        geom_run = replace(geometry)
        if "Footing Lx" in row: geom_run.footing_length_x_mm = safe_float(row["Footing Lx"])
        if "Footing Ly" in row: geom_run.footing_width_y_mm = safe_float(row["Footing Ly"])
        if "Thickness" in row: geom_run.footing_thickness_mm = safe_float(row["Thickness"])
        
        item = envelope_group_design(name, [lc_serv], [lc_ult], material, geom_run, reinforcement)
        items.append(item)
        
    return {"items": items}

def build_loadcase_from_sap_row(
    row: pd.Series, 
    ctype: str, 
    v_sign: float, 
    m_sign: float,
    v_col: str = "F3",
    mx_col: str = "M1",
    my_col: str = "M2",
    vx_col: str = "F1",
    vy_col: str = "F2"
) -> LoadCase:
    name = f"{row.get('Joint', 'J')}-{row.get('OutputCase', 'LC')}"
    return LoadCase(
        name=name,
        case_type=ctype,
        Pu_kN=safe_float(row.get(v_col, 0.0)) * v_sign,
        Mux_kNm=safe_float(row.get(mx_col, 0.0)) * m_sign,
        Muy_kNm=safe_float(row.get(my_col, 0.0)) * m_sign,
        Vux_kN=safe_float(row.get(vx_col, 0.0)),
        Vuy_kN=safe_float(row.get(vy_col, 0.0))
    )

def design_batch_from_sap_table(
    sap_df: pd.DataFrame,
    groups_df: pd.DataFrame,
    material: Material,
    geometry: FootingGeometry,
    reinforcement: Reinforcement,
    service_cases: List[str] = None,
    ultimate_cases: List[str] = None,
    v_sign: float = 1.0,
    m_sign: float = 1.0,
    cmap: Dict[str, str] = None,
) -> Dict[str, Any]:
    
    if cmap is None:
        if sap_df is not None and "F3 (kN)" in sap_df.columns:
            cmap = {"v_col": "F3 (kN)", "mx_col": "M1 (kN-m)", "my_col": "M2 (kN-m)", "vx_col": "F1 (kN)", "vy_col": "F2 (kN)"}
        else:
            cmap = {"v_col": "F3", "mx_col": "M1", "my_col": "M2", "vx_col": "F1", "vy_col": "F2"}
        
    items = []
    groups_df = normalize_manual_service_load_columns(groups_df)
    
    for i, group_row in groups_df.iterrows():
        name = str(group_row.get("Group Name", f"F{i+1}"))
        joints = parse_joint_list(group_row.get("Joint IDs", ""))
        if not joints:
            continue
            
        geom_run = replace(geometry)
        if "Footing Lx" in group_row: geom_run.footing_length_x_mm = safe_float(group_row["Footing Lx"])
        if "Footing Ly" in group_row: geom_run.footing_width_y_mm = safe_float(group_row["Footing Ly"])
        if "Thickness" in group_row: geom_run.footing_thickness_mm = safe_float(group_row["Thickness"])
        
        # Filter sap table for these joints
        mask_joints = sap_df["Joint"].astype(str).isin(joints)
        
        # Hardcode CS-xx = Service, CU-xx = Ultimate (per project convention)
        # Only rows whose OutputCase starts with 'CS-' are used for service sizing.
        # Only rows whose OutputCase starts with 'CU-' are used for RC strength design.
        all_cases = sap_df["OutputCase"].astype(str)
        mask_cs = all_cases.str.upper().str.startswith("CS-")
        mask_cu = all_cases.str.upper().str.startswith("CU-")
        
        # Build service cases (CS-xx only)
        srv_cases = []
        if service_cases:
            # Allow caller override (for manual table mode)
            mask_srv = mask_joints & all_cases.isin(service_cases)
        else:
            mask_srv = mask_joints & mask_cs
            
        for _, r in sap_df[mask_srv].iterrows():
            srv_cases.append(build_loadcase_from_sap_row(r, "Service", v_sign, m_sign, **cmap))
            
        # Build ultimate cases (CU-xx only)
        ult_cases = []
        if ultimate_cases:
            # Allow caller override (for manual table mode)
            mask_ult = mask_joints & all_cases.isin(ultimate_cases)
        else:
            mask_ult = mask_joints & mask_cu
            
        for _, r in sap_df[mask_ult].iterrows():
            ult_cases.append(build_loadcase_from_sap_row(r, "Ultimate", v_sign, m_sign, **cmap))
            
        if not srv_cases and not ult_cases:
            continue
            
        if not srv_cases: srv_cases = [LoadCase("DummySrv", "Service")]
        if not ult_cases: ult_cases = [LoadCase("DummyUlt", "Ultimate")]
            
        item = envelope_group_design(name, srv_cases, ult_cases, material, geom_run, reinforcement)
        items.append(item)
        
    return {"items": items}

# =============================================================================
# MATERIAL TAKEOFF
# =============================================================================

def calculate_material_takeoff(state: DesignState, results: Dict) -> Dict[str, float]:
    geom = state.geometry
    reinf = state.reinforcement
    
    # Concrete Volume
    vol_m3 = (geom.footing_length_x_mm/1000) * (geom.footing_width_y_mm/1000) * (geom.footing_thickness_mm/1000)
    
    # Rebar weight
    # Bottom X
    spac_x = results.get("spacing_x", {})
    n_x = spac_x.get("n_bars", 0)
    len_x_m = (geom.footing_length_x_mm - 2 * geom.side_cover_mm + 2 * reinf.hook_extension_mm) / 1000
    wt_x = n_x * len_x_m * bar_area(reinf.main_bar_x) * 7850 / 1e6
    
    # Bottom Y
    spac_y = results.get("spacing_y", {})
    n_y = spac_y.get("n_bars", 0)
    len_y_m = (geom.footing_width_y_mm - 2 * geom.side_cover_mm + 2 * reinf.hook_extension_mm) / 1000
    wt_y = n_y * len_y_m * bar_area(reinf.main_bar_y) * 7850 / 1e6
    
    # Dowels
    wt_dowels = reinf.dowel_count * 2.0 * bar_area(reinf.dowel_bar) * 7850 / 1e6 # Approx 2m long
    
    total_rebar_kg = wt_x + wt_y + wt_dowels
    ratio_kg_m3 = total_rebar_kg / max(vol_m3, 1e-9)
    
    return {
        "concrete_vol_m3": vol_m3,
        "rebar_kg": total_rebar_kg,
        "rebar_ratio_kg_m3": ratio_kg_m3
    }

# =============================================================================
# DRAWING
# =============================================================================

def plot_plan(state: DesignState, results: Dict) -> plt.Figure:
    """Engineering plan view with proper dimension lines, rebar grid, and callouts."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')
    geom = state.geometry
    reinf = state.reinforcement
    Lx = geom.footing_length_x_mm
    Ly = geom.footing_width_y_mm
    bx, by = get_loaded_area_dims(geom)
    d_avg = (state.effective_depth_x_mm + state.effective_depth_y_mm) / 2
    spac_x = reinf.spacing_x_mm
    spac_y = reinf.spacing_y_mm
    cover = geom.side_cover_mm
    margin = 600

    # ---- Footing outline ----
    rect = Rectangle((-Lx/2, -Ly/2), Lx, Ly, fill=True, facecolor='#f0f0f0', edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # ---- Rebar grid (X-direction bars — horizontal lines) ----
    n_y_bars = max(1, int((Ly - 2*cover) / spac_y))
    y_bar_positions = np.linspace(-Ly/2 + cover, Ly/2 - cover, n_y_bars)
    for yb in y_bar_positions:
        ax.plot([-Lx/2 + cover, Lx/2 - cover], [yb, yb],
                color='#1565C0', linewidth=1.0, alpha=0.7, zorder=2)

    # ---- Rebar grid (Y-direction bars — vertical lines) ----
    n_x_bars = max(1, int((Lx - 2*cover) / spac_x))
    x_bar_positions = np.linspace(-Lx/2 + cover, Lx/2 - cover, n_x_bars)
    for xb in x_bar_positions:
        ax.plot([xb, xb], [-Ly/2 + cover, Ly/2 - cover],
                color='#C62828', linewidth=1.0, alpha=0.7, zorder=2)

    # ---- Column (filled hatch) ----
    col = Rectangle((-bx/2, -by/2), bx, by, fill=True, facecolor='#9E9E9E',
                    edgecolor='black', linewidth=2, hatch='//', zorder=5)
    ax.add_patch(col)
    ax.text(0, 0, 'COL', ha='center', va='center', fontsize=8,
            fontweight='bold', color='white', zorder=6)

    # ---- Punching perimeter ----
    bo_x, bo_y = bx + d_avg, by + d_avg
    punch = Rectangle((-bo_x/2, -bo_y/2), bo_x, bo_y,
                      fill=False, edgecolor='red', linewidth=1.5,
                      linestyle='--', zorder=4)
    ax.add_patch(punch)
    ax.text(bo_x/2 + 60, -bo_y/2, 'Critical\nPerimeter', fontsize=7,
            color='red', va='center', zorder=6)

    # ---- DIMENSION LINE: Bottom (Lx) ----
    dim_y = -Ly/2 - 300
    ext_y0, ext_y1 = -Ly/2, dim_y - 30
    ax.plot([-Lx/2, -Lx/2], [ext_y0, ext_y1], 'k-', linewidth=0.8)
    ax.plot([Lx/2, Lx/2], [ext_y0, ext_y1], 'k-', linewidth=0.8)
    ax.annotate('', xy=(Lx/2, dim_y), xytext=(-Lx/2, dim_y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(0, dim_y - 80, f'Lx = {Lx:,.0f} mm', ha='center', va='top',
            fontsize=10, fontweight='bold')

    # ---- DIMENSION LINE: Left (Ly) ----
    dim_x = -Lx/2 - 300
    ext_x0, ext_x1 = -Lx/2, dim_x - 30
    ax.plot([ext_x0, ext_x1], [-Ly/2, -Ly/2], 'k-', linewidth=0.8)
    ax.plot([ext_x0, ext_x1], [Ly/2, Ly/2], 'k-', linewidth=0.8)
    ax.annotate('', xy=(dim_x, Ly/2), xytext=(dim_x, -Ly/2),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(dim_x - 80, 0, f'Ly = {Ly:,.0f} mm', ha='right', va='center',
            fontsize=10, fontweight='bold', rotation=90)

    # ---- Rebar callout box (top right) ----
    info = (f"Bottom Reinf.\n"
            f"X-bars: {reinf.main_bar_x} @ {spac_x:.0f} mm ({n_x_bars} pcs)\n"
            f"Y-bars: {reinf.main_bar_y} @ {spac_y:.0f} mm ({n_y_bars} pcs)\n"
            f"Cover = {cover:.0f} mm")
    ax.text(Lx/2 + 60, Ly/2, info, ha='left', va='top', fontsize=8,
            bbox=dict(facecolor='#FFFDE7', edgecolor='#F57F17', linewidth=1),
            linespacing=1.5, zorder=10)

    # Leader line from callout to top-right bar
    ax.annotate('', xy=(Lx/2 - cover, Ly/2 - cover),
                xytext=(Lx/2 + 60, Ly/2 - 60),
                arrowprops=dict(arrowstyle='->', color='#F57F17', lw=1.2))

    ax.set_aspect('equal')
    ax.set_xlim(-Lx/2 - margin - 200, Lx/2 + margin + 200)
    ax.set_ylim(-Ly/2 - margin - 200, Ly/2 + margin + 100)
    ax.axis('off')
    ax.set_title('Plan View — Footing Reinforcement', fontsize=13,
                 fontweight='bold', pad=12)
    plt.tight_layout()
    return fig

def plot_section(state: DesignState, results: Dict, direction: str = "X") -> plt.Figure:
    """Engineering section view with proper dimension lines, extension lines, and rebar callouts."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_facecolor('white')
    geom = state.geometry
    reinf = state.reinforcement
    h = geom.footing_thickness_mm
    cover_bot = geom.bottom_cover_mm
    cover_side = geom.side_cover_mm

    if direction == "X":
        L = geom.footing_length_x_mm
        bcol = get_loaded_area_dims(geom)[0]
        bar_main = reinf.main_bar_x   # runs in X — seen as a line
        bar_sec  = reinf.main_bar_y   # runs in Y — seen as dots
        spac_main = reinf.spacing_x_mm
        spac_sec  = reinf.spacing_y_mm
        d = state.effective_depth_x_mm
        dir_label = "X-X"
    else:
        L = geom.footing_width_y_mm
        bcol = get_loaded_area_dims(geom)[1]
        bar_main = reinf.main_bar_y
        bar_sec  = reinf.main_bar_x
        spac_main = reinf.spacing_y_mm
        spac_sec  = reinf.spacing_x_mm
        d = state.effective_depth_y_mm
        dir_label = "Y-Y"

    db_main = bar_diameter(bar_main)   # mm
    db_sec  = bar_diameter(bar_sec)    # mm

    # Y-positions of rebar layers (bottom of footing = y=0)
    y_bot_main = cover_bot + db_main / 2.0        # outermost layer (runs in X)
    y_bot_sec  = cover_bot + db_main + db_sec / 2.0  # inner layer (runs in Y)

    # ---- Footing body (concrete hatch) ----
    rect = Rectangle((-L/2, 0), L, h,
                     fill=True, facecolor='#ECEFF1', edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # ---- Column stub ----
    col_h = 450
    col = Rectangle((-bcol/2, h), bcol, col_h,
                    fill=True, facecolor='#9E9E9E', edgecolor='black', linewidth=2, hatch='//')
    ax.add_patch(col)
    ax.text(0, h + col_h/2, 'COLUMN', ha='center', va='center',
            fontsize=7, color='white', fontweight='bold')

    # ---- Ground / GL line ----
    ax.plot([-L/2 - 200, L/2 + 200], [h, h],
            color='#795548', linestyle='--', linewidth=1.2)
    ax.text(L/2 + 220, h, 'G.L.', va='center', fontsize=8, color='#795548')

    # ---- Effective depth arrow (inside footing) ----
    ax.annotate('', xy=(L/2 - 200, h - d), xytext=(L/2 - 200, h),
                arrowprops=dict(arrowstyle='<->', color='#37474F', lw=1.2))
    ax.text(L/2 - 250, h - d/2, f'd={d:.0f}', va='center', ha='right',
            fontsize=8, color='#37474F')

    # ---- Main bars: continuous line (runs along length) ----
    x0_bar = -L/2 + cover_side
    x1_bar =  L/2 - cover_side
    ax.plot([x0_bar, x1_bar], [y_bot_main, y_bot_main],
            color='#1565C0', linewidth=3, solid_capstyle='round', zorder=4)

    # ---- Secondary bars: circles (cross-section dots) ----
    n_sec = max(2, int((L - 2*cover_side) / max(spac_sec, 50)) + 1)
    x_sec_bars = np.linspace(x0_bar, x1_bar, n_sec)
    r_sec = db_sec / 2.0
    for xb in x_sec_bars:
        circ = Circle((xb, y_bot_sec), r_sec,
                      facecolor='#C62828', edgecolor='black', linewidth=0.8, zorder=5)
        ax.add_patch(circ)

    # ---- Cover annotation (bottom) ----
    ax.annotate('', xy=(x0_bar - 30, 0), xytext=(x0_bar - 30, y_bot_main),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=0.9))
    ax.text(x0_bar - 80, y_bot_main / 2, f'c={cover_bot:.0f}',
            ha='right', va='center', fontsize=8, color='#555')

    # ---- Rebar leader: main bars ----
    lx_main = 0
    ax.annotate(
        f'{bar_main} @ {spac_main:.0f} mm\n(bottom layer)',
        xy=(lx_main, y_bot_main),
        xytext=(lx_main, y_bot_main + h * 0.45),
        fontsize=9, ha='center', color='#1565C0', fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='#1565C0', linewidth=0.8, pad=3),
        arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.2))

    # ---- Rebar leader: secondary bars ----
    lx_sec = L/4
    ax.annotate(
        f'{bar_sec} @ {spac_sec:.0f} mm',
        xy=(lx_sec, y_bot_sec),
        xytext=(lx_sec, y_bot_sec + h * 0.38),
        fontsize=9, ha='center', color='#C62828', fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='#C62828', linewidth=0.8, pad=3),
        arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.2))

    # ---- DIMENSION: Total length (below footing) ----
    dim_y_len = -260
    ext_drop  = -30
    ax.plot([-L/2, -L/2], [0, dim_y_len + ext_drop], 'k-', linewidth=0.7)
    ax.plot([ L/2,  L/2], [0, dim_y_len + ext_drop], 'k-', linewidth=0.7)
    ax.annotate('', xy=(L/2, dim_y_len), xytext=(-L/2, dim_y_len),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(0, dim_y_len - 60, f'L = {L:,.0f} mm',
            ha='center', va='top', fontsize=10, fontweight='bold')

    # ---- DIMENSION: Footing depth (right side) ----
    dim_x_h = L/2 + 350
    ax.plot([L/2, dim_x_h + 30], [0, 0], 'k-', linewidth=0.7)
    ax.plot([L/2, dim_x_h + 30], [h, h], 'k-', linewidth=0.7)
    ax.annotate('', xy=(dim_x_h, h), xytext=(dim_x_h, 0),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(dim_x_h + 60, h/2, f'h = {h:,.0f} mm',
            ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlim(-L/2 - 500, L/2 + 700)
    ax.set_ylim(-400, h + 600)
    ax.axis('off')
    ax.set_title(f'Section {dir_label} — Footing Reinforcement Detail',
                 fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig



