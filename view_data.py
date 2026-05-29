"""Data formatter and table builder for the spread footing app.
Converts footing_engine results into pandas DataFrames for the UI.
"""
from typing import Any, Dict, List
import pandas as pd
import footing_engine as eng

def metric_summary(state: eng.DesignState, results: Dict[str, Any]) -> Dict[str, str]:
    """Generates a high-level summary of the governing results."""
    checks = results.get("checks", [])
    if not checks:
        return {}

    max_ratio = max([c.ratio for c in checks] + [0.0])
    
    # Check if there are any STM or near-failures
    status_set = {c.status for c in checks}
    if "FAIL" in status_set or max_ratio > 1.0:
        overall_status = "FAIL"
    elif "NEAR" in status_set or max_ratio >= 0.95:
        overall_status = "NEAR"
    else:
        overall_status = "PASS"

    geom = state.geometry
    Lx = geom.footing_length_x_mm
    Ly = geom.footing_width_y_mm
    h = geom.footing_thickness_mm
    
    soil_res = results.get("service_soil", {})
    if not soil_res and "soil_pressure" in results:
        soil_res = results["soil_pressure"]
        
    q_max = soil_res.get("q_max_kPa", 0.0)

    return {
        "Overall max D/C": f"{max_ratio:.2f}",
        "Status": overall_status,
        "Footing size": f"{Lx:,.0f} x {Ly:,.0f} mm",
        "Thickness h": f"{h:,.0f} mm",
        "Total Axial (incl. SW)": f"{state.total_axial_kN:,.0f} kN",
        "Max Soil Pressure": f"{q_max:,.1f} kPa",
    }


def checks_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Converts the list of CheckResult objects into a DataFrame."""
    checks = results.get("checks", [])
    if not checks:
        return pd.DataFrame()

    data = []
    for c in checks:
        data.append({
            "Check": c.name,
            "Demand": c.demand,
            "Capacity": c.capacity,
            "Unit": c.unit,
            "Ratio": c.ratio,
            "Status": c.status,
            "Note": c.note
        })
    df = pd.DataFrame(data)
    if not df.empty and "Ratio" in df.columns:
        df["Ratio"] = pd.to_numeric(df["Ratio"], errors="coerce")
    return df


def flexural_summary_dataframe(state: eng.DesignState, results: Dict[str, Any]) -> pd.DataFrame:
    """Summarizes flexural reinforcement design for bottom bars."""
    rows = []
    reinf = state.reinforcement
    geom = state.geometry
    
    # Bottom X
    fx = results.get("flex_x", {})
    sx = results.get("spacing_x", {})
    cx = results.get("cap_xbars", {})
    
    rows.append({
        "Direction": "X (Bottom)",
        "Bar": reinf.main_bar_x,
        "Req As (mm²)": fx.get("As_req_mm2", 0.0),
        "Min As (mm²)": fx.get("As_min_mm2", 0.0),
        "Use spacing (mm)": sx.get("spacing_use_mm", 0.0),
        "Bars count": sx.get("n_bars", 0),
        "Provided As (mm²)": sx.get("As_prov_mm2", 0.0),
        "phiMn (kN-m)": cx.get("phiMn_kNm", 0.0),
        "D/C": results.get("moment_demand_x", {}).get("Mu_kNm", 0.0) / max(cx.get("phiMn_kNm", 0.0), 1e-9)
    })
    
    # Bottom Y
    fy = results.get("flex_y", {})
    sy = results.get("spacing_y", {})
    cy = results.get("cap_ybars", {})
    
    rows.append({
        "Direction": "Y (Bottom)",
        "Bar": reinf.main_bar_y,
        "Req As (mm²)": fy.get("As_req_mm2", 0.0),
        "Min As (mm²)": fy.get("As_min_mm2", 0.0),
        "Use spacing (mm)": sy.get("spacing_use_mm", 0.0),
        "Bars count": sy.get("n_bars", 0),
        "Provided As (mm²)": sy.get("As_prov_mm2", 0.0),
        "phiMn (kN-m)": cy.get("phiMn_kNm", 0.0),
        "D/C": results.get("moment_demand_y", {}).get("Mu_kNm", 0.0) / max(cy.get("phiMn_kNm", 0.0), 1e-9)
    })

    df = pd.DataFrame(rows)
    return df


def detail_summary_dataframe(state: eng.DesignState, results: Dict[str, Any]) -> pd.DataFrame:
    """Summarizes detailed engineering parameters and checks."""
    items = []
    
    # Material takeoff
    mto = eng.calculate_material_takeoff(state, results)
    items.append({"Item": "Concrete Volume", "Value": mto["concrete_vol_m3"], "Unit": "m³"})
    items.append({"Item": "Total Rebar Weight", "Value": mto["rebar_kg"], "Unit": "kg"})
    items.append({"Item": "Rebar Ratio", "Value": mto["rebar_ratio_kg_m3"], "Unit": "kg/m³"})
    
    # Effective depths
    items.append({"Item": "Effective depth (X bars) d_x", "Value": state.effective_depth_x_mm, "Unit": "mm"})
    items.append({"Item": "Effective depth (Y bars) d_y", "Value": state.effective_depth_y_mm, "Unit": "mm"})
    
    # Shear
    v_x = results.get("one_way_x", {})
    items.append({"Item": "One-way critical length (X)", "Value": v_x.get("critical_length_mm", 0.0), "Unit": "mm"})
    items.append({"Item": "One-way phiVc (X)", "Value": v_x.get("phiVc_kN", 0.0), "Unit": "kN"})
    
    v_y = results.get("one_way_y", {})
    items.append({"Item": "One-way critical length (Y)", "Value": v_y.get("critical_length_mm", 0.0), "Unit": "mm"})
    items.append({"Item": "One-way phiVc (Y)", "Value": v_y.get("phiVc_kN", 0.0), "Unit": "kN"})
    
    punch = results.get("punch", {})
    items.append({"Item": "Punching critical perimeter bo", "Value": punch.get("bo_mm", 0.0), "Unit": "mm"})
    items.append({"Item": "Punching phiVc", "Value": punch.get("phiVc_kN", 0.0), "Unit": "kN"})
    items.append({"Item": "Punching governing eqn", "Value": punch.get("governing_equation", ""), "Unit": ""})
    
    # Dowels
    dowels = results.get("dowels", {})
    items.append({"Item": "Pure concrete bearing phiPn", "Value": dowels.get("pure_bearing_kN", 0.0), "Unit": "kN"})
    items.append({"Item": "Dowel required area", "Value": dowels.get("As_dowel_req_mm2", 0.0), "Unit": "mm²"})
    
    # Dev Length
    ld_x = results.get("ld_x", {})
    items.append({"Item": "Development Length X", "Value": ld_x.get("ld_mm", 0.0), "Unit": "mm"})
    ld_y = results.get("ld_y", {})
    items.append({"Item": "Development Length Y", "Value": ld_y.get("ld_mm", 0.0), "Unit": "mm"})
    
    return pd.DataFrame(items)


def soil_pressure_dataframe(state: eng.DesignState, results: Dict[str, Any]) -> pd.DataFrame:
    """Summarizes soil bearing pressure distribution."""
    soil_res = results.get("soil_pressure", {})
    service_soil = results.get("service_soil", soil_res)
    
    items = []
    items.append({"Parameter": "Total Axial (incl. SW)", "Value": state.total_axial_kN, "Unit": "kN"})
    items.append({"Parameter": "Max Pressure (Service)", "Value": service_soil.get("q_max_kPa", 0.0), "Unit": "kPa"})
    items.append({"Parameter": "Min Pressure (Service)", "Value": service_soil.get("q_min_kPa", 0.0), "Unit": "kPa"})
    items.append({"Parameter": "Avg Pressure (Service)", "Value": service_soil.get("q_avg_kPa", 0.0), "Unit": "kPa"})
    items.append({"Parameter": "Eccentricity X (Service)", "Value": service_soil.get("ex_mm", 0.0), "Unit": "mm"})
    items.append({"Parameter": "Eccentricity Y (Service)", "Value": service_soil.get("ey_mm", 0.0), "Unit": "mm"})
    items.append({"Parameter": "Contact Type", "Value": service_soil.get("contact_type", ""), "Unit": ""})
    items.append({"Parameter": "Within Kern", "Value": str(service_soil.get("kern_check", False)), "Unit": ""})
    items.append({"Parameter": "Effective Area", "Value": service_soil.get("effective_area_m2", 0.0), "Unit": "m²"})
    
    return pd.DataFrame(items)


def governing_combos_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Shows which combination governed."""
    combos = results.get("governing_combos", {})
    data = [{"Item": k, "Combination": v} for k, v in combos.items()]
    return pd.DataFrame(data)

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8-sig")

def calculation_html(state: 'footing_engine.DesignState', results: Dict[str, Any],
                     srv_state: 'footing_engine.DesignState' = None) -> str:
    """
    state     = governing ULTIMATE (CU-xx) DesignState — used for RC strength sections
    srv_state = governing SERVICE (CS-xx) DesignState — used for soil pressure section
    """
    if srv_state is None:
        srv_state = state   # single-run / manual mode: same state

    geom = state.geometry
    mat = state.material
    reinf = state.reinforcement
    
    html = []
    html.append("<!DOCTYPE html><html><head>")
    html.append("<script id='MathJax-script' async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>")
    html.append("<style>")
    html.append("body { font-family: 'Segoe UI', sans-serif; font-size: 14px; color: #e2e8f0; background-color: #0f172a; margin: 20px; }")
    html.append("h2 { color: #38bdf8; border-bottom: 2px solid #38bdf8; padding-bottom: 4px; }")
    html.append("h3 { color: #7dd3fc; border-bottom: 1px solid #475569; padding-bottom: 4px; margin-top: 20px; }")
    html.append(".formula { background-color: #1e293b; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #38bdf8; text-align: center; font-size: 16px; overflow-x: auto; }")
    html.append("th, td { padding: 8px 12px; border: 1px solid #475569; text-align: left; }")
    html.append("table { border-collapse: collapse; margin-bottom: 15px; width: 100%; }")
    html.append("th { background-color: #1e293b; }")
    html.append(".pass { color: #4ade80; font-weight: bold; } .fail { color: #f87171; font-weight: bold; }")
    html.append("</style></head><body>")
    
    def n(val, decimals=2):
        try: return f"{float(val):,.{decimals}f}"
        except: return str(val)
        
    def check_result(name):
        for c in results.get("checks", []):
            if c.name == name:
                cls = "pass" if c.ratio <= 1.0 else "fail"
                return f"<b>Demand = {n(c.demand)} {c.unit}, Capacity = {n(c.capacity)} {c.unit}, D/C = {n(c.ratio, 3)} <span class='{cls}'>[{c.status}]</span></b>"
        return "<i>Check not found</i>"

    html.append(f"<h2>Spread Footing Design Calculation - Step by Step</h2>")
    
    # 1. Inputs
    html.append(f"<h3>1. Input Parameters</h3>")
    html.append(f"<ul>")
    html.append(f"<li><b>Footing size:</b> \\( L_x \\times L_y \\times h \\) = {n(geom.footing_length_x_mm)} &times; {n(geom.footing_width_y_mm)} &times; {n(geom.footing_thickness_mm)} mm</li>")
    html.append(f"<li><b>Column/Pedestal size:</b> \\( b_x \\times b_y \\) = {n(geom.column_bx_mm)} &times; {n(geom.column_by_mm)} mm ({geom.column_location})</li>")
    html.append(f"<li><b>Concrete:</b> \\( f'_c \\) = {n(mat.fc_MPa)} MPa, \\( \\gamma_c \\) = {n(mat.gamma_conc_kN_m3)} kN/m³</li>")
    html.append(f"<li><b>Steel:</b> \\( f_y \\) = {n(mat.fy_MPa)} MPa</li>")
    html.append(f"<li><b>Effective Depth (d):</b> \\( d_x \\) = {n(state.effective_depth_x_mm)} mm, \\( d_y \\) = {n(state.effective_depth_y_mm)} mm</li>")
    html.append(f"</ul>")
    
    # 2. Soil Pressure — use SERVICE (CS-xx) governing case
    soil = results.get("service_soil", {})
    if not soil and "soil_pressure" in results: soil = results["soil_pressure"]
    html.append(f"<h3>2. Soil Bearing Pressure (Service) — Governing Combo: <span style='color:#fbbf24'>{srv_state.loadcase.name}</span></h3>")
    html.append("<p>The soil bearing pressure is calculated using the rigid footing assumption.</p>")
    html.append(f"<div class='formula'>$$ q = \\frac{{P}}{{A}} \\pm \\frac{{M_x \\cdot y}}{{I_x}} \\pm \\frac{{M_y \\cdot x}}{{I_y}} $$</div>")
    # --- Axial load breakdown ---
    Lx_m  = geom.footing_length_x_mm / 1000.0
    Ly_m  = geom.footing_width_y_mm  / 1000.0
    h_m   = geom.footing_thickness_mm / 1000.0
    Df_m  = geom.soil_above_footing_mm / 1000.0
    bx_m, by_m = eng.get_loaded_area_dims(geom)
    bx_m /= 1000.0;  by_m /= 1000.0
    A_foot_m2   = Lx_m * Ly_m
    A_col_m2    = bx_m * by_m
    A_soil_m2   = max(0.0, A_foot_m2 - A_col_m2)
    P_sap       = srv_state.loadcase.Pu_kN
    W_footing   = srv_state.self_weight_kN
    W_soil      = srv_state.overburden_weight_kN
    P_total     = srv_state.total_axial_kN

    html.append(f"<ul>")
    html.append(f"<li><b>Structural Axial Load from SAP2000 (CS-xx):</b> "
                f"\\( P_{{SAP}} \\) = {n(P_sap)} kN</li>")
    html.append(f"<li><b>Footing Self-Weight:</b><br>"
                f"\\( W_{{footing}} = L_x \\times L_y \\times h \\times \\gamma_c "
                f"= {n(Lx_m,3)} \\times {n(Ly_m,3)} \\times {n(h_m,3)} \\times {n(mat.gamma_conc_kN_m3)} "
                f"= \\mathbf{{{n(W_footing)}}} \\) kN</li>")
    html.append(f"<li><b>Soil Overburden Weight:</b><br>"
                f"Net soil area \\( = A_{{footing}} - A_{{column}} "
                f"= ({n(Lx_m,3)} \\times {n(Ly_m,3)}) - ({n(bx_m,3)} \\times {n(by_m,3)}) "
                f"= {n(A_soil_m2,4)} \\) m²<br>"
                f"\\( W_{{soil}} = A_{{net}} \\times D_f \\times \\gamma_{{soil}} "
                f"= {n(A_soil_m2,4)} \\times {n(Df_m,3)} \\times {n(geom.gamma_soil_kN_m3)} "
                f"= \\mathbf{{{n(W_soil)}}} \\) kN</li>")
    html.append(f"<li><b>Total Design Axial Load:</b><br>"
                f"\\( P_{{total}} = P_{{SAP}} + W_{{footing}} + W_{{soil}} "
                f"= {n(P_sap)} + {n(W_footing)} + {n(W_soil)} "
                f"= \\mathbf{{{n(P_total)}}} \\) kN</li>")
    html.append(f"</ul>")

    html.append(f"<p><b>Applied Moments from SAP2000 (CS-xx governing combo):</b></p>")
    html.append(f"<ul>")
    html.append(f"<li>\\( M_x \\) = {n(srv_state.loadcase.Mux_kNm)} kN-m &nbsp;&nbsp; "
                f"\\( M_y \\) = {n(srv_state.loadcase.Muy_kNm)} kN-m</li>")
    html.append(f"</ul>")

    kern_x = geom.footing_length_x_mm / 6.0
    kern_y = geom.footing_width_y_mm / 6.0
    html.append(f"<p><b>Eccentricity &amp; Kern Check:</b></p>")
    html.append(f"<ul>")
    html.append(f"<li>\\( e_x = M_y / P_{{total}} = {n(srv_state.loadcase.Muy_kNm)} / {n(P_total)} "
                f"= {n(soil.get('ex_mm', 0))} \\) mm</li>")
    html.append(f"<li>\\( e_y = M_x / P_{{total}} = {n(srv_state.loadcase.Mux_kNm)} / {n(P_total)} "
                f"= {n(soil.get('ey_mm', 0))} \\) mm</li>")
    html.append(f"<li>Middle Third Limit: \\( L_x/6 \\) = {n(kern_x)} mm, \\( L_y/6 \\) = {n(kern_y)} mm</li>")
    pass_kern = soil.get('kern_check', False)
    kern_status = "<span class='pass'>[PASS] (Full Contact)</span>" if pass_kern else "<span class='fail'>[UPLIFT] (Partial Contact)</span>"
    html.append(f"<li>Kern Check (\\( e_x \\le L_x/6 \\) and \\( e_y \\le L_y/6 \\)): {kern_status}</li>")
    html.append(f"</ul>")
    html.append(f"<p><b>Calculated Pressures:</b></p>")
    html.append(f"<ul>")
    html.append(f"<li>\\( q_{{max}} \\) = {n(soil.get('q_max_kPa', 0))} kPa</li>")
    html.append(f"<li>\\( q_{{min}} \\) = {n(soil.get('q_min_kPa', 0))} kPa</li>")
    html.append(f"</ul>")
    html.append(f"<p>Conclusion: {check_result('Soil Bearing Pressure')}</p>")

    # 3. Flexure
    mdx = results.get("moment_demand_x", {})
    mdy = results.get("moment_demand_y", {})
    fx = results.get("flex_x", {})
    fy = results.get("flex_y", {})
    html.append(f"<h3>3. Flexural Design (Ultimate)</h3>")
    html.append("<p>The critical section for flexure is taken at the face of the column/pedestal.</p>")
    html.append(f"<div class='formula'>$$ M_u = \\frac{{q_{{ult}} \\cdot L_c^2}}{{2}} $$<br>$$ A_{{s,req}} = \\left( \\frac{{0.85 f'_c b}}{{f_y}} \\right) \\left[ 1 - \\sqrt{{1 - \\frac{{2 M_u}}{{\\phi \\cdot 0.85 f'_c b d^2}}}} \\right] d $$</div>")
    html.append("<table><tr><th>Direction</th><th>\\( M_u \\) (kN-m)</th><th>Cantilever \\( L_c \\) (mm)</th><th>\\( A_{{s,req}} \\) (mm²)</th><th>\\( A_{{s,prov}} \\) (mm²)</th><th>Result</th></tr>")
    html.append(f"<tr><td>X-Dir</td><td>{n(mdx.get('Mu_kNm', 0))}</td><td>{n(mdx.get('overhang_mm', 0))}</td><td>{n(fx.get('As_req_mm2', 0))}</td><td>{n(results.get('spacing_x', {}).get('As_prov_mm2', 0))}</td><td>{check_result('Flexure - Bottom X Bars')}</td></tr>")
    html.append(f"<tr><td>Y-Dir</td><td>{n(mdy.get('Mu_kNm', 0))}</td><td>{n(mdy.get('overhang_mm', 0))}</td><td>{n(fy.get('As_req_mm2', 0))}</td><td>{n(results.get('spacing_y', {}).get('As_prov_mm2', 0))}</td><td>{check_result('Flexure - Bottom Y Bars')}</td></tr>")
    html.append("</table>")

    # 4. One-Way Shear
    sx = results.get("one_way_x", {})
    sy = results.get("one_way_y", {})
    html.append(f"<h3>4. One-Way Shear (Ultimate)</h3>")
    html.append("<p>Evaluated at a distance \\( d \\) from the column face.</p>")
    html.append(f"<div class='formula'>$$ \\phi V_c = \\phi \\cdot 0.17 \\lambda \\sqrt{{f'_c}} \\cdot b_w d $$</div>")
    html.append("<ul>")
    html.append(f"<li><b>X-Dir:</b> \\( V_{{ux}} \\) = {n(sx.get('Vu_kN', 0))} kN, \\( \\phi V_c \\) = {n(sx.get('phiVc_kN', 0))} kN. {check_result('One-Way Shear X')}</li>")
    html.append(f"<li><b>Y-Dir:</b> \\( V_{{uy}} \\) = {n(sy.get('Vu_kN', 0))} kN, \\( \\phi V_c \\) = {n(sy.get('phiVc_kN', 0))} kN. {check_result('One-Way Shear Y')}</li>")
    html.append("</ul>")

    # 5. Two-Way Punching
    p = results.get("punch", {})
    html.append(f"<h3>5. Two-Way Punching Shear (Ultimate)</h3>")
    html.append("<p>Evaluated at \\( d/2 \\) from the column perimeter. Capacity is the minimum of 3 ACI empirical equations.</p>")
    html.append(f"<div class='formula'>$$ \\phi V_c = \\min \\left[ 0.33\\lambda\\sqrt{{f'_c}}, \\quad 0.17\\left(1+\\frac{{2}}{{\\beta}}\\right)\\lambda\\sqrt{{f'_c}}, \\quad 0.083\\left(\\frac{{\\alpha_s d}}{{b_o}}+2\\right)\\lambda\\sqrt{{f'_c}} \\right] b_o d $$</div>")
    html.append("<ul>")
    html.append(f"<li>Perimeter \\( b_o \\) = {n(p.get('bo_mm', 0))} mm</li>")
    html.append(f"<li>Aspect ratio \\( \\beta \\) = {n(p.get('beta', 0))}</li>")
    html.append(f"<li>Position factor \\( \\alpha_s \\) = {n(p.get('alpha_s', 40))} (40 int, 30 edge, 20 corner)</li>")
    html.append(f"<li>\\( V_u \\) = {n(p.get('Vu_kN', 0))} kN</li>")
    html.append(f"<li>\\( \\phi V_c \\) = {n(p.get('phiVc_kN', 0))} kN</li>")
    html.append(f"</ul>")
    html.append(f"<p>Conclusion: {check_result('Two-Way Punching Shear')}</p>")

    # 6. Bearing & Dowel
    dw = results.get("dowels", {})
    br = results.get("bearing", {})
    html.append(f"<h3>6. Concrete Bearing & Dowels</h3>")
    html.append(f"<p>Evaluates bearing stress at column base and excess load transfer.</p>")
    html.append(f"<div class='formula'>$$ \\phi P_n = \\phi \\cdot 0.85 f'_c A_1 \\cdot \\min\\left(\\sqrt{{\\frac{{A_2}}{{A_1}}}}, 2.0\\right) $$</div>")
    html.append(f"<ul>")
    html.append(f"<li>Column Bearing \\( P_u \\) = {n(state.loadcase.Pu_kN)} kN</li>")
    html.append(f"<li>Footing Bearing Capacity \\( \\phi P_n \\) = {n(br.get('phiPn_kN', 0))} kN</li>")
    html.append(f"<li>Required Dowel \\( A_s \\) = {n(dw.get('As_dowel_req_mm2', 0))} mm²</li>")
    html.append(f"</ul>")
    html.append(f"<p>Conclusion: {check_result('Concrete Bearing')}</p>")

    html.append("</body></html>")
    return "".join(html)
