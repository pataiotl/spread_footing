# Spread Footing Designer — PySide6

> **ACI 318-19 compliant isolated spread footing design desktop application.**  
> Integrates directly with **SAP2000** via OAPI to import joint reactions and perform full service + ultimate batch design with envelope logic.

---

## Features

### Load Input
- **Manual Table** — enter service loads (P, Mx, My, Vx, Vy) directly per footing group
- **SAP2000 OAPI Import** — live connection to a running SAP2000 instance; fetches selected joint reactions for all load combinations in one click

### Load Case Convention
| Prefix | Purpose |
|--------|---------|
| `CS-xx` | **Service** combinations — used for footing sizing and soil bearing pressure checks |
| `CU-xx` | **Ultimate** combinations — used for RC strength design (flexure, shear, punching) |

Any load combination not starting with `CS-` or `CU-` is automatically excluded.

### Grouping (Multi-Joint Footings)
- Joints from multiple columns can be **grouped** into a single footing (e.g. combined footing supporting 2–4 columns)
- All load combinations for all joints in the group are **summed** per combination and **enveloped** to find the governing service and governing ultimate case
- The worst-case P/A ± Mc/I result governs footing size; the worst-case ultimate combination governs RC design

### Design Engine (`footing_engine.py`)
Core formula applied at every load combination:

$$q = \frac{P}{A} \pm \frac{M_x \cdot y}{I_x} \pm \frac{M_y \cdot x}{I_y}$$

**Axial Load:**
$$P_{total} = P_{SAP} + W_{footing} + W_{soil}$$

Where:
- $W_{footing} = L_x \times L_y \times h \times \gamma_c$
- $W_{soil} = (A_{footing} - A_{column}) \times D_f \times \gamma_{soil}$

**Checks performed (ACI 318-19):**

| # | Check | Standard |
|---|-------|----------|
| 1 | Soil Bearing Pressure (service) | Soil mechanics / Middle-Third rule |
| 2 | Eccentricity / Kern boundary ($e \le L/6$) | Meyerhof / ACI |
| 3 | Overturning stability | Statics |
| 4 | Sliding stability | Statics |
| 5 | Flexural design — Bottom X & Y bars | ACI 318-19 §22.6 |
| 6 | One-way shear (beam shear) — X & Y | ACI 318-19 §22.5 |
| 7 | Two-way punching shear | ACI 318-19 §22.6 |
| 8 | Concrete bearing & column dowels | ACI 318-19 §22.8 |
| 9 | Development length — X & Y bars | ACI 318-19 §25.4 |

### UI Tabs
| Tab | Contents |
|-----|----------|
| **1 - Input Tables** | Manual table or SAP2000 OAPI importer with joint picker and group manager |
| **Dashboard** | Summary metrics (status, max D/C, footing size, governing soil pressure) |
| **Structural Checks** | Full check table with Demand / Capacity / D/C ratio per check |
| **Flexural Design** | Bar size, spacing, required vs provided area per direction |
| **Detailed Outputs** | Material takeoff (concrete volume, rebar weight), effective depths, shear capacities |
| **Calculation** | Step-by-step calculation report with rendered LaTeX equations |
| **Visuals** | Engineering drawing — Plan View (rebar grid, dimension lines, punching perimeter) and Section X-X / Y-Y (bar callouts, effective depth, cover) |

---

## Installation

### Requirements
- **Python** ≥ 3.10
- **SAP2000** v22+ (for OAPI import, optional)
- **Windows** (OAPI integration uses Windows COM)

### Install dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
PySide6>=6.5.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
openpyxl>=3.1.0
fpdf>=1.7.2
comtypes>=1.2.0
```

> `comtypes` is only required for SAP2000 OAPI. The app runs fully in manual mode without it.

---

## Running the App

**Windows (recommended):**
```bat
start_app.bat
```

**Or directly with Python:**
```bash
python run_pyside6.py
```

---

## SAP2000 OAPI Workflow

1. Open SAP2000 and **run the analysis**
2. **Select the restraint joint nodes** (the small blue crosses at column base supports) in SAP2000
3. In the app, switch Load Source to **SAP2000 OAPI Import** → click **Fetch Selected Nodes from SAP2000**
4. The app reads all joint reactions for all load combinations and lists **unique joint IDs** in the joint picker
5. Select one or more joints (Ctrl+Click for multi-select) → press **Add →** to assign them to a footing group
6. Repeat for each footing group (F1, F2, …)
7. Set footing dimensions (Lx, Ly, Thickness) per group in the groups table
8. Press **Run Design**

> **Important:** SAP2000 and this app must run at the **same privilege level** (both as normal user, or both as Administrator). A privilege mismatch will cause COM connection errors.

---

## Load Case Naming Convention

The engine automatically filters by prefix. Name your SAP2000 load combinations as:

```
CS-01, CS-02, CS-03, ...   →  Service (footing size / bearing pressure)
CU-01, CU-02, CU-03, ...   →  Ultimate (RC flexure / shear design)
```

Any other naming is silently ignored.

---

## Project Structure

```
Spread_Footing_Pyside6/
├── run_pyside6.py        # Entry point
├── start_app.bat         # Windows launcher
├── requirements.txt      # Python dependencies
│
├── footing_engine.py     # Core structural design engine
│   ├── Material, FootingGeometry, Reinforcement, LoadCase (dataclasses)
│   ├── DesignState       # Computed state per load case
│   ├── design_all()      # Full ACI 318-19 check suite
│   ├── envelope_group_design()  # Service + Ultimate envelope logic
│   ├── design_batch_from_sap_table()  # SAP2000 batch processor
│   ├── plot_plan()       # Engineering plan view (matplotlib)
│   └── plot_section()    # Engineering section view (matplotlib)
│
├── main_window.py        # PySide6 main window and UI logic
│   ├── Input Tables tab  # Manual + SAP2000 OAPI input
│   ├── Result tabs       # Dashboard, Checks, Flexure, Details
│   ├── Calculation tab   # HTML report renderer
│   └── Visuals tab       # Matplotlib canvas
│
├── view_data.py          # Data formatters
│   ├── calculation_html()       # Step-by-step LaTeX HTML report
│   ├── checks_dataframe()       # Check results table
│   ├── flexural_summary_dataframe()
│   └── soil_pressure_dataframe()
│
├── qt_models.py          # PandasTableModel (sortable, editable)
└── sap2000_oapi.py       # SAP2000 COM/OAPI bridge
```

---

## Design Assumptions

- Rigid footing assumption (linear soil pressure distribution)
- ACI 318-19 strength reduction factors: φ = 0.90 (flexure), φ = 0.75 (shear)
- Concrete unit weight: 24 kN/m³ (configurable)
- Soil unit weight: 18 kN/m³ (configurable)
- Bottom cover governs effective depth calculation
- Self-weight and soil overburden are **added to the SAP2000 axial reaction** before soil pressure check
- Moments from SAP2000 are applied **as-is** (not amplified by self-weight eccentricity)
- Middle Third rule (e ≤ L/6) is used as a serviceability indicator; Meyerhof effective area method is used when e > L/6

---

## Screenshots

> *(Add screenshots here)*

---

## License

This project is for internal engineering use. All rights reserved.
