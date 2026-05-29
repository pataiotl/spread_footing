"""PySide6 main window for Spread Footing Designer."""
import sys
import traceback
from typing import Any, Dict

import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QIcon, QFont, QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QSplitter, QGroupBox, QFormLayout, QLineEdit,
    QPushButton, QComboBox, QTableView, QHeaderView, QLabel,
    QMessageBox, QDoubleSpinBox, QSpinBox, QCheckBox, QScrollArea,
    QProgressBar, QFileDialog, QListWidget, QGridLayout
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Local imports
import footing_engine as eng
import view_data as vdata
from qt_models import PandasTableModel
try:
    import sap2000_oapi
except ImportError:
    sap2000_oapi = None

class DesignWorker(QThread):
    """Background thread to run the design without freezing the UI."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, state: eng.DesignState, load_cases_df: pd.DataFrame = None, 
                 sap_df: pd.DataFrame = None, sap_groups_df: pd.DataFrame = None, mode: str = "single"):
        super().__init__()
        self.state = state
        self.load_cases_df = load_cases_df
        self.sap_df = sap_df
        self.sap_groups_df = sap_groups_df
        self.mode = mode

    def run(self):
        try:
            if self.mode == "manual_batch":
                # Batch processing from manual table
                res = eng.design_batch_from_manual_table(
                    self.load_cases_df, 
                    self.state.material,
                    self.state.geometry,
                    self.state.reinforcement
                )
                self.finished.emit(res)
            elif self.mode == "sap_batch":
                # Batch processing from SAP2000 groups
                res = eng.design_batch_from_sap_table(
                    self.sap_df, 
                    self.sap_groups_df, 
                    self.state.material, self.state.geometry, self.state.reinforcement, 
                    None, None, 
                    v_sign=1.0 # Positive F3 reaction means compression on footing
                )
                self.finished.emit(res)
            else:
                # Single envelope run
                res = eng.design_all(self.state, is_service=self.state.loadcase.case_type == "Service")
                self.finished.emit({"single_run": res})
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(eng.APP_TITLE)
        self.resize(1400, 900)
        
        self.current_state = None
        self.worker = None
        self.batch_df = pd.DataFrame()
        self.sap_df = pd.DataFrame()
        self.batch_mode = False
        
        self._init_ui()
        self._setup_defaults()
        
    def _init_ui(self):
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Left Panel (Inputs)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumWidth(350)
        self.scroll_area.setMaximumWidth(450)
        
        self.input_widget = QWidget()
        self.input_layout = QVBoxLayout(self.input_widget)
        self._build_input_panel()
        self.input_layout.addStretch()
        self.scroll_area.setWidget(self.input_widget)
        
        self.splitter.addWidget(self.scroll_area)
        
        # Right Panel (Tabs)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self._build_tabs()
        
        # Bottom status bar equivalent
        self.status_layout = QHBoxLayout()
        self.calc_btn = QPushButton("Run Design")
        self.calc_btn.setMinimumHeight(40)
        self.calc_btn.setStyleSheet("font-weight: bold; background-color: #2e86c1; color: white;")
        self.calc_btn.clicked.connect(self.run_design)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.hide()
        
        self.status_layout.addWidget(self.calc_btn)
        self.status_layout.addWidget(self.progress_bar)
        
        self.right_layout.addLayout(self.status_layout)
        
        self.splitter.addWidget(self.right_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        
    def _build_input_panel(self):
        # Material
        gb_mat = QGroupBox("Material Properties")
        fl_mat = QFormLayout(gb_mat)
        self.inp_fc = QDoubleSpinBox(); self.inp_fc.setRange(15, 100); self.inp_fc.setSuffix(" MPa")
        self.inp_fy = QDoubleSpinBox(); self.inp_fy.setRange(200, 600); self.inp_fy.setSuffix(" MPa")
        self.inp_gamma_c = QDoubleSpinBox(); self.inp_gamma_c.setRange(20, 30); self.inp_gamma_c.setSuffix(" kN/m³")
        fl_mat.addRow("Concrete f'c:", self.inp_fc)
        fl_mat.addRow("Rebar fy:", self.inp_fy)
        fl_mat.addRow("Unit Weight γc:", self.inp_gamma_c)
        self.input_layout.addWidget(gb_mat)
        
        # Geometry
        gb_geom = QGroupBox("Footing Geometry")
        fl_geom = QFormLayout(gb_geom)
        self.inp_Lx = QDoubleSpinBox(); self.inp_Lx.setRange(500, 10000); self.inp_Lx.setSuffix(" mm"); self.inp_Lx.setSingleStep(100)
        self.inp_Ly = QDoubleSpinBox(); self.inp_Ly.setRange(500, 10000); self.inp_Ly.setSuffix(" mm"); self.inp_Ly.setSingleStep(100)
        self.inp_h = QDoubleSpinBox(); self.inp_h.setRange(200, 3000); self.inp_h.setSuffix(" mm"); self.inp_h.setSingleStep(50)
        self.inp_embed = QDoubleSpinBox(); self.inp_embed.setRange(0, 5000); self.inp_embed.setSuffix(" mm"); self.inp_embed.setSingleStep(100)
        self.inp_soil_h = QDoubleSpinBox(); self.inp_soil_h.setRange(0, 5000); self.inp_soil_h.setSuffix(" mm"); self.inp_soil_h.setSingleStep(100)
        fl_geom.addRow("Length (X):", self.inp_Lx)
        fl_geom.addRow("Width (Y):", self.inp_Ly)
        fl_geom.addRow("Thickness (h):", self.inp_h)
        fl_geom.addRow("Embedment Depth:", self.inp_embed)
        fl_geom.addRow("Soil Above Footing:", self.inp_soil_h)
        self.input_layout.addWidget(gb_geom)
        
        # Column
        gb_col = QGroupBox("Column / Pedestal")
        fl_col = QFormLayout(gb_col)
        self.inp_cbx = QDoubleSpinBox(); self.inp_cbx.setRange(100, 2000); self.inp_cbx.setSuffix(" mm"); self.inp_cbx.setSingleStep(50)
        self.inp_cby = QDoubleSpinBox(); self.inp_cby.setRange(100, 2000); self.inp_cby.setSuffix(" mm"); self.inp_cby.setSingleStep(50)
        self.inp_col_shape = QComboBox(); self.inp_col_shape.addItems(["Rectangular", "Circular"])
        self.inp_col_loc = QComboBox(); self.inp_col_loc.addItems(["Interior", "Edge", "Corner"])
        fl_col.addRow("Column Cx:", self.inp_cbx)
        fl_col.addRow("Column Cy:", self.inp_cby)
        fl_col.addRow("Shape:", self.inp_col_shape)
        fl_col.addRow("Location:", self.inp_col_loc)
        self.input_layout.addWidget(gb_col)
        
        # Soil
        gb_soil = QGroupBox("Geotechnical Params")
        fl_soil = QFormLayout(gb_soil)
        self.inp_qa = QDoubleSpinBox(); self.inp_qa.setRange(50, 2000); self.inp_qa.setSuffix(" kPa")
        self.inp_gamma_s = QDoubleSpinBox(); self.inp_gamma_s.setRange(10, 25); self.inp_gamma_s.setSuffix(" kN/m³")
        self.inp_fric = QDoubleSpinBox(); self.inp_fric.setRange(0.1, 0.8); self.inp_fric.setSingleStep(0.05)
        self.inp_passive = QDoubleSpinBox(); self.inp_passive.setRange(0, 500); self.inp_passive.setSuffix(" kPa")
        fl_soil.addRow("Allowable Bearing:", self.inp_qa)
        fl_soil.addRow("Soil Unit Weight:", self.inp_gamma_s)
        fl_soil.addRow("Friction Coeff:", self.inp_fric)
        fl_soil.addRow("Passive Pressure:", self.inp_passive)
        self.input_layout.addWidget(gb_soil)
        
        # Reinforcement
        gb_reinf = QGroupBox("Reinforcement")
        fl_reinf = QFormLayout(gb_reinf)
        bars = list(eng.BAR_DATABASE_MM.keys())
        self.inp_bar_x = QComboBox(); self.inp_bar_x.addItems(bars)
        self.inp_bar_y = QComboBox(); self.inp_bar_y.addItems(bars)
        self.inp_spac_x = QDoubleSpinBox(); self.inp_spac_x.setRange(50, 500); self.inp_spac_x.setSuffix(" mm")
        self.inp_spac_y = QDoubleSpinBox(); self.inp_spac_y.setRange(50, 500); self.inp_spac_y.setSuffix(" mm")
        fl_reinf.addRow("Bottom X Bar:", self.inp_bar_x)
        fl_reinf.addRow("Spacing X:", self.inp_spac_x)
        fl_reinf.addRow("Bottom Y Bar:", self.inp_bar_y)
        fl_reinf.addRow("Spacing Y:", self.inp_spac_y)
        self.input_layout.addWidget(gb_reinf)
        
        # Load Case (Envelope Override)
        gb_load = QGroupBox("Envelope Loads (Manual Override)")
        fl_load = QFormLayout(gb_load)
        self.inp_pu = QDoubleSpinBox(); self.inp_pu.setRange(0, 50000); self.inp_pu.setSuffix(" kN")
        self.inp_mux = QDoubleSpinBox(); self.inp_mux.setRange(-5000, 5000); self.inp_mux.setSuffix(" kN-m")
        self.inp_muy = QDoubleSpinBox(); self.inp_muy.setRange(-5000, 5000); self.inp_muy.setSuffix(" kN-m")
        self.inp_vux = QDoubleSpinBox(); self.inp_vux.setRange(-2000, 2000); self.inp_vux.setSuffix(" kN")
        self.inp_vuy = QDoubleSpinBox(); self.inp_vuy.setRange(-2000, 2000); self.inp_vuy.setSuffix(" kN")
        fl_load.addRow("Pu (Axial):", self.inp_pu)
        fl_load.addRow("Mux (Mom X):", self.inp_mux)
        fl_load.addRow("Muy (Mom Y):", self.inp_muy)
        fl_load.addRow("Vux (Shear X):", self.inp_vux)
        fl_load.addRow("Vuy (Shear Y):", self.inp_vuy)
        
        self.chk_service = QCheckBox("Run as Service Load (Soil Check)")
        fl_load.addRow(self.chk_service)
        self.input_layout.addWidget(gb_load)

    def _build_tabs(self):
        # Global batch result selector
        self.result_group_combo = QComboBox()
        self.result_group_combo.currentIndexChanged.connect(self._on_result_group_changed)
        
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("<b>Viewing Results for Group:</b>"))
        selector_layout.addWidget(self.result_group_combo)
        selector_layout.addStretch()
        
        self.right_layout.addLayout(selector_layout)
        
        self.tabs = QTabWidget()
        self.right_layout.addWidget(self.tabs)
        
        # Batch Input Tab
        self._build_input_tab()
        
        # Dashboard Tab
        self.tab_dash = QWidget()
        dash_layout = QVBoxLayout(self.tab_dash)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold; color: gray;")
        dash_layout.addWidget(self.lbl_status)
        
        self.tbl_summary = QTableView()
        self.tbl_summary.horizontalHeader().setStretchLastSection(True)
        self.tbl_summary.setAlternatingRowColors(True)
        dash_layout.addWidget(QLabel("<b>Design Status</b>"))
        dash_layout.addWidget(self.tbl_summary)
        
        self.tbl_soil = QTableView()
        self.tbl_soil.horizontalHeader().setStretchLastSection(True)
        self.tbl_soil.setAlternatingRowColors(True)
        dash_layout.addWidget(QLabel("<b>Soil Bearing Pressure</b>"))
        dash_layout.addWidget(self.tbl_soil)
        
        self.tabs.addTab(self.tab_dash, "Dashboard")
        
        # Checks Tab
        self.tab_checks = QWidget()
        checks_layout = QVBoxLayout(self.tab_checks)
        self.tbl_checks = QTableView()
        self.tbl_checks.horizontalHeader().setStretchLastSection(True)
        self.tbl_checks.setAlternatingRowColors(True)
        checks_layout.addWidget(self.tbl_checks)
        self.tabs.addTab(self.tab_checks, "Structural Checks")
        
        # Flexure Tab
        self.tab_flex = QWidget()
        flex_layout = QVBoxLayout(self.tab_flex)
        self.tbl_flex = QTableView()
        self.tbl_flex.horizontalHeader().setStretchLastSection(True)
        self.tbl_flex.setAlternatingRowColors(True)
        flex_layout.addWidget(self.tbl_flex)
        self.tabs.addTab(self.tab_flex, "Flexural Design")
        
        # Details Tab
        self.tab_details = QWidget()
        details_layout = QVBoxLayout(self.tab_details)
        self.tbl_details = QTableView()
        self.tbl_details.horizontalHeader().setStretchLastSection(True)
        self.tbl_details.setAlternatingRowColors(True)
        details_layout.addWidget(self.tbl_details)
        self.tabs.addTab(self.tab_details, "Detailed Outputs")
        
        # Calculation Tab
        self.tab_calc = QWidget()
        calc_layout = QVBoxLayout(self.tab_calc)
        from PySide6.QtWebEngineWidgets import QWebEngineView
        self.calculation_text = QWebEngineView()
        calc_layout.addWidget(self.calculation_text)
        self.tabs.addTab(self.tab_calc, "Calculation")
        
        # Plots Tab
        self.tab_plots = QWidget()
        plots_layout = QVBoxLayout(self.tab_plots)
        
        # Matplotlib canvas
        self.fig_plan, self.ax_plan = plt.subplots(figsize=(5, 5))
        self.canvas_plan = FigureCanvas(self.fig_plan)
        
        self.fig_sec, self.ax_sec = plt.subplots(figsize=(6, 3))
        self.canvas_sec = FigureCanvas(self.fig_sec)
        
        plots_splitter = QSplitter(Qt.Horizontal)
        plots_splitter.addWidget(self.canvas_plan)
        plots_splitter.addWidget(self.canvas_sec)
        
        plots_layout.addWidget(plots_splitter)
        self.tabs.addTab(self.tab_plots, "Visuals")
        
    def _build_input_tab(self):
        self.tab_input = QWidget()
        layout = QVBoxLayout(self.tab_input)
        
        top = QHBoxLayout()
        self.load_source = QComboBox()
        self.load_source.addItems(["Manual input table", "SAP2000 OAPI Import"])
        self.load_source.currentIndexChanged.connect(self._switch_load_source)
        top.addWidget(QLabel("Load Source:"))
        top.addWidget(self.load_source, 0)
        top.addStretch(1)
        layout.addLayout(top)
        
        self.input_stack = QTabWidget()
        
        # Manual Page
        self.manual_page = QWidget()
        man_layout = QVBoxLayout(self.manual_page)
        man_btns = QHBoxLayout()
        btn_add = QPushButton("Add Footing")
        btn_add.clicked.connect(self.add_manual_row)
        man_btns.addWidget(btn_add)
        man_btns.addStretch()
        man_layout.addLayout(man_btns)
        
        default_manual = eng.default_manual_foundations(eng.FootingGeometry())
        self.manual_model = PandasTableModel(default_manual, editable=True)
        self.manual_table = QTableView()
        self.manual_table.setModel(self.manual_model)
        self.manual_table.setAlternatingRowColors(True)
        man_layout.addWidget(QLabel("Manual Service Load & Geometry Input:"))
        man_layout.addWidget(self.manual_table)
        self.input_stack.addTab(self.manual_page, "Manual Table")
        
        # SAP2000 Page
        self.sap_page = QWidget()
        sap_layout = QVBoxLayout(self.sap_page)
        sap_layout.setSpacing(6)

        # ── Row 1: Fetch button + status ──────────────────────────────────────
        sap_btns = QHBoxLayout()
        btn_fetch = QPushButton("Fetch Selected Nodes from SAP2000")
        btn_fetch.clicked.connect(self.fetch_sap_oapi)
        btn_fetch.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; padding: 6px 14px;")
        sap_btns.addWidget(btn_fetch)
        self.lbl_sap_status = QLabel("Not connected")
        sap_btns.addWidget(self.lbl_sap_status)
        sap_btns.addStretch()
        sap_layout.addLayout(sap_btns)

        # ── Row 2: Full raw data table ────────────────────────────────────────
        sap_layout.addWidget(QLabel("Raw SAP2000 Imported Data (all load combinations):"))
        self.sap_raw_model = PandasTableModel(pd.DataFrame())
        self.sap_raw_table = QTableView()
        self.sap_raw_table.setModel(self.sap_raw_model)
        self.sap_raw_table.setAlternatingRowColors(True)
        self.sap_raw_table.setSelectionBehavior(QTableView.SelectRows)
        self.sap_raw_table.setSortingEnabled(True)   # click header to sort
        self.sap_raw_table.setMaximumHeight(220)
        sap_layout.addWidget(self.sap_raw_table)

        # ── Row 3: Joint picker panel ─────────────────────────────────────────
        picker_group = QGroupBox("Assign Joints to Group")
        picker_layout = QGridLayout(picker_group)
        picker_layout.setColumnStretch(1, 1)
        picker_layout.setColumnStretch(3, 1)

        # Target group (left col)
        picker_layout.addWidget(QLabel("Target Group:"), 0, 0)
        self.assign_group_combo = QComboBox()
        self.assign_group_combo.setEditable(True)
        self.assign_group_combo.setMinimumWidth(120)
        picker_layout.addWidget(self.assign_group_combo, 0, 1)

        # Joint multi-select list (left col, spans rows)
        picker_layout.addWidget(QLabel("Available\nJoints\n(multi-select):"), 1, 0, Qt.AlignTop)
        self.joint_pick_list = QListWidget()
        self.joint_pick_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.joint_pick_list.setMaximumHeight(130)
        self.joint_pick_list.setMaximumWidth(160)
        picker_layout.addWidget(self.joint_pick_list, 1, 1)

        # Arrow button column
        arrow_col = QVBoxLayout()
        arrow_col.setAlignment(Qt.AlignVCenter)
        btn_add_joint = QPushButton("Add →")
        btn_add_joint.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 6px;")
        btn_add_joint.clicked.connect(self._add_joint_to_group)
        arrow_col.addStretch()
        arrow_col.addWidget(btn_add_joint)
        arrow_col.addStretch()
        picker_layout.addLayout(arrow_col, 1, 2)

        # Assigned joints list (right col)
        picker_layout.addWidget(QLabel("Assigned\nJoints:"), 1, 3, Qt.AlignTop)
        assigned_col = QVBoxLayout()
        self.assigned_joints_list = QListWidget()
        self.assigned_joints_list.setMaximumHeight(100)
        self.assigned_joints_list.setSelectionMode(QListWidget.ExtendedSelection)
        assigned_col.addWidget(self.assigned_joints_list)
        btn_remove_joint = QPushButton("✕ Remove Selected")
        btn_remove_joint.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
        btn_remove_joint.clicked.connect(self._remove_joints_from_group)
        assigned_col.addWidget(btn_remove_joint)
        picker_layout.addLayout(assigned_col, 1, 4)

        sap_layout.addWidget(picker_group)

        # ── Row 4: Groups table ───────────────────────────────────────────────
        groups_header_layout = QHBoxLayout()
        groups_header_layout.addWidget(QLabel("SAP2000 Foundation Groups:"))
        btn_add_group = QPushButton("+ Add Foundation Group")
        btn_add_group.clicked.connect(self.add_sap_group_row)
        groups_header_layout.addWidget(btn_add_group)
        groups_header_layout.addStretch()
        sap_layout.addLayout(groups_header_layout)

        df_initial_groups = eng.default_sap_groups(pd.DataFrame(), eng.FootingGeometry())
        self.sap_groups_model = PandasTableModel(df_initial_groups, editable=True)
        self.sap_groups_table = QTableView()
        self.sap_groups_table.setModel(self.sap_groups_model)
        self.sap_groups_table.setAlternatingRowColors(True)
        self.sap_groups_table.selectionModel().currentRowChanged.connect(self._on_group_row_selected)
        sap_layout.addWidget(self.sap_groups_table)

        self.input_stack.addTab(self.sap_page, "SAP2000 OAPI")
        
        layout.addWidget(self.input_stack)
        self.tabs.addTab(self.tab_input, "1 - Input Tables")
        self._update_group_combo(df_initial_groups)

    def _switch_load_source(self):
        self.input_stack.setCurrentIndex(self.load_source.currentIndex())
        
    def _update_group_combo(self, df_groups):
        self.assign_group_combo.clear()
        if "Group Name" in df_groups.columns:
            groups = df_groups["Group Name"].dropna().unique().tolist()
            self.assign_group_combo.addItems(groups)
            
    def _update_group_combo(self, df_groups):
        self.assign_group_combo.clear()
        if "Group Name" in df_groups.columns:
            groups = df_groups["Group Name"].dropna().unique().tolist()
            self.assign_group_combo.addItems([str(g) for g in groups])

    def _refresh_joint_pick_combo(self):
        """Populate the joint picker list with unique joint IDs from the fetched SAP2000 data."""
        self.joint_pick_list.clear()
        if self.sap_df is not None and not self.sap_df.empty and "Joint" in self.sap_df.columns:
            unique_joints = sorted(
                self.sap_df["Joint"].astype(str).unique().tolist(),
                key=lambda x: int(x) if x.isdigit() else x
            )
            self.joint_pick_list.addItems(unique_joints)

    def _on_group_row_selected(self, current, previous):
        """When user clicks a group row, load its current joint IDs into the assigned-joints list."""
        if not current.isValid():
            return
        df = self.sap_groups_model.dataframe
        row_idx = current.row()
        if row_idx < 0 or row_idx >= len(df):
            return
        group_name = str(df.at[row_idx, "Group Name"]) if "Group Name" in df.columns else ""
        self.assign_group_combo.setCurrentText(group_name)
        joint_str = str(df.at[row_idx, "Joint IDs"]) if "Joint IDs" in df.columns else ""
        joints = [j.strip() for j in joint_str.split(",") if j.strip() and j.strip() != "nan"]
        self.assigned_joints_list.clear()
        self.assigned_joints_list.addItems(joints)

    def _add_joint_to_group(self):
        """Add all selected joints from the multi-select list to the current group."""
        selected_items = self.joint_pick_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Select one or more joints from the list first.")
            return
        existing = [self.assigned_joints_list.item(i).text()
                    for i in range(self.assigned_joints_list.count())]
        added = 0
        for item in selected_items:
            joint_id = item.text().strip()
            if joint_id and joint_id not in existing:
                self.assigned_joints_list.addItem(joint_id)
                existing.append(joint_id)
                added += 1
        if added:
            self._sync_assigned_joints_to_group_table()

    def _remove_joints_from_group(self):
        """Remove selected joints from the assigned-joints list."""
        for item in self.assigned_joints_list.selectedItems():
            self.assigned_joints_list.takeItem(self.assigned_joints_list.row(item))
        self._sync_assigned_joints_to_group_table()

    def _sync_assigned_joints_to_group_table(self):
        """Write the current assigned-joints list back into the groups dataframe."""
        group_name = self.assign_group_combo.currentText().strip()
        if not group_name:
            return
        joints = [self.assigned_joints_list.item(i).text()
                  for i in range(self.assigned_joints_list.count())]
        joint_str = ", ".join(joints)
        df_groups = self.sap_groups_model.dataframe.copy()
        if "Group Name" in df_groups.columns and group_name in df_groups["Group Name"].values:
            idx = df_groups.index[df_groups["Group Name"] == group_name][0]
            df_groups.at[idx, "Joint IDs"] = joint_str
        else:
            # Group doesn't exist yet — add it
            template = eng.default_sap_groups(pd.DataFrame(), eng.FootingGeometry())
            new_row = template.iloc[0].copy()
            new_row["Group Name"] = group_name
            new_row["Joint IDs"] = joint_str
            df_groups = pd.concat([df_groups, pd.DataFrame([new_row])], ignore_index=True)
        self.sap_groups_model = PandasTableModel(df_groups, editable=True)
        self.sap_groups_table.setModel(self.sap_groups_model)
        self.sap_groups_table.selectionModel().currentRowChanged.connect(self._on_group_row_selected)
        self._update_group_combo(df_groups)

    def _assign_joints_to_group(self):
        """Legacy fallback — kept for compatibility but no longer called from UI."""
        pass

    def add_manual_row(self):
        df = self.manual_model.dataframe.copy()
        new_row = df.iloc[-1].copy() if not df.empty else eng.default_manual_foundations(eng.FootingGeometry()).iloc[0]
        new_row["Group Name"] = f"F{len(df)+1}"
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.manual_model = PandasTableModel(df, editable=True)
        self.manual_table.setModel(self.manual_model)
        
    def add_sap_group_row(self):
        df = self.sap_groups_model.dataframe.copy()
        new_row = df.iloc[-1].copy() if not df.empty else eng.default_sap_groups(pd.DataFrame(), eng.FootingGeometry()).iloc[0]
        new_row["Group Name"] = f"F{len(df)+1}"
        new_row["Joint IDs"] = ""
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.sap_groups_model = PandasTableModel(df, editable=True)
        self.sap_groups_table.setModel(self.sap_groups_model)
        self._update_group_combo(df)
        
    def fetch_sap_oapi(self):
        if sap2000_oapi is None:
            QMessageBox.critical(self, "Error", "comtypes not installed. Cannot use SAP2000 OAPI.")
            return
            
        from PySide6.QtWidgets import QProgressDialog, QApplication
        from PySide6.QtCore import Qt
        progress = QProgressDialog("Communicating with SAP2000 OAPI... Please wait.", None, 0, 0, self)
        progress.setWindowTitle("SAP2000 Connection")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        try:
            df = sap2000_oapi.fetch_selected_joint_reactions()
            self.sap_df = df
            unique_j = df["Joint"].astype(str).unique() if "Joint" in df.columns else []
            self.lbl_sap_status.setText(
                f"Fetched {len(df)} rows | {len(unique_j)} unique joints: {', '.join(sorted(unique_j, key=lambda x: int(x) if x.isdigit() else x))}"
            )

            # Update raw data table (full data)
            self.sap_raw_model = PandasTableModel(df, editable=False)
            self.sap_raw_table.setModel(self.sap_raw_model)

            # Populate the joint picker multi-select list
            self._refresh_joint_pick_combo()

            # Update default groups ONLY IF empty
            if self.sap_groups_model.dataframe.empty or "Group Name" not in self.sap_groups_model.dataframe.columns:
                geom = eng.FootingGeometry(footing_length_x_mm=self.inp_Lx.value(), footing_width_y_mm=self.inp_Ly.value(), footing_thickness_mm=self.inp_h.value())
                df_groups = eng.default_sap_groups(df, geom)
                self.sap_groups_model = PandasTableModel(df_groups, editable=True)
                self.sap_groups_table.setModel(self.sap_groups_model)
                self.sap_groups_table.selectionModel().currentRowChanged.connect(self._on_group_row_selected)
                self._update_group_combo(df_groups)

            progress.close()
            QMessageBox.information(self, "Success", f"Successfully imported {len(unique_j)} joints from SAP2000.")
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "SAP2000 Error", str(e))

    def _setup_defaults(self):
        mat = eng.Material()
        geom = eng.FootingGeometry()
        reinf = eng.Reinforcement()
        
        self.inp_fc.setValue(mat.fc_MPa)
        self.inp_fy.setValue(mat.fy_MPa)
        self.inp_gamma_c.setValue(mat.gamma_conc_kN_m3)
        
        self.inp_Lx.setValue(geom.footing_length_x_mm)
        self.inp_Ly.setValue(geom.footing_width_y_mm)
        self.inp_h.setValue(geom.footing_thickness_mm)
        self.inp_embed.setValue(geom.footing_embedment_depth_mm)
        self.inp_soil_h.setValue(geom.soil_above_footing_mm)
        
        self.inp_cbx.setValue(geom.column_bx_mm)
        self.inp_cby.setValue(geom.column_by_mm)
        self.inp_col_shape.setCurrentText(geom.column_shape.capitalize())
        self.inp_col_loc.setCurrentText(geom.column_location.capitalize())
        
        self.inp_qa.setValue(geom.allowable_bearing_kPa)
        self.inp_gamma_s.setValue(geom.gamma_soil_kN_m3)
        self.inp_fric.setValue(geom.friction_coefficient)
        self.inp_passive.setValue(geom.passive_pressure_kPa)
        
        self.inp_bar_x.setCurrentText(reinf.main_bar_x)
        self.inp_bar_y.setCurrentText(reinf.main_bar_y)
        self.inp_spac_x.setValue(reinf.spacing_x_mm)
        self.inp_spac_y.setValue(reinf.spacing_y_mm)
        
        # Default envelope loads
        self.inp_pu.setValue(1000.0)
        self.inp_mux.setValue(50.0)
        self.inp_muy.setValue(0.0)
        
    def get_current_state(self) -> eng.DesignState:
        mat = eng.Material(
            fc_MPa=self.inp_fc.value(),
            fy_MPa=self.inp_fy.value(),
            gamma_conc_kN_m3=self.inp_gamma_c.value()
        )
        
        geom = eng.FootingGeometry(
            footing_length_x_mm=self.inp_Lx.value(),
            footing_width_y_mm=self.inp_Ly.value(),
            footing_thickness_mm=self.inp_h.value(),
            footing_embedment_depth_mm=self.inp_embed.value(),
            soil_above_footing_mm=self.inp_soil_h.value(),
            column_bx_mm=self.inp_cbx.value(),
            column_by_mm=self.inp_cby.value(),
            column_shape=self.inp_col_shape.currentText().lower(),
            column_location=self.inp_col_loc.currentText(),
            allowable_bearing_kPa=self.inp_qa.value(),
            gamma_soil_kN_m3=self.inp_gamma_s.value(),
            friction_coefficient=self.inp_fric.value(),
            passive_pressure_kPa=self.inp_passive.value(),
        )
        
        reinf = eng.Reinforcement(
            main_bar_x=self.inp_bar_x.currentText(),
            main_bar_y=self.inp_bar_y.currentText(),
            spacing_x_mm=self.inp_spac_x.value(),
            spacing_y_mm=self.inp_spac_y.value()
        )
        
        c_type = "Service" if self.chk_service.isChecked() else "Ultimate"
        load = eng.LoadCase(
            name="Manual",
            case_type=c_type,
            Pu_kN=self.inp_pu.value(),
            Mux_kNm=self.inp_mux.value(),
            Muy_kNm=self.inp_muy.value(),
            Vux_kN=self.inp_vux.value(),
            Vuy_kN=self.inp_vuy.value(),
        )
        
        return eng.DesignState(mat, geom, reinf, load)

    @Slot()
    def run_design(self):
        self.calc_btn.setEnabled(False)
        self.progress_bar.show()
        
        state = self.get_current_state()
        self.current_state = state
        
        idx = self.load_source.currentIndex()
        if idx == 0:
            # Manual batch
            df = self.manual_model.dataframe
            self.worker = DesignWorker(state, load_cases_df=df, mode="manual_batch")
        else:
            # SAP batch
            if self.sap_df.empty:
                QMessageBox.warning(self, "Warning", "No SAP2000 data fetched.")
                self.calc_btn.setEnabled(True)
                self.progress_bar.hide()
                return
            self.worker = DesignWorker(state, sap_df=self.sap_df, sap_groups_df=self.sap_groups_model.dataframe, mode="sap_batch")
            
        self.worker.finished.connect(self.on_design_finished)
        self.worker.error.connect(self.on_design_error)
        self.worker.start()

    @Slot(dict)
    def on_design_finished(self, results_payload: dict):
        self.calc_btn.setEnabled(True)
        self.progress_bar.hide()
        
        if "single_run" in results_payload:
            res = results_payload["single_run"]
            self.batch_results = []
            self.result_group_combo.blockSignals(True)
            self.result_group_combo.clear()
            self.result_group_combo.blockSignals(False)
            self._update_tables(self.current_state, res)
            self._update_plots(self.current_state, res)
        elif "items" in results_payload:
            # Batch mode
            items = results_payload["items"]
            if not items:
                QMessageBox.warning(self, "Warning", "No items designed in batch.")
                return
            
            self.batch_results = items
            self.result_group_combo.blockSignals(True)
            self.result_group_combo.clear()
            for item in items:
                self.result_group_combo.addItem(item.get('group', 'Unknown'))
            self.result_group_combo.blockSignals(False)
            
            if self.result_group_combo.count() > 0:
                self.result_group_combo.setCurrentIndex(0)
                self._on_result_group_changed(0)

    def _on_result_group_changed(self, idx):
        if not hasattr(self, 'batch_results') or not self.batch_results or idx < 0 or idx >= len(self.batch_results):
            return

        res = self.batch_results[idx]
        self.lbl_status.setText(f"Batch completed. Showing {res.get('group', 'Unknown')}...")
        srv_state = res.get("srv_state", res["state"])   # fall back to ult state if not present
        self._update_tables(res["state"], res["results"], srv_state=srv_state)
        self._update_plots(res["state"], res["results"])

    @Slot(str)
    def on_design_error(self, err_msg: str):
        self.calc_btn.setEnabled(True)
        self.progress_bar.hide()
        QMessageBox.critical(self, "Calculation Error", err_msg)

    def _update_tables(self, state: eng.DesignState, results: dict, srv_state: eng.DesignState = None):
        # srv_state = governing SERVICE load case state (CS-xx)
        # state     = governing ULTIMATE load case state (CU-xx, used for RC strength)
        if srv_state is None:
            srv_state = state   # single-run mode: same state used for everything

        # 1. Dashboard summary
        sum_dict = vdata.metric_summary(srv_state, results)
        df_sum = pd.DataFrame([{"Metric": k, "Value": v} for k, v in sum_dict.items()])
        self.tbl_summary.setModel(PandasTableModel(df_sum))
        
        # Check overall status to color label
        status = sum_dict.get("Status", "UNKNOWN")
        if status == "PASS":
            self.lbl_status.setText("Design: PASS")
            self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")
        elif status == "NEAR":
            self.lbl_status.setText("Design: NEAR LIMIT")
            self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold; color: orange;")
        else:
            self.lbl_status.setText("Design: FAIL")
            self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold; color: red;")
            
        # 2. Soil Pressure (use SERVICE state)
        df_soil = vdata.soil_pressure_dataframe(srv_state, results)
        self.tbl_soil.setModel(PandasTableModel(df_soil))
        
        # 3. Structural Checks
        df_chk = vdata.checks_dataframe(results)
        self.tbl_checks.setModel(PandasTableModel(df_chk))
        
        # 4. Flexure (use ULTIMATE state for RC design)
        df_flex = vdata.flexural_summary_dataframe(state, results)
        self.tbl_flex.setModel(PandasTableModel(df_flex))
        
        # 5. Details (use ULTIMATE state)
        df_det = vdata.detail_summary_dataframe(state, results)
        self.tbl_details.setModel(PandasTableModel(df_det))
        
        # 6. Calculation Step-by-Step (pass both states)
        html_str = vdata.calculation_html(state, results, srv_state=srv_state)
        self.calculation_text.setHtml(html_str)
        
    def _update_plots(self, state: eng.DesignState, results: dict):
        self.ax_plan.clear()
        self._draw_plan_on_ax(state, self.ax_plan)
        self.canvas_plan.draw()
        
        self.ax_sec.clear()
        self._draw_section_on_ax(state, self.ax_sec)
        self.canvas_sec.draw()

    def _draw_plan_on_ax(self, state: eng.DesignState, ax):
        from matplotlib.patches import Rectangle
        geom = state.geometry
        Lx = geom.footing_length_x_mm
        Ly = geom.footing_width_y_mm
        
        # Footing boundary
        rect = Rectangle((-Lx/2, -Ly/2), Lx, Ly, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Column
        bx, by = eng.get_loaded_area_dims(geom)
        col = Rectangle((-bx/2, -by/2), bx, by, fill=True, facecolor='gray', edgecolor='black')
        ax.add_patch(col)
        
        # Punching Perimeter
        d_avg = (state.effective_depth_x_mm + state.effective_depth_y_mm)/2
        if d_avg == 0: d_avg = geom.footing_thickness_mm - 75 # Fallback
        bo_x, bo_y = bx + d_avg, by + d_avg
        punch = Rectangle((-bo_x/2, -bo_y/2), bo_x, bo_y, fill=False, edgecolor='red', linestyle='--')
        ax.add_patch(punch)
        
        ax.set_aspect('equal')
        ax.set_xlim(-Lx/2 - 500, Lx/2 + 500)
        ax.set_ylim(-Ly/2 - 500, Ly/2 + 500)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title("Footing Plan View")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

    def _draw_section_on_ax(self, state: eng.DesignState, ax):
        from matplotlib.patches import Rectangle
        geom = state.geometry
        h = geom.footing_thickness_mm
        L = geom.footing_length_x_mm
        bx = eng.get_loaded_area_dims(geom)[0]
        
        # Footing
        rect = Rectangle((-L/2, 0), L, h, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Column/Pedestal projection
        col = Rectangle((-bx/2, h), bx, 500, fill=True, facecolor='gray')
        ax.add_patch(col)
        
        # Ground line
        ax.axhline(y=h + geom.soil_above_footing_mm, color='brown', linestyle='--', label="Ground")
        
        ax.set_aspect('equal')
        ax.set_xlim(-L/2 - 500, L/2 + 500)
        ax.set_ylim(-100, h + geom.soil_above_footing_mm + 600)
        ax.axis('off')
        ax.set_title("Section X-X")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

