"""SAP2000 OAPI Integration for Spread Footing Designer."""

import pandas as pd

try:
    import comtypes.client
except ImportError:
    comtypes = None


def _extract_arrays(ret):
    """Extract only tuple/list items from a COM return value."""
    return [item for item in ret if isinstance(item, (tuple, list))]


def fetch_selected_joint_reactions() -> pd.DataFrame:
    """
    Connects to an active SAP2000 instance, gets the selected joints,
    and returns their joint reactions in a Pandas DataFrame.
    """
    if comtypes is None:
        raise ImportError("comtypes package is not installed. Cannot connect to SAP2000 OAPI.")

    try:
        SapObject = comtypes.client.GetActiveObject("CSI.SAP2000.API.SapObject")
    except Exception as e:
        err_str = str(e)
        if "-2147221021" in err_str or "unavailable" in err_str.lower():
            msg = (
                "COM Operation Unavailable (-2147221021).\n\n"
                "SAP2000 is open, but Python cannot attach to it due to Windows COM security.\n\n"
                "Fixes:\n"
                "1. Privilege Mismatch: Ensure BOTH SAP2000 and this app are running at the SAME permission level "
                "(e.g. neither as Administrator, or both as Administrator).\n"
                "2. Registration: Close SAP2000, then right-click SAP2000 and 'Run as Administrator' once. Close it, then open normally."
            )
            raise ConnectionError(msg)
        else:
            raise ConnectionError(f"Could not connect to SAP2000. Ensure SAP2000 is open. Error: {e}")

    SapModel = SapObject.SapModel

    # Check if the model is locked/analyzed
    if not SapModel.GetModelIsLocked():
        raise RuntimeError("SAP2000 model is not locked. Please run analysis first.")

    # ── Get Selected Joints ──────────────────────────────────────────────────
    ret = SapModel.SelectObj.GetSelected()
    arrays = _extract_arrays(ret)
    if len(arrays) == 2:
        # Determine which array contains strings (ObjectName) vs integers (ObjectType)
        if len(arrays[0]) > 0 and isinstance(arrays[0][0], str):
            ObjectName, ObjectType = arrays[0], arrays[1]
        else:
            ObjectType, ObjectName = arrays[0], arrays[1]
    else:
        raise ValueError(f"Unexpected array count from GetSelected: {len(arrays)}")

    NumberItems = len(ObjectName)
    if NumberItems == 0:
        raise ValueError("No objects are currently selected in SAP2000.")

    selected_joints = []
    for i in range(NumberItems):
        if int(ObjectType[i]) == 1:  # 1 = Point/Joint object
            selected_joints.append(str(ObjectName[i]))

    if not selected_joints:
        raise ValueError(
            f"No JOINTS are currently selected. Found {NumberItems} objects of types: {ObjectType}. "
            "Please select the restraint joint nodes (small blue crosses) in SAP2000 and try again."
        )

    # ── Fetch Reactions Per Joint (ItemType=0 = Object, most reliable) ───────
    all_rows = []
    for joint_name in selected_joints:
        result = SapModel.Results.JointReact(joint_name, 0)
        arrays = _extract_arrays(result)

        if len(arrays) == 11:
            # Standard: Obj, Elm, ACase, StepType, StepNum, F1, F2, F3, M1, M2, M3
            Obj, Elm, ACase, StepType, StepNum, F1, F2, F3, M1, M2, M3 = arrays
        elif len(arrays) == 10:
            # Without Elm column
            Obj, ACase, StepType, StepNum, F1, F2, F3, M1, M2, M3 = arrays
        else:
            raise ValueError(
                f"JointReact returned {len(arrays)} arrays for joint '{joint_name}'. "
                f"Raw result length={len(result)}. "
                f"Types: {[type(x).__name__ for x in result]}"
            )

        for i in range(len(Obj)):
            all_rows.append({
                "Joint":      str(Obj[i]),
                "OutputCase": str(ACase[i]),
                "StepType":   str(StepType[i]),
                "StepNum":    StepNum[i],
                "F1 (kN)":    F1[i],
                "F2 (kN)":    F2[i],
                "F3 (kN)":    F3[i],
                "M1 (kN-m)":  M1[i],
                "M2 (kN-m)":  M2[i],
                "M3 (kN-m)":  M3[i],
            })

    if not all_rows:
        raise ValueError(
            "No reaction results found for the selected joints. "
            "Ensure analysis has been run and load cases are set for output."
        )

    df = pd.DataFrame(all_rows)
    numeric_cols = ["StepNum", "F1 (kN)", "F2 (kN)", "F3 (kN)", "M1 (kN-m)", "M2 (kN-m)", "M3 (kN-m)"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
