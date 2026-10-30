"""Constants and column definitions for NASA C-MAPSS dataset."""

INDEX_COLS = ["unit_id", "cycle"]

SETTING_COLS = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
]

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

ALL_RAW_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Detailed descriptions of sensors for EDA and domain interpretation
SENSOR_DESCRIPTIONS = {
    "sensor_1": "Total temperature at fan inlet (°R)",
    "sensor_2": "Total temperature at LPC outlet (°R)",
    "sensor_3": "Total temperature at HPC outlet (°R)",
    "sensor_4": "Total temperature at LPT outlet (°R)",
    "sensor_5": "Pressure at fan inlet (psia)",
    "sensor_6": "Total pressure in bypass-duct (psia)",
    "sensor_7": "Total pressure at HPC outlet (psia)",
    "sensor_8": "Physical fan speed (rpm)",
    "sensor_9": "Physical core speed (rpm)",
    "sensor_10": "Engine pressure ratio (P50/P2)",
    "sensor_11": "Static pressure at HPC outlet (psia)",
    "sensor_12": "Ratio of fuel flow to Ps30 (pps/psia)",
    "sensor_13": "Corrected fan speed (rpm)",
    "sensor_14": "Corrected core speed (rpm)",
    "sensor_15": "Bypass Ratio (BPR)",
    "sensor_16": "Burner fuel-air ratio",
    "sensor_17": "Bleed Enthalpy",
    "sensor_18": "Required fan speed (rpm)",
    "sensor_19": "Required fan conversion speed (rpm)",
    "sensor_20": "High-pressure turbines cool air flow (lbm/s)",
    "sensor_21": "Low-pressure turbines cool air flow (lbm/s)",
}

# In FD001, sensors with zero or near-zero variance across all cycles:
# (Note: sensor dropping must be fitted strictly on the development training split)
ZERO_VARIANCE_SENSORS_FD001 = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]
