import pandas as pd
import os


LABEL_FILE = "data/raw/hv_double_line_90kv_labels.csv"
DATA_DIR = "data/raw/hv_double_line_90kv_preprocessed_data"

SAMPLE_ID = 0
WINDOW_SIZE = 256


# Load metadata
labels = pd.read_csv(LABEL_FILE)

metadata = labels[labels["sample_id"] == SAMPLE_ID].iloc[0]


# Load waveform
waveform_file = os.path.join(
    DATA_DIR,
    f"{SAMPLE_ID}_sample_hv_double_line_90kv.pkl"
)

waveform = pd.read_pickle(waveform_file)


# Separate time and signals
time = waveform["time_s"].values

signal_columns = [
    column for column in waveform.columns
    if column != "time_s"
]

signals = waveform[signal_columns].values


# Sampling information
sampling_interval = time[1] - time[0]
sampling_rate = 1 / sampling_interval

print("Sampling rate:", sampling_rate, "Hz")
print("Total samples:", len(time))
print("Number of channels:", len(signal_columns))


# Fault start index
fault_start_time = metadata["t_evnt_start"]

fault_start_index = abs(time - fault_start_time).argmin()

print("\nFault start time:", fault_start_time)
print("Fault start index:", fault_start_index)


# ------------------------------------------------
# Pre-fault window
# ------------------------------------------------

prefault_start = fault_start_index - WINDOW_SIZE
prefault_end = fault_start_index

prefault_window = signals[prefault_start:prefault_end]


# ------------------------------------------------
# Fault window
# ------------------------------------------------

fault_start = fault_start_index
fault_end = fault_start_index + WINDOW_SIZE

fault_window = signals[fault_start:fault_end]


# ------------------------------------------------
# Convert to TCN format
# ------------------------------------------------

prefault_tcn = prefault_window.T
fault_tcn = fault_window.T


# ------------------------------------------------
# Display results
# ------------------------------------------------

print("\nPre-fault window")
print("Shape:", prefault_window.shape)
print("TCN shape:", prefault_tcn.shape)

print("\nFault window")
print("Shape:", fault_window.shape)
print("TCN shape:", fault_tcn.shape)


# ------------------------------------------------
# Labels
# ------------------------------------------------

print("\nDetection labels:")
print("Pre-fault:", 0)
print("Fault:", 1)

print("\nFault classification label:")
print("sc_type:", metadata["sc_type"])

print("\nPreprocessing test completed successfully.")