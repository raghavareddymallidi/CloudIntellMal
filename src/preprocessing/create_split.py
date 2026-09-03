import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os


LABEL_FILE = "data/raw/hv_double_line_90kv_labels.csv"
OUTPUT_DIR = "data/processed"


# Load metadata
df = pd.read_csv(LABEL_FILE)

# Keep episode IDs and fault types
episodes = df[["sample_id", "sc_type"]].copy()


# 70% train, 30% temporary
train, temp = train_test_split(
    episodes,
    test_size=0.30,
    stratify=episodes["sc_type"],
    random_state=42
)


# Split remaining 30% equally
val, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["sc_type"],
    random_state=42
)


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Save split IDs
splits = {
    "train": train["sample_id"].tolist(),
    "validation": val["sample_id"].tolist(),
    "test": test["sample_id"].tolist()
}


with open(
    os.path.join(OUTPUT_DIR, "episode_split.json"),
    "w"
) as f:
    json.dump(splits, f, indent=2)


# Display results
print("Episode split created successfully.")
print()
print("Train episodes:", len(train))
print("Validation episodes:", len(val))
print("Test episodes:", len(test))


print("\nTrain class distribution:")
print(train["sc_type"].value_counts().sort_index())


print("\nValidation class distribution:")
print(val["sc_type"].value_counts().sort_index())


print("\nTest class distribution:")
print(test["sc_type"].value_counts().sort_index())