import pandas as pd
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "problems.jsonl")

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

df = pd.DataFrame(data)

print("Columns:", df.columns)
print("\nFirst 5 rows:\n", df.head())
print("\nTotal samples:", len(df))
