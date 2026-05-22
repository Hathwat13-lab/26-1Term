import os
import pandas as pd

base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_path = os.path.join(base, "tfim_train_L8_12_16_24_32.csv")
test_path  = os.path.join(base, "tfim_test_L48_64_96_128_inf.csv")

for path, label in [(train_path, 'Train'), (test_path, 'Test')]:
    print(f"\n--- {label}: {path} ---")
    if not os.path.isfile(path):
        print("MISSING")
        continue
    df = pd.read_csv(path)
    print("shape:", df.shape)
    unique_invL = sorted(df['inv_L'].unique().tolist())
    print("unique inv_L:", unique_invL)

    group_counts = df.groupby(['h','T'])['inv_L'].count()
    unique_group_counts = sorted(group_counts.unique().tolist())
    print("groupby(['h','T'])['inv_L'].count().unique():", unique_group_counts)
    if len(unique_group_counts) == 1:
        print("Balance check: single value ->", unique_group_counts[0])
    else:
        print("Balance check: NOT single-valued -> values:", unique_group_counts)
