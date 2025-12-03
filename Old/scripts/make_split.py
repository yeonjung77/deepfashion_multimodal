import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = "../data/processed/labels.csv"
df = pd.read_csv(csv_path)

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_df.to_csv("../data/processed/train.csv", index=False)
val_df.to_csv("../data/processed/val.csv", index=False)

print("Train:", len(train_df))
print("Val:", len(val_df))
