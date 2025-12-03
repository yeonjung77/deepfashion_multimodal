import os
import pandas as pd

# -----------------------------
# 1) Load Label TXT files
# -----------------------------
def load_shape_labels(path):
    d = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            img = parts[0]
            values = list(map(int, parts[1:]))
            d[img] = values
    return d

def load_fabric_labels(path):
    d = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            img = parts[0]
            values = list(map(int, parts[1:]))
            d[img] = values
    return d

def load_pattern_labels(path):
    d = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            img = parts[0]
            values = list(map(int, parts[1:]))
            d[img] = values
    return d

# -----------------------------
# 2) Main Processing
# -----------------------------
def main():

    IMAGE_ROOT = "../data/raw/images"
    SHAPE_TXT  = "../data/raw/labels/shape/shape_anno_all.txt"
    FABRIC_TXT = "../data/raw/labels/texture/fabric_ann.txt"
    PATTERN_TXT= "../data/raw/labels/texture/pattern_ann.txt"

    print("Loading label files...")
    shape_dict = load_shape_labels(SHAPE_TXT)
    fabric_dict = load_fabric_labels(FABRIC_TXT)
    pattern_dict = load_pattern_labels(PATTERN_TXT)

    print(f"shape labels:   {len(shape_dict)}")
    print(f"fabric labels:  {len(fabric_dict)}")
    print(f"pattern labels: {len(pattern_dict)}")

    print("Scanning image folder...")
    all_images = sorted([f for f in os.listdir(IMAGE_ROOT) if f.endswith(".jpg")])
    print("Total images:", len(all_images))

    rows = []

    for img in all_images:
        shape = shape_dict.get(img, [0]*12)
        fabric = fabric_dict.get(img, [0]*3)
        pattern = pattern_dict.get(img, [0]*3)

        row = {"image_name": img}

        # Expand attributes
        for i, v in enumerate(shape):
            row[f"shape_{i}"] = v
        for i, v in enumerate(fabric):
            row[f"fabric_{i}"] = v
        for i, v in enumerate(pattern):
            row[f"pattern_{i}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("../data/processed", exist_ok=True)
    out_csv = "../data/processed/labels.csv"
    df.to_csv(out_csv, index=False)

    print(f"CSV Created Successfully: {out_csv}")
    print("Total rows:", len(df))

if __name__ == "__main__":
    main()
