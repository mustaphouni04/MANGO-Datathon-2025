import pandas as pd
import numpy as np


df = pd.read_csv("../data/train.csv", sep=";")
print(df.columns)

grouped = df.groupby('ID')

summary = {}

for col in df.columns:
    summary[col] = {
        "dtype": df[col].dtype,
        "num_missing": df[col].isna().sum(),
        "num_unique": df[col].nunique(),
        "example_value": df[col].iloc[0],
    }

import pprint
pprint.pprint(summary)


def column_profile(df):
    rows = []
    for col in df.columns:
        values = df[col]
        dtype = values.dtype
        missing = values.isna().sum()
        nunique = values.nunique()

        # safe example extraction
        non_null = values.dropna()
        example = non_null.iloc[0] if len(non_null) > 0 else None

        # safe constant-per-ID detection
        try:
            constant = values.groupby(df["ID"]).nunique().max() == 1
        except Exception:
            constant = None

        rows.append({
            "column": col,
            "dtype": str(dtype),
            "missing": int(missing),
            "unique": int(nunique),
            "example": example,
            "is_constant_per_ID": constant
        })

    return pd.DataFrame(rows)

profile = column_profile(df)
print(profile)

df["image_embedding"] = df["image_embedding"].apply(
    lambda x: np.fromstring(x, sep=",")
)

for product_id, product_df in grouped:
    print(product_df["image_embedding"].iloc[0].shape)
    print(f"Product ID: {product_id}")
    print(f"Number of weeks: {len(product_df)}")
    print(f"Time range: {product_df['num_week_iso'].min()} to {product_df['num_week_iso'].max()}")
    print(f"Production: {product_df['Production'].iloc[0]}")  # Constant per product

    print(product_df["id_season"])
    print(product_df["aggregated_family"])
    print(product_df["family"])
    print(product_df["category"])
    print(product_df["fabric"])
    print(product_df["color_name"])
    print(product_df["color_rgb"])
    print(product_df["length_type"])
    print(product_df["silhouette_type"])
    print(product_df["waist_type"])
    print(product_df["neck_lapel_type"])
    print(product_df["sleeve_length_type"])
    print(product_df["woven_structure"])
    print(product_df["knit_structure"]) 
    print(product_df["print_type"])
    print(product_df["archetype"])
    print(product_df["moment"])
    print(product_df["phase_in"])
    print(product_df["phase_out"])
    print(product_df["life_cycle_length"])
    print(product_df["num_stores"])
    print(product_df["num_sizes"])
    print(product_df["has_plus_sizes"])
    print(product_df["price"])
    print(product_df["year"])
    print(product_df["weekly_sales"])
    print(product_df["weekly_demand"])
    
    print("---")

    break


