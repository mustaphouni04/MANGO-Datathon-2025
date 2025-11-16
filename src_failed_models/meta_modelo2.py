import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _repo_root() -> str:
    # src/ -> project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_preds(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # Normalize expected columns
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id", "ID" if "ID" in df.columns else None)
    pred_col = cols.get("production", "Production" if "Production" in df.columns else None)
    if id_col is None or pred_col is None:
        raise ValueError(f"File {file_path} must contain 'ID' and 'Production' columns")
    out = df[[id_col, pred_col]].copy()
    out.columns = ["ID", "Production"]
    return out


def main() -> None:
    root = _repo_root()

    # Primary attempt: read from project root where predictions are saved
    paths = [
        os.path.join(root, "predictions_lgb.csv"),
        os.path.join(root, "predictions_xgb.csv"),
        os.path.join(root, "predictions_nn.csv"),
    ]

    # Fallback: try current working directory
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            alt = os.path.basename(p)
            if os.path.exists(alt):
                paths[i] = alt

    lgb_df = _load_preds(paths[0])
    xgb_df = _load_preds(paths[1])
    ctb_df = _load_preds(paths[2])

    # Merge on ID to align rows
    merged = lgb_df.merge(xgb_df, on="ID", suffixes=("_lgb", "_xgb")).merge(
        ctb_df.rename(columns={"Production": "Production_catboost"}), on="ID"
    )

    # X = predictions from the 3 base models
    X = merged[["Production_lgb", "Production_xgb", "Production_catboost"]].to_numpy(dtype=np.float64)

    # Pseudo-target: average of base predictions (simple, stable in absence of ground truth)
    y = X.mean(axis=1)

    # Standardize features for Ridge stability
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    # Fit Ridge (simple alpha; tune if you later have validation targets)
    ridge = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
    ridge.fit(X_scaled, y)

    # Predict blended values
    blended = ridge.predict(X_scaled)

    # Output
    out = pd.DataFrame({
        "ID": merged["ID"].values,
        "Production": blended
    })

    out_path = os.path.join(root, "predictions_meta_ridge.csv")
    out.to_csv(out_path, index=False)

    # Print brief info
    coefs = ridge.coef_
    print("Ridge coefficients (on standardized features):", coefs)
    print("Ridge intercept:", ridge.intercept_)
    print(f"Saved blended predictions to {out_path}")


if __name__ == "__main__":
    main()

