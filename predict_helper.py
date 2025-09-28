
import joblib, numpy as np, pandas as pd
def load_model(path_or_bytes): return joblib.load(path_or_bytes)
def predict_df(pipe, df_in: pd.DataFrame) -> pd.DataFrame:
    proba = pipe.predict_proba(df_in)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    out = df_in.copy()
    out["risk_probability"] = np.round(proba, 4)
    out["risk_predicted"] = pred
    out["risk_level_predicted"] = np.where(out["risk_probability"] >= 0.67, "High",
                                   np.where(out["risk_probability"] >= 0.40, "Medium", "Low"))
    return out
