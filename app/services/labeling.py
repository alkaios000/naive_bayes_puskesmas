from typing import Dict, Any, List
import pandas as pd

def compute_trend_label(kasus_t: int, kasus_prev: int, threshold_pct: float = 0.10, min_abs_change: int = 3) -> str:
    """
    Menghitung label tren (Naik, Stabil, Turun) sesuai rumus PRD Bagian 5.1:
    delta = (kasus_t - kasus_{t-1}) / max(kasus_{t-1}, 1)
    - Naik jika delta > +threshold_pct dan (kasus_t - kasus_{t-1}) >= min_abs_change
    - Turun jika delta < -threshold_pct dan (kasus_{t-1} - kasus_t) >= min_abs_change
    - Stabil selain itu
    """
    diff = kasus_t - kasus_prev
    denom = max(kasus_prev, 1)
    delta = diff / denom

    if delta > threshold_pct and diff >= min_abs_change:
        return "Naik"
    elif delta < -threshold_pct and (-diff) >= min_abs_change:
        return "Turun"
    else:
        return "Stabil"

def generate_labels_for_dataframe(df: pd.DataFrame, threshold_pct: float = 0.10, min_abs_change: int = 3) -> pd.DataFrame:
    """
    Menerapkan pembentukan label tren untuk setiap penyakit secara kronologis
    """
    df_sorted = df.sort_values(by=["kode_icd10", "tahun", "bulan"]).copy()
    
    labels: List[Any] = []
    deltas: List[Any] = []
    
    grouped = df_sorted.groupby("kode_icd10")
    for icd, group in grouped:
        prev_cases = None
        for idx, row in group.iterrows():
            curr_cases = int(row["jumlah_kasus"])
            if prev_cases is None:
                labels.append(None)
                deltas.append(0.0)
            else:
                lbl = compute_trend_label(curr_cases, prev_cases, threshold_pct, min_abs_change)
                denom = max(prev_cases, 1)
                delta_val = (curr_cases - prev_cases) / denom
                labels.append(lbl)
                deltas.append(delta_val)
            prev_cases = curr_cases
            
    df_sorted["label"] = labels
    df_sorted["delta_pct"] = deltas
    return df_sorted
