import pytest
from app.services.labeling import compute_trend_label

def test_labeling_threshold_and_min_abs_change():
    # 1. Kasus Naik: +15 kasus dari 100 (delta = +15% > 10%, diff = 15 >= 3) -> Naik
    assert compute_trend_label(115, 100, threshold_pct=0.10, min_abs_change=3) == "Naik"

    # 2. Kasus Turun: -20 kasus dari 100 (delta = -20% < -10%, diff = -20 <= -3) -> Turun
    assert compute_trend_label(80, 100, threshold_pct=0.10, min_abs_change=3) == "Turun"

    # 3. Fluktuasi kecil pada kasus sedikit (TB): dari 2 ke 3 (+50% > 10%, tapi diff = 1 < 3) -> Stabil
    assert compute_trend_label(3, 2, threshold_pct=0.10, min_abs_change=3) == "Stabil"

    # 4. Fluktuasi kecil: dari 100 ke 105 (+5% <= 10%, diff = 5 >= 3) -> Stabil
    assert compute_trend_label(105, 100, threshold_pct=0.10, min_abs_change=3) == "Stabil"

    # 5. Nilai identik -> Stabil
    assert compute_trend_label(50, 50, threshold_pct=0.10, min_abs_change=3) == "Stabil"

    # 6. Dinamika parameter: ambang diubah ke 5% dan min_abs_change ke 1
    assert compute_trend_label(3, 2, threshold_pct=0.05, min_abs_change=1) == "Naik"
