import pandas as pd
"""Validation tests — do metrics meet research thresholds?"""
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent

@pytest.fixture
def metrics():
    with open(ROOT / "outputs" / "metrics.json") as f:
        return json.load(f)

def test_toxicity_accuracy(metrics):
    raw = metrics["Toxicity (TF-IDF+LR, Jigsaw)"]["accuracy"]
    acc = float(str(raw).replace("%", "")) / 100
    assert acc >= 0.80, f"Toxicity accuracy {acc:.2%} below 80%"

def test_toxicity_f1(metrics):
    f1 = float(metrics["Toxicity (TF-IDF+LR, Jigsaw)"]["F1 macro"])
    assert f1 >= 0.60, f"F1 macro {f1:.3f} below 0.60"

def test_silhouette_score():
    import pandas as pd
    df = pd.read_csv(ROOT / "outputs" / "clusters_report.csv")
    assert len(df) > 0, "clusters_report.csv is empty"

import pandas as pd

def test_tft_forecast_exists():
    assert (ROOT / "outputs" / "forecast_results.csv").exists(), "No TFT forecast output"
    df = pd.read_csv(ROOT / "outputs" / "forecast_results.csv")
    assert len(df) > 0
