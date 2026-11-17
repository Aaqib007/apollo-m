"""Smoke tests — confirm pipeline components load and run without crashing."""
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent

def test_outputs_exist():
    assert (ROOT / "outputs" / "eda_report.txt").exists()
    assert (ROOT / "outputs" / "metrics.json").exists()
    assert (ROOT / "outputs" / "meso_report.csv").exists()
    assert (ROOT / "outputs" / "clusters_report.csv").exists()

def test_comments_csv_loads():
    df = pd.read_csv(ROOT / "data" / "apollo_comments.csv", nrows=50)
    assert len(df) == 50
    assert "body" in df.columns
    assert "toxicity_score" in df.columns

def test_daily_csv_loads():
    df = pd.read_csv(ROOT / "data" / "apollo_daily.csv")
    assert len(df) > 0
    assert "subreddit" in df.columns
    assert "avg_toxicity" in df.columns
