"""Sanity tests — do outputs make sense?"""
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent

def test_toxicity_rate_nonzero():
    df = pd.read_csv(ROOT / "data" / "apollo_daily.csv")
    assert df["avg_toxicity"].mean() > 0.0, "avg_toxicity is 0 in daily aggregates"

def test_toxicity_rate_realistic():
    df = pd.read_csv(ROOT / "data" / "apollo_daily.csv")
    rate = df["toxic_rate"].mean()
    assert 0.01 < rate < 0.5, f"Toxic rate {rate:.2%} outside realistic range"

def test_multiple_subreddits():
    df = pd.read_csv(ROOT / "data" / "apollo_comments.csv")
    assert df["subreddit"].nunique() >= 10

def test_meso_has_communities():
    df = pd.read_csv(ROOT / "outputs" / "meso_report.csv")
    assert len(df) >= 10

def test_clusters_report_valid():
    df = pd.read_csv(ROOT / "outputs" / "clusters_report.csv")
    assert len(df) > 0
    assert df["cluster"].nunique() >= 2
