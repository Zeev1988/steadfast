"""Tests for visualization.summary (KB correlation helper)."""

from __future__ import annotations

import pandas as pd
import pytest

from loader import KB_COL_ORDER


@pytest.fixture
def minimal_kb_csv(tmp_path):
    rows = []
    for i in range(15):
        rows.append(
            {
                "ticket_id": f"T{i}",
                "customer_name": "Co",
                "plan": "Growth" if i % 2 == 0 else "Starter",
                "subject": "S" * (5 + i % 3),
                "body": "B" * (20 + i),
                "category": ["bug", "billing", "bug"][i % 3],
                "priority": ["low", "medium", "high"][i % 3],
                "resolution": "R" * (15 + (i % 4) * 2),
            }
        )
    df = pd.DataFrame(rows)
    path = tmp_path / "kb.csv"
    df.to_csv(path, index=False)
    return path


def test_correlation_matrix_shape(minimal_kb_csv) -> None:
    pytest.importorskip("matplotlib")
    from visualization.summary import correlation_matrix

    df = pd.read_csv(minimal_kb_csv, dtype=str, keep_default_na=False, na_filter=False)
    df = df.loc[:, list(KB_COL_ORDER)]
    corr = correlation_matrix(df)
    assert corr.shape[0] == corr.shape[1]
    assert "body_len" in corr.index


def test_main_writes_png(minimal_kb_csv, tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from visualization.summary import main

    out = tmp_path / "c.png"
    code = main(["--csv", str(minimal_kb_csv), "--output", str(out), "--top", "3"])
    assert code == 0
    assert out.exists()
    assert out.stat().st_size > 800
