"""Fast tests for M11 — report generation and reproduce helpers."""

from __future__ import annotations



from arena.eval.report import (
    format_leaderboard_table,
    format_exploitability_curve,
    save_results,
    load_results,
)


class TestFormatLeaderboardTable:
    def test_empty(self):
        out = format_leaderboard_table([], title="Test")
        assert "No data" in out

    def test_basic(self):
        rows = [
            {"name": "causal_monitor", "auroc": 0.98, "exploitability": 0.69},
            {"name": "arena_blue", "auroc": 0.927, "exploitability": 0.78},
        ]
        out = format_leaderboard_table(rows, title="Leaderboard")
        assert "Leaderboard" in out
        assert "causal_monitor" in out
        assert "0.980" in out
        assert "0.690" in out

    def test_float_formatting(self):
        rows = [{"metric": 0.123456}]
        out = format_leaderboard_table(rows)
        assert "0.123" in out


class TestFormatExploitabilityCurve:
    def test_empty(self):
        out = format_exploitability_curve([])
        assert "No data" in out

    def test_basic(self):
        pts = [
            {"label": "gen0", "exploitability": 0.72, "fresh_red_start": 0.10},
            {"label": "gen1", "exploitability": 0.765, "fresh_red_start": 0.12},
        ]
        out = format_exploitability_curve(pts)
        assert "gen0" in out
        assert "gen1" in out
        assert "0.720" in out
        assert "+0.045" in out  # delta

    def test_single_point_no_delta(self):
        pts = [{"label": "gen0", "exploitability": 0.5, "fresh_red_start": 0.1}]
        out = format_exploitability_curve(pts)
        assert "—" in out  # no delta for first point


class TestSaveLoadResults:
    def test_roundtrip(self, tmp_path):
        data = {"config": "small", "sections": {"leaderboard": [{"name": "test"}]}}
        p = save_results(data, tmp_path / "results.json")
        assert p.exists()
        loaded = load_results(p)
        assert loaded == data

    def test_creates_parent(self, tmp_path):
        data = {"x": 1}
        p = save_results(data, tmp_path / "sub" / "dir" / "results.json")
        assert p.exists()
