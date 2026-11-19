from __future__ import annotations

from pathlib import Path

from backbone_eval import load, rank


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "backbones.txt"


def test_ranking_is_deterministic() -> None:
    weights, models = load(artifact_path())
    ranking = rank(weights, models)

    assert len(ranking) == 3
    assert ranking == rank(weights, models)
