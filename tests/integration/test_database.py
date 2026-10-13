from collections import Counter, defaultdict
import json
import sqlite3

import pytest


def test_frozen_counts_pair_splits_and_training_folds(benchmark):
    assert Counter(c["evaluator_only"]["split"] for c in benchmark.dataset.cases) == {
        "train": 864, "validation": 288, "test": 288,
    }
    groups = defaultdict(list)
    for case in benchmark.dataset.cases:
        groups[case["evaluator_only"]["base_id"]].append(case)
    assert len(groups) == 720
    for pair in groups.values():
        assert len(pair) == 2
        left, right = pair
        assert left["consumer_view"] == right["consumer_view"]
        assert left["evaluator_only"]["truth"]["T"] == right["evaluator_only"]["truth"]["T"]
        assert left["evaluator_only"]["truth"]["S"] != right["evaluator_only"]["truth"]["S"]
        assert left["evaluator_only"]["split"] == right["evaluator_only"]["split"]
        if left["evaluator_only"]["split"] == "train":
            assert benchmark.folds[left["case_id"]] == benchmark.folds[right["case_id"]]


def test_sqlite_mirror_and_public_view_exclusion(benchmark, release_root):
    database = sqlite3.connect("file:" + str(release_root / "data/benchmark.sqlite3") + "?mode=ro", uri=True)
    try:
        assert database.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert database.execute("SELECT COUNT(*) FROM worlds").fetchone()[0] == 1440
        assert database.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == 720
        for cid, model, public, evaluator in database.execute("SELECT case_id,model_view,consumer_view,evaluator_only FROM worlds"):
            case = benchmark.dataset.case_map[cid]
            assert json.loads(model) == case["model_view"]
            assert json.loads(public) == case["consumer_view"]
            assert json.loads(evaluator) == case["evaluator_only"]
        columns = [r[1] for r in database.execute("PRAGMA table_info(public_cases)")]
        assert columns == ["case_id", "domain_id", "consumer_view"]
    finally:
        database.close()


@pytest.mark.parametrize("model", ["typesafe", "openjev", "jevk5", "clef", "clef_flash", "decider", "sage", "d1", "mercury_decide"])
def test_all_final_model_responses_validate(benchmark, model):
    responses = benchmark.responses(model)
    assert len(responses) == 1440
    assert all(r["status"] == "completed" for r in responses.values())
