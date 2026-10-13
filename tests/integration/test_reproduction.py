import numpy as np
import pytest

from tuplebench import Defense, evaluate_primary, evaluate_refusal
from tuplebench.defenses.quantizer import verify_release
from tuplebench.defenses.training import fit_wcpq


def test_primary_smoke_matches_frozen_domain_results(release_root):
    report = evaluate_primary(release_root, models=["typesafe"], domains=["delivery_operations"])
    actual = {r["release"]: r["recovery"]["mean"] for r in report["recovery"]}
    assert actual == pytest.approx({"public_only": 0.5, "choice": 51 / 96,
                                    "choice_confidence": 0.5, "full_probabilities": 59 / 96,
                                    "full_native": 60 / 96})
    assert report["API_calls"] == 0


@pytest.mark.parametrize("model", ["typesafe", "openjev", "clef"])
@pytest.mark.parametrize("domain", ["delivery_operations", "equipment_service", "it_operations"])
def test_strict_release_preserves_all_hypothetical_workflow_costs(benchmark, model, domain):
    defense = Defense(benchmark, model, domain)
    case = benchmark.cases(domain)[0]
    original = benchmark.responses(model)[case["case_id"]]["response"]
    released = defense.release(original, case["consumer_view"])
    audit = verify_release(original, released, case["consumer_view"], defense.domain,
                            defense.policies, defense.floor, None)
    assert audit["passed"]
    assert audit["same_signature"]
    assert released["choice"] == original["choice"]


def test_refusal_counts_match_all_nine_final_exports(release_root):
    expected = {"typesafe": (891, 548, 874), "openjev": (0, 1396, 0), "jevk5": (1333, 107, 1216),
                "clef": (1137, 303, 1126), "clef_flash": (465, 942, 407), "decider": (1102, 338, 1101),
                "sage": (1425, 15, 1400), "d1": (878, 545, 807), "mercury_decide": (1410, 30, 1403)}
    report = evaluate_refusal(release_root)
    for row in report["results"]:
        assert (row["withheld"], row["direct_correct"], row["withheld_correct"]) == expected[row["model"]]


def test_strict_codebook_rebuild_matches_frozen_codes_without_test_access(benchmark, monkeypatch):
    original = benchmark.cases

    def training_only(domain=None, split="test"):
        assert split == "train", "Defense fitting requested a held-out split"
        return original(domain, split)

    monkeypatch.setattr(benchmark, "cases", training_only)
    rebuilt = fit_wcpq(benchmark, "typesafe", "delivery_operations")
    frozen = Defense(benchmark, "typesafe", "delivery_operations")
    np.testing.assert_array_equal(rebuilt["codebooks"]["strict_expanded"]["codes"],
                                  frozen.package["codebooks"]["strict_expanded"]["codes"])
    assert rebuilt["test_used_for_training"] is False


@pytest.mark.parametrize("model", ["typesafe", "openjev", "clef"])
@pytest.mark.parametrize("domain", ["delivery_operations", "equipment_service", "it_operations"])
@pytest.mark.parametrize("method", ["privacy_unpaired_zero", "privacy_paired_zero",
                                    "privacy_unpaired_low", "privacy_paired_low",
                                    "privacy_unpaired_high", "privacy_paired_high"])
def test_learned_release_respects_every_hypothetical_truth_budget(benchmark, model, domain, method):
    defense = Defense(benchmark, model, domain)
    case = benchmark.cases(domain)[0]
    original = benchmark.responses(model)[case["case_id"]]["response"]
    released = defense.release(original, case["consumer_view"], method)
    fraction = defense.package["codebooks"][method]["fraction"]
    audit = verify_release(original, released, case["consumer_view"], defense.domain,
                            defense.policies, defense.floor, fraction)
    assert audit["passed"]
    assert set(released) == {"choice", "probabilities", "confidence"}
