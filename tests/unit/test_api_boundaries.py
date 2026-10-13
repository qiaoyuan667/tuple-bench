import copy

import numpy as np
import pytest

from tuplebench import Defense, TypedDecisionClient
from tuplebench.attacks.strong_attacker_v1 import checked
from tuplebench.core.io import ContractError
from tuplebench.defenses.probability import inverse_temperature, transform


@pytest.fixture
def sample(benchmark):
    case = benchmark.cases("delivery_operations")[0]
    return case, benchmark.responses("typesafe")[case["case_id"]]["response"]


@pytest.mark.parametrize("field", ["T", "S", "hypotheses", "case_id", "paired_response", "evaluator_only"])
def test_defense_rejects_private_context(benchmark, sample, field):
    case, response = sample
    defense = Defense(benchmark, "typesafe", case["domain_id"])
    with pytest.raises(ContractError):
        defense.release(response, dict(case["consumer_view"], **{field: "forbidden"}))


@pytest.mark.parametrize("field", ["S", "case_id", "private_evidence", "paired_response"])
def test_attacker_rejects_undeclared_inputs(sample, field):
    case, response = sample
    row = dict(consumer_view=case["consumer_view"], payload=response,
               T=case["evaluator_only"]["truth"]["T"], hypotheses=case["evaluator_only"]["hypotheses"])
    with pytest.raises(ContractError):
        checked(dict(row, **{field: "forbidden"}))


def test_temperature_inverse_preserves_native_numbers(sample):
    case, response = sample
    order = [c["id"] for c in case["model_view"]["question"]["candidates"]]
    observed = transform(response, order, "temperature_2")
    restored = inverse_temperature(observed)
    np.testing.assert_allclose([restored["probabilities"][label] for label in order],
                               [response["probabilities"][label] for label in order], atol=1e-12, rtol=1e-12)
    assert observed["choice"] == response["choice"]
    assert observed["confidence"] == response["confidence"]


def test_quantization_is_simplex_and_keeps_choice_confidence(sample):
    case, response = sample
    order = [c["id"] for c in case["model_view"]["question"]["candidates"]]
    observed = transform(response, order, "quantized_10")
    values = list(observed["probabilities"].values())
    assert sum(values) == pytest.approx(1)
    assert all(v * 10 == pytest.approx(round(v * 10)) for v in values)
    assert observed["choice"] == response["choice"]
    assert observed["confidence"] == response["confidence"]


def test_http_adapter_sends_no_evaluator_fields(sample):
    case, response = sample
    sent = []

    def transport(body):
        sent.append(copy.deepcopy(body))
        return dict(model="test-model", answers={"coarse_cause": dict(type="choice", **response)})

    client = TypedDecisionClient("https://example.invalid/decisions", "test-model", transport=transport)
    assert client.predict(case["model_view"]) == response
    assert len(sent) == 1
    assert set(sent[0]) == {"model", "state", "questions"}
    assert "evaluator_only" not in str(sent[0])
    with pytest.raises(ContractError):
        client.predict(dict(case["model_view"], evaluator_only=case["evaluator_only"]))


def test_http_adapter_does_not_retry(sample):
    case, _ = sample
    calls = []

    def fail(body):
        calls.append(body)
        raise RuntimeError("transport interrupted")

    client = TypedDecisionClient("https://example.invalid/decisions", "test-model", transport=fail)
    with pytest.raises(RuntimeError):
        client.predict(case["model_view"])
    assert len(calls) == 1


def test_api_selector_and_returned_version_can_be_pinned_separately(sample):
    case, response = sample

    def transport(body):
        assert body["model"] == "requested-selector"
        return dict(model="pinned-version", answers={"coarse_cause": dict(type="choice", **response)})

    client = TypedDecisionClient("https://example.invalid/decisions", "requested-selector",
                                 expected_version="pinned-version", transport=transport)
    assert client.predict(case["model_view"]) == response
