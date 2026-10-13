"""TUPLE-Bench evaluation APIs."""
import os

# Keep small numerical fits reproducible and avoid uncontrolled BLAS oversubscription.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

from .api import Benchmark, Defense, evaluate_defenses, evaluate_primary, evaluate_response_file
from .client import TypedDecisionClient
from .refusal import evaluate_refusal
from .verification import verify_release

__all__ = ["Benchmark", "Defense", "TypedDecisionClient", "evaluate_primary", "evaluate_defenses", "evaluate_refusal", "evaluate_response_file", "verify_release"]
