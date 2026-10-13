"""Explicit HTTP API integration. No vendor CLI, subprocess or automatic retry."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .core.io import ContractError, require
from .core.native import canonical_response, to_request


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class TypedDecisionClient:
    """Caller-supplied HTTPS endpoint and credentials; one request per predict call.

    A response adapter may translate a provider envelope to the typed response
    schema. It must preserve the returned choice, probabilities and confidence.
    Tokens are read from the named environment variable and never serialized.
    """

    def __init__(self, endpoint, model, *, expected_version=None, api_key_env="MODEL_API_KEY",
                 timeout=180, response_adapter=None, transport=None):
        url = urlsplit(endpoint)
        require(url.scheme == "https" and url.netloc and not url.username and not url.password,
                "A credential-free HTTPS API endpoint is required")
        require(bool(model) and timeout > 0, "Model and positive timeout required")
        self.endpoint = endpoint
        self.model = model
        self.expected_version = expected_version or model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.response_adapter = response_adapter or (lambda value: value)
        self.transport = transport

    def predict(self, model_view):
        require(set(model_view) == {"recipient_role", "permission", "state", "question"},
                "Only the registered model view may enter the API request")
        case = {"model_view": model_view}
        body = to_request(case, self.model)
        if self.transport is not None:
            raw = self.transport(body)
        else:
            token = os.environ.get(self.api_key_env)
            require(bool(token), "Set the configured API-key environment variable")
            request = Request(self.endpoint, data=json.dumps(body).encode(), method="POST",
                              headers={"Authorization": "Bearer " + token,
                                       "Content-Type": "application/json"})
            try:
                with build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                    raw = json.load(response)
            except HTTPError as error:
                raise ContractError("Model API returned HTTP " + str(error.code)) from None
            except (URLError, TimeoutError):
                raise ContractError("Model API transport failed; inspect before another call") from None
        version, result = canonical_response(case, self.response_adapter(raw))
        require(version == self.expected_version, "Returned model version differs from the configured version")
        return result
