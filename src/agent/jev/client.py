"""Jev client: one protocol, three transports.

    LiveTransport      -> the real API
    RecordingTransport -> the real API, writing a cassette as it goes
    ReplayTransport    -> a cassette only, raising on a miss

The offline test suite uses ReplayTransport, which makes it deterministic
despite the API's ~±0.03 sampling jitter, and free to run. Testing Jev's own
stability is the live suite's job; conflating the two is how these harnesses
rot.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Protocol

import httpx

from .errors import (
    CassetteMiss,
    JevTransportError,
    classify_http_error,
)
from .schema import JevRequest, JevResponse

DEFAULT_BASE_URL = "https://api.typesafe.ai"
ENDPOINT = "/v1/systemone"

# Measured p50 was ~0.25s and the worst realistic payload ~0.9s. 30s is far
# beyond anything observed, so a timeout here means something is actually wrong.
DEFAULT_TIMEOUT = 30.0

# 12 concurrent requests all succeeded with no rate limiting, but nothing is
# documented and there is no auth, so leave headroom.
MAX_CONCURRENCY = 8


def cassette_key(request: JevRequest) -> str:
    """Stable hash of a request.

    Canonicalised with sorted keys so that dict ordering -- which carries no
    meaning to the server, and which we know does not affect the answer since
    Jev shows no positional bias -- never produces a spurious cassette miss.
    """
    payload = request.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class Transport(Protocol):
    def send(self, request: JevRequest) -> JevResponse: ...


class LiveTransport:
    """Talks to the real API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Jev needs no credential today. Send the header only if we have one,
        # so nothing breaks if auth is switched on later.
        self.api_key = api_key if api_key is not None else os.getenv("TYPESAFE_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def send(self, request: JevRequest) -> JevResponse:
        payload = request.model_dump(mode="json")
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(
                    f"{self.base_url}{ENDPOINT}",
                    json=payload,
                    headers=self._headers(),
                )
            except httpx.HTTPError as exc:
                last_error = JevTransportError(f"request failed: {exc}")
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise last_error from exc

            if response.status_code == 200:
                return JevResponse.model_validate(response.json())

            try:
                body = response.json()
            except ValueError:
                body = response.text

            error = classify_http_error(response.status_code, body)
            # Only transport-class failures are worth retrying; a 422 will
            # fail identically forever.
            if isinstance(error, JevTransportError) and attempt < self.max_retries:
                last_error = error
                time.sleep(2**attempt)
                continue
            raise error

        raise last_error or JevTransportError("exhausted retries")

    def close(self) -> None:
        self._client.close()


class Cassette:
    """A JSONL record of request/response pairs, keyed by request hash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry["key"]] = entry

    def get(self, key: str) -> JevResponse | None:
        entry = self._entries.get(key)
        return JevResponse.model_validate(entry["response"]) if entry else None

    def put(self, key: str, request: JevRequest, response: JevResponse, latency_ms: float) -> None:
        entry = {
            "key": key,
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
            "model": response.model,
            "latency_ms": round(latency_ms, 1),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._entries[key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def closest_key(self, key: str) -> str | None:
        """Best-effort nearest recorded key, to make a miss diagnosable."""
        if not self._entries:
            return None
        return max(
            self._entries,
            key=lambda k: sum(a == b for a, b in zip(k, key)),
        )

    def __len__(self) -> int:
        return len(self._entries)


class ReplayTransport:
    """Serves only from a cassette. A miss raises rather than hitting the network."""

    def __init__(self, cassette: Cassette) -> None:
        self.cassette = cassette

    def send(self, request: JevRequest) -> JevResponse:
        key = cassette_key(request)
        response = self.cassette.get(key)
        if response is None:
            raise CassetteMiss(key, self.cassette.closest_key(key))
        return response


class RecordingTransport:
    """Hits the real API and appends every exchange to a cassette."""

    def __init__(self, live: LiveTransport, cassette: Cassette) -> None:
        self.live = live
        self.cassette = cassette

    def send(self, request: JevRequest) -> JevResponse:
        key = cassette_key(request)
        cached = self.cassette.get(key)
        if cached is not None:
            return cached
        started = time.perf_counter()
        response = self.live.send(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.cassette.put(key, request, response, elapsed_ms)
        return response


class JevClient:
    """The thing the rest of the codebase talks to.

    Validation happens here, before the wire, so a malformed panel fails with
    our error message rather than a 422 we then have to decode.
    """

    def __init__(self, transport: Transport, model: str = "jev-latest") -> None:
        self.transport = transport
        self.model = model
        self.last_model_served: str | None = None

    def ask(self, state: str, questions: dict) -> JevResponse:
        request = JevRequest(model=self.model, state=state, questions=questions)
        response = self.transport.send(request)
        self.last_model_served = response.model
        return response

    @classmethod
    def replaying(cls, cassette_path: Path, model: str = "jev-latest") -> JevClient:
        return cls(ReplayTransport(Cassette(cassette_path)), model=model)

    @classmethod
    def recording(cls, cassette_path: Path, model: str = "jev-latest") -> JevClient:
        return cls(RecordingTransport(LiveTransport(), Cassette(cassette_path)), model=model)

    @classmethod
    def live(cls, model: str = "jev-latest") -> JevClient:
        return cls(LiveTransport(), model=model)
