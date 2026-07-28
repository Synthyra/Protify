"""Cloud backend abstraction for remote Protify job execution.

Provides a pluggable backend protocol and a built-in HTTP implementation.
External packages can register custom backends via register_cloud_backend().
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class CloudBackend(ABC):
    """Protocol for remote execution backends.

    Any server implementing the /v1/protify/* REST API contract can be used.
    """

    @abstractmethod
    def submit_job(
        self,
        config: Dict[str, Any],
        gpu_type: str = "A100",
        timeout_seconds: int = 86400,
    ) -> Dict[str, Any]:
        """Submit a training/eval job. Returns dict with at least 'job_id' and 'status'."""
        ...

    @abstractmethod
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Poll job status. Returns dict with 'job_id', 'status', 'phase', etc."""
        ...

    @abstractmethod
    def get_job_logs(
        self,
        job_id: str,
        offset: int = 0,
        max_chars: int = 50000,
    ) -> Dict[str, Any]:
        """Read log delta. Returns dict with 'content', 'offset', 'next_offset', 'total_size'."""
        ...

    @abstractmethod
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job. Returns dict with 'job_id' and 'status'."""
        ...

    @abstractmethod
    def get_results(self, job_id: str) -> Dict[str, Any]:
        """Fetch job results (metrics TSV, base64 plot images, hub URL). Returns result dict."""
        ...

    @abstractmethod
    def list_jobs(self) -> Dict[str, Any]:
        """List all jobs. Returns dict with 'jobs' list."""
        ...


class HTTPCloudBackend(CloudBackend):
    """Built-in HTTP backend. Talks to any server implementing the /v1/protify/* API."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 30) -> None:
        import requests

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post(
        self,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = self._session.post(
            self._url(path),
            json=json_data,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self._session.get(
            self._url(path),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def submit_job(
        self,
        config: Dict[str, Any],
        gpu_type: str = "A100",
        timeout_seconds: int = 86400,
    ) -> Dict[str, Any]:
        payload = {
            "config": config,
            "gpu_type": gpu_type,
            "timeout_seconds": timeout_seconds,
        }
        return self._post("/v1/protify/train", json_data=payload)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        return self._get("/v1/protify/job", params={"job_id": job_id})

    def get_job_logs(
        self,
        job_id: str,
        offset: int = 0,
        max_chars: int = 50000,
    ) -> Dict[str, Any]:
        return self._get(
            "/v1/protify/logs",
            params={
                "job_id": job_id,
                "offset": offset,
                "max_chars": max_chars,
            },
        )

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        return self._post("/v1/protify/cancel", params={"job_id": job_id})

    def get_results(self, job_id: str) -> Dict[str, Any]:
        return self._get("/v1/protify/results", params={"job_id": job_id})

    def list_jobs(self) -> Dict[str, Any]:
        return self._get("/v1/protify/jobs")


# Process-wide backend registry.

_CLOUD_BACKEND: Optional[CloudBackend] = None


def register_cloud_backend(backend: CloudBackend) -> None:
    """Register a cloud backend instance for use by CLI and GUI."""
    global _CLOUD_BACKEND
    assert isinstance(backend, CloudBackend), f"Expected CloudBackend, got {type(backend)}"
    _CLOUD_BACKEND = backend


def get_cloud_backend() -> Optional[CloudBackend]:
    """Return the currently registered cloud backend, or None."""
    return _CLOUD_BACKEND


def get_or_create_cloud_backend(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[CloudBackend]:
    """Return the registered backend, or create an HTTPCloudBackend from credentials.

    Returns None if no backend is registered and no api_key is provided.
    """
    if _CLOUD_BACKEND is not None:
        return _CLOUD_BACKEND
    if api_key:
        url = base_url or "https://api.synthyra.com"
        return HTTPCloudBackend(base_url=url, api_key=api_key)
    return None
