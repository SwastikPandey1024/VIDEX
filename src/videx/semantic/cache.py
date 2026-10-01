"""Lightweight in-process semantic request hashing, deduplication, and result caching."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any
from uuid import UUID

from videx.semantic.schemas import CandidateEvent, EvidenceBundle, SemanticEventPayload


class CacheEntry:
    """An individual cached semantic inference response with TTL expiration."""

    __slots__ = ("payload", "expires_at", "request_hash", "created_at")

    def __init__(
        self, payload: SemanticEventPayload, request_hash: str, ttl_seconds: float
    ) -> None:
        self.payload = payload
        self.request_hash = request_hash
        self.created_at = time.monotonic()
        self.expires_at = self.created_at + max(1.0, ttl_seconds)

    @property
    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


class SemanticCache:
    """In-memory LRU/TTL cache for VLM inference results and candidate deduplication.

    Provides stable SHA-256 request hashing over evidence bundle contents,
    candidate metadata, and query parameters to avoid redundant inference calls.
    """

    def __init__(self, default_ttl_seconds: float = 300.0, max_entries: int = 1000) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, CacheEntry] = {}
        # Track recent candidate signatures
        # (video_id, participant_str, start_ts_bucket) -> candidate_id
        self._recent_candidate_signatures: dict[str, tuple[UUID, float]] = {}

    def compute_request_hash(
        self,
        bundle: EvidenceBundle,
        query: str | None = None,
        provider_name: str = "default",
    ) -> str:
        """Compute deterministic SHA-256 fingerprint for bundle + query + provider."""
        hasher = hashlib.sha256()

        components: dict[str, Any] = {
            "video_id": str(bundle.video_id),
            "start_ts": f"{bundle.start_timestamp_seconds:.3f}",
            "end_ts": f"{bundle.end_timestamp_seconds:.3f}",
            "evidence_ids": sorted(str(eid) for eid in bundle.evidence_ids),
            "supporting_event_ids": sorted(str(evid) for evid in bundle.supporting_event_ids),
            "spatial_context": sorted(bundle.spatial_context),
            "query": (query or "").strip().lower(),
            "provider": provider_name,
        }

        serialized = json.dumps(components, sort_keys=True, separators=(",", ":"))
        hasher.update(serialized.encode("utf-8"))
        return hasher.hexdigest()

    def get(self, request_hash: str) -> SemanticEventPayload | None:
        """Retrieve cached payload if present and not expired."""
        entry = self._entries.get(request_hash)
        if entry is None:
            return None

        if entry.is_expired:
            self._entries.pop(request_hash, None)
            return None

        return entry.payload

    def set(
        self,
        request_hash: str,
        payload: SemanticEventPayload,
        ttl_seconds: float | None = None,
    ) -> None:
        """Store payload in cache, evicting oldest entries if at capacity."""
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds

        # Evict expired entries if approaching capacity
        if len(self._entries) >= self.max_entries:
            self._evict_expired()
            # If still full, pop first entry
            if len(self._entries) >= self.max_entries:
                first_k = next(iter(self._entries))
                self._entries.pop(first_k, None)

        self._entries[request_hash] = CacheEntry(payload, request_hash, ttl)

    def is_duplicate_candidate(
        self,
        candidate: CandidateEvent,
        temporal_bucket_seconds: float = 1.0,
    ) -> bool:
        """Check if candidate with identical participants and time bucket was recently seen."""
        now = time.monotonic()
        self._prune_old_candidate_signatures(now)

        ts_bucket = int(candidate.start_timestamp / max(0.1, temporal_bucket_seconds))
        participants_key = "_".join(sorted(candidate.participant_ids))
        sig = f"{candidate.video_id}:{participants_key}:{ts_bucket}"

        if sig in self._recent_candidate_signatures:
            return True

        self._recent_candidate_signatures[sig] = (candidate.candidate_id, now)
        return False

    def clear(self) -> None:
        """Clear all cached entries and candidate deduplication state."""
        self._entries.clear()
        self._recent_candidate_signatures.clear()

    def size(self) -> int:
        """Return total active entries."""
        self._evict_expired()
        return len(self._entries)

    def _evict_expired(self) -> None:
        """Evict all expired cache entries."""
        expired_keys = [k for k, entry in self._entries.items() if entry.is_expired]
        for k in expired_keys:
            self._entries.pop(k, None)

    def _prune_old_candidate_signatures(self, now: float) -> None:
        """Prune deduplication signatures older than 60 seconds."""
        cutoff = now - 60.0
        old_sigs = [k for k, (_, ts) in self._recent_candidate_signatures.items() if ts < cutoff]
        for k in old_sigs:
            self._recent_candidate_signatures.pop(k, None)
