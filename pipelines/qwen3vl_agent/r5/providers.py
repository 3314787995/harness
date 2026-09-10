"""R5 consumes the existing read-only provider contract, never generates transcripts."""

from qwen3vl_agent.r1.providers import ExternalEvidenceProvider, ExternalSegment, ProviderResult
from qwen3vl_agent.r4.providers import (
    FileEvidenceProvider,
    InventoryProviderResult,
    ProviderAdapter,
    read_external_file,
)

SummaryProviderResult = InventoryProviderResult

__all__ = [
    "ExternalEvidenceProvider",
    "ExternalSegment",
    "FileEvidenceProvider",
    "ProviderAdapter",
    "ProviderResult",
    "SummaryProviderResult",
    "read_external_file",
]
