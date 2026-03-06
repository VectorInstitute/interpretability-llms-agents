"""Model Evaluation Packet (MEP) schema — portable trace artifact.

MEP v1 stores everything needed to replay and audit a single agent run.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ImageRef:
    path: str
    sha256: str


@dataclass
class MEPConfig:
    planner_backend: str       # "openai" | "gemini"
    vision_backend: str
    judge_backend: str
    config_name: str           # e.g. "openai_gemini"
    planner_model: str
    vision_model: str


@dataclass
class MEPSample:
    dataset: str
    sample_id: str
    question: str
    question_type: str
    expected_output: str
    image_ref: ImageRef
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MEPPlan:
    prompt: str
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    parse_error: bool = False


@dataclass
class ToolTrace:
    tool: str
    backend: str
    model: str
    start_ts: str
    end_ts: str
    elapsed_ms: float = 0.0
    provider_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MEPVision:
    prompt: str
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)   # {answer, explanation}
    parse_error: bool = False
    tool_trace: List[Dict] = field(default_factory=list)


@dataclass
class MEPOcr:
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)   # {chart_type, title, x_axis, y_axis, legend, data_labels, annotations}
    parse_error: bool = False
    tool_trace: List[Dict] = field(default_factory=list)


@dataclass
class MEPVerifier:
    prompt: str
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)   # {verdict, answer, reasoning}
    parse_error: bool = False
    verdict: str = "skipped"   # "confirmed" | "revised" | "skipped"


@dataclass
class MEPTimestamps:
    start: str
    end: str
    planner_ms: float = 0.0
    ocr_ms: float = 0.0          # 0.0 when OCR step is skipped
    vision_ms: float = 0.0
    verifier_ms: float = 0.0


@dataclass
class MEP:
    schema_version: str = "mep.v1"
    run_id: str = ""
    config: Optional[MEPConfig] = None
    sample: Optional[MEPSample] = None
    plan: Optional[MEPPlan] = None
    ocr: Optional[MEPOcr] = None             # None when OCR step is skipped
    vision: Optional[MEPVision] = None
    verifier: Optional[MEPVerifier] = None   # Pass 2.5 — None when skipped
    timestamps: Optional[MEPTimestamps] = None
    errors: List[str] = field(default_factory=list)
    opik_trace_id: Optional[str] = None  # set when Opik tracing is active

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
