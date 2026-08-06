# ============================================================
# schema.py — Phase 1 (HPC)
# Unified task schema and generation record structures.
# Unchanged in substance from Phase 0 — this file was already
# clean, portable Python with no Colab-specific dependencies,
# so it ports over as-is aside from the import statement.
# ============================================================

from dataclasses import dataclass, field
from typing import List, Optional

import config


@dataclass
class LegalTask:
    """A single LegalBench task instance, normalized to a common schema."""
    task_id: str
    task_type: str          # one of config.CATEGORIES
    context: str
    question: str
    label_options: List[str]
    expected_output: str
    jurisdiction: str = "US General"
    source_dataset: str = "LegalBench"


@dataclass
class Demonstration:
    """A single fixed few-shot demonstration example (no reasoning trace)."""
    context: str
    question: str
    label: str

    def render(self) -> str:
        if self.question:
            return f"Context: {self.context}\nQuestion: {self.question}\n{config.FINAL_ANSWER_PREFIX} {self.label}"
        return f"Context: {self.context}\n{config.FINAL_ANSWER_PREFIX} {self.label}"


@dataclass
class TaskDemonstrationSet:
    """Cached, fixed demonstration set for one task — reused across all rephrasings/runs."""
    task_id: str
    demonstrations: List[Demonstration] = field(default_factory=list)


@dataclass
class GenerationRecord:
    """One logged generation — the atomic unit written to outputs/raw_generations/."""
    task_id: str
    category: str
    strategy: str
    rephrasing_id: int      # 0, 1, or 2
    run_id: int              # 0 through N_RUNS-1
    model_name: str
    prompt_text: str
    raw_output: str
    parsed_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    timestamp: Optional[str] = None   # ISO 8601, set at generation time