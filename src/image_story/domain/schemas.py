"""Domain schemas for the Image → Story V2 system.

Structured, typed data contracts for all pipeline stages.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
import uuid
import time

from .enums import (
    EvidenceType,
    EvidenceConfidence,
    InformationClass,
    SourceModel,
)


@dataclass
class BoundingBox:
    """Bounding box in [x1, y1, x2, y2] format."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_list(cls, lst: list[float]) -> BoundingBox:
        return cls(lst[0], lst[1], lst[2], lst[3])


@dataclass
class EvidenceRecord:
    """A single piece of visual evidence with full provenance."""
    id: str = field(default_factory=lambda: f"obs_{uuid.uuid4().hex[:8]}")
    entity: str = ""
    type: EvidenceType = EvidenceType.OBJECT
    action: str | None = None
    relationship: str | None = None
    frame_id: int = 0
    bbox: BoundingBox | None = None
    confidence: float = 0.0
    confidence_class: EvidenceConfidence = EvidenceConfidence.LOW
    source: SourceModel = SourceModel.FLORENCE2
    evidence_text: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sequence_position: int = 0
    information_class: InformationClass = InformationClass.HARD_FACT
    
    def __post_init__(self):
        if self.confidence >= 0.85:
            self.confidence_class = EvidenceConfidence.HIGH
        elif self.confidence >= 0.5:
            self.confidence_class = EvidenceConfidence.MEDIUM
        else:
            self.confidence_class = EvidenceConfidence.LOW
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity": self.entity,
            "type": self.type.value,
            "action": self.action,
            "relationship": self.relationship,
            "frame_id": self.frame_id,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "confidence": self.confidence,
            "confidence_class": self.confidence_class.value,
            "source": self.source.value,
            "evidence_text": self.evidence_text,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "sequence_position": self.sequence_position,
            "information_class": self.information_class.value,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceRecord:
        bbox = BoundingBox.from_list(data["bbox"]) if data.get("bbox") else None
        return cls(
            id=data.get("id", f"obs_{uuid.uuid4().hex[:8]}"),
            entity=data.get("entity", ""),
            type=EvidenceType(data.get("type", "object")),
            action=data.get("action"),
            relationship=data.get("relationship"),
            frame_id=data.get("frame_id", 0),
            bbox=bbox,
            confidence=data.get("confidence", 0.0),
            source=SourceModel(data.get("source", "florence2")),
            evidence_text=data.get("evidence_text", ""),
            provenance=data.get("provenance", {}),
            timestamp=data.get("timestamp", time.time()),
            sequence_position=data.get("sequence_position", 0),
            information_class=InformationClass(data.get("information_class", "hard_fact")),
        )


@dataclass
class VisualObservations:
    """Structured output from the vision perception stage."""
    image_id: str
    frame_id: int
    scene: str = ""
    detailed_caption: str = ""
    objects: list[str] = field(default_factory=list)
    od_labels: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    spatial_relations: list[str] = field(default_factory=list)
    region_descriptions: list[str] = field(default_factory=list)
    style_or_mood: str = ""
    ocr_text: str = ""
    grounding_detections: list[dict[str, Any]] = field(default_factory=list)
    evidence_records: list[EvidenceRecord] = field(default_factory=list)
    runtime_s: float = 0.0
    model_load_s: float = 0.0
    models_used: list[str] = field(default_factory=list)


@dataclass
class WorldEntity:
    """An entity tracked across frames."""
    id: str
    label: str
    entity_type: str  # person, object, location
    first_frame: int
    last_frame: int
    frames_present: list[int] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    bounding_boxes: dict[int, BoundingBox] = field(default_factory=dict)
    associated_entities: list[str] = field(default_factory=list)
    is_recurring: bool = False
    disappearance_frame: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "entity_type": self.entity_type,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames_present": self.frames_present,
            "attributes": self.attributes,
            "bounding_boxes": {k: v.to_list() for k, v in self.bounding_boxes.items()},
            "associated_entities": self.associated_entities,
            "is_recurring": self.is_recurring,
            "disappearance_frame": self.disappearance_frame,
        }


@dataclass
class WorldState:
    """Persistent world state across multiple frames."""
    characters: list[WorldEntity] = field(default_factory=list)
    objects: list[WorldEntity] = field(default_factory=list)
    locations: list[WorldEntity] = field(default_factory=list)
    goals: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    open_loops: list[dict[str, Any]] = field(default_factory=list)
    previous_events: list[dict[str, Any]] = field(default_factory=list)
    frame_count: int = 0
    
    def get_all_entities(self) -> list[WorldEntity]:
        return self.characters + self.objects + self.locations
    
    def get_entity_by_label(self, label: str) -> WorldEntity | None:
        for entity in self.get_all_entities():
            if entity.label.lower() == label.lower():
                return entity
        return None
    
    def get_disappeared_entities(self) -> list[WorldEntity]:
        """Get entities that have disappeared (disappearance_frame is set)."""
        return [e for e in self.get_all_entities() if e.disappearance_frame is not None]
    
    def get_recurring_entities(self) -> list[WorldEntity]:
        """Get entities that appear in multiple frames."""
        return [e for e in self.get_all_entities() if e.is_recurring]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "characters": [e.to_dict() for e in self.characters],
            "objects": [e.to_dict() for e in self.objects],
            "locations": [e.to_dict() for e in self.locations],
            "goals": self.goals,
            "relationships": self.relationships,
            "open_loops": self.open_loops,
            "previous_events": self.previous_events,
            "frame_count": self.frame_count,
        }


@dataclass
class RetrievedEvidence:
    """Evidence retrieved from FAISS with ranking metadata."""
    record: EvidenceRecord
    semantic_similarity: float = 0.0
    rank_score: float = 0.0
    rank_factors: dict[str, float] = field(default_factory=dict)


@dataclass
class CreativePlan:
    """Structured creative plan for story generation."""
    genre: str = "whimsical"
    tone: str = "comedic"
    creativity_level: float = 0.7
    surprise_level: float = 0.6
    humor_level: float = 0.5
    mystery_level: float = 0.4
    emotion_level: float = 0.5
    dialogue_level: float = 0.4
    metaphor_level: float = 0.3
    characters: list[dict[str, Any]] = field(default_factory=list)
    central_conflict: dict[str, Any] | None = None
    open_loops: list[str] = field(default_factory=list)
    foreshadowing_elements: list[dict[str, Any]] = field(default_factory=list)
    callback_plan: list[dict[str, Any]] = field(default_factory=list)
    narrative_arc: list[dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "genre": self.genre,
            "tone": self.tone,
            "creativity_level": self.creativity_level,
            "surprise_level": self.surprise_level,
            "humor_level": self.humor_level,
            "mystery_level": self.mystery_level,
            "emotion_level": self.emotion_level,
            "dialogue_level": self.dialogue_level,
            "metaphor_level": self.metaphor_level,
            "characters": self.characters,
            "central_conflict": self.central_conflict,
            "open_loops": self.open_loops,
            "foreshadowing_elements": self.foreshadowing_elements,
            "callback_plan": self.callback_plan,
            "narrative_arc": self.narrative_arc,
        }


@dataclass
class StoryBeat:
    """A single beat in the story plan."""
    beat_number: int
    beat_type: str  # setup, goal, conflict, escalation, surprise, resolution, callback
    description: str
    key_entities: list[str] = field(default_factory=list)
    key_evidence_ids: list[str] = field(default_factory=list)
    creative_elements: list[str] = field(default_factory=list)
    target_words: int = 30


@dataclass
class StoryPlan:
    """Complete story plan with beats."""
    beats: list[StoryBeat] = field(default_factory=list)
    creative_plan: CreativePlan | None = None
    target_total_words: int = 250
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "beats": [
                {
                    "beat_number": b.beat_number,
                    "beat_type": b.beat_type,
                    "description": b.description,
                    "key_entities": b.key_entities,
                    "key_evidence_ids": b.key_evidence_ids,
                    "creative_elements": b.creative_elements,
                    "target_words": b.target_words,
                }
                for b in self.beats
            ],
            "creative_plan": self.creative_plan.to_dict() if self.creative_plan else None,
            "target_total_words": self.target_total_words,
        }


@dataclass
class StoryDraft:
    """Generated story with metadata."""
    text: str
    story_plan: StoryPlan | None = None
    word_count: int = 0
    generation_time_s: float = 0.0
    model_used: str = "qwen2.5-0.5b-instruct"
    prompt_used: str = ""
    
    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class StoryClaim:
    """A claim extracted from the generated story."""
    id: str = field(default_factory=lambda: f"claim_{uuid.uuid4().hex[:8]}")
    subject: str = ""
    relation: str = ""
    object: str = ""
    original_sentence: str = ""
    claim_type: InformationClass = InformationClass.HARD_FACT
    confidence: float = 0.0
    
    def to_natural_language(self) -> str:
        return f"The {self.subject} {self.relation} the {self.object}."
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "original_sentence": self.original_sentence,
            "claim_type": self.claim_type.value,
            "confidence": self.confidence,
            "natural_language": self.to_natural_language(),
        }


@dataclass
class VerificationResult:
    """Result of verifying a claim against visual evidence."""
    claim_id: str
    claim: StoryClaim
    status: Literal["supported", "unsupported", "contradicted"]
    supporting_evidence: list[EvidenceRecord] = field(default_factory=list)
    contradicting_evidence: list[EvidenceRecord] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim.to_dict(),
            "status": self.status,
            "supporting_evidence_count": len(self.supporting_evidence),
            "contradicting_evidence_count": len(self.contradicting_evidence),
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results."""
    grounding_score: float = 0.0
    clip_image_story_mean: float = 0.0
    nli_contra_mean: float = 0.0
    attribute_conflict: list[str] = field(default_factory=list)
    repetition_rate: float = 0.0
    length_valid: bool = False
    grounding_pass: bool = False
    
    # V2 metrics
    claim_support_rate: float = 0.0
    claim_grounding_score: float = 0.0
    continuity_score: float = 0.0
    narrative_coherence: float = 0.0
    contradiction_count: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    
    # Runtime
    eval_clip_s: float = 0.0
    eval_nli_s: float = 0.0
    eval_rules_s: float = 0.0
    eval_verification_s: float = 0.0
    eval_total_s: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "grounding_score": self.grounding_score,
            "clip_image_story_mean": self.clip_image_story_mean,
            "nli_contra_mean": self.nli_contra_mean,
            "attribute_conflict": self.attribute_conflict,
            "repetition_rate": self.repetition_rate,
            "length_valid": self.length_valid,
            "grounding_pass": self.grounding_pass,
            "claim_support_rate": self.claim_support_rate,
            "claim_grounding_score": self.claim_grounding_score,
            "continuity_score": self.continuity_score,
            "narrative_coherence": self.narrative_coherence,
            "contradiction_count": self.contradiction_count,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "contradicted_claims": self.contradicted_claims,
            "eval_clip_s": self.eval_clip_s,
            "eval_nli_s": self.eval_nli_s,
            "eval_rules_s": self.eval_rules_s,
            "eval_verification_s": self.eval_verification_s,
            "eval_total_s": self.eval_total_s,
        }


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    mode: Literal["fast", "standard", "full"] = "standard"
    use_grounding_dino: bool = True
    use_ocr: bool = False
    use_faiss: bool = True
    use_creative_planner: bool = True
    use_verification: bool = True
    max_evidence_per_frame: int = 50
    faiss_top_k: int = 10
    context_max_words: int = 500
    target_story_words: int = 250
    creativity_config: dict[str, float] = field(default_factory=lambda: {
        "creativity": 0.7,
        "surprise": 0.6,
        "humor": 0.5,
        "mystery": 0.4,
        "emotion": 0.5,
        "dialogue": 0.4,
        "metaphor": 0.3,
    })
    genre: str = "whimsical"
    tone: str = "comedic"
    seed: int = 0
    device: str = "cpu"
    
    @classmethod
    def from_mode(cls, mode: str) -> PipelineConfig:
        """Create config from preset mode."""
        if mode == "fast":
            return cls(
                mode="fast",
                use_grounding_dino=False,
                use_ocr=False,
                use_faiss=False,
                use_creative_planner=False,
                use_verification=False,
                target_story_words=100,
            )
        elif mode == "standard":
            return cls(
                mode="standard",
                use_grounding_dino=True,
                use_ocr=False,
                use_faiss=True,
                use_creative_planner=True,
                use_verification=True,
                target_story_words=250,
            )
        elif mode == "full":
            return cls(
                mode="full",
                use_grounding_dino=True,
                use_ocr=True,
                use_faiss=True,
                use_creative_planner=True,
                use_verification=True,
                target_story_words=300,
                creativity_config={
                    "creativity": 0.8,
                    "surprise": 0.7,
                    "humor": 0.6,
                    "mystery": 0.5,
                    "emotion": 0.6,
                    "dialogue": 0.5,
                    "metaphor": 0.4,
                },
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "use_grounding_dino": self.use_grounding_dino,
            "use_ocr": self.use_ocr,
            "use_faiss": self.use_faiss,
            "use_creative_planner": self.use_creative_planner,
            "use_verification": self.use_verification,
            "max_evidence_per_frame": self.max_evidence_per_frame,
            "faiss_top_k": self.faiss_top_k,
            "context_max_words": self.context_max_words,
            "target_story_words": self.target_story_words,
            "creativity_config": self.creativity_config,
            "genre": self.genre,
            "tone": self.tone,
            "seed": self.seed,
            "device": self.device,
        }


@dataclass
class ExperimentManifest:
    """Manifest for experiment reproducibility."""
    git_commit: str = ""
    vision_model: str = ""
    language_model: str = ""
    embedding_model: str = ""
    dataset: str = ""
    dataset_hash: str = ""
    seed: int = 0
    device: str = "cpu"
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "vision_model": self.vision_model,
            "language_model": self.language_model,
            "embedding_model": self.embedding_model,
            "dataset": self.dataset,
            "dataset_hash": self.dataset_hash,
            "seed": self.seed,
            "device": self.device,
            "config": self.config,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }


@dataclass
class PipelineArtifacts:
    """Artifacts produced by a pipeline run."""
    run_id: str
    manifest: ExperimentManifest
    observations: list[VisualObservations] = field(default_factory=list)
    world_state: WorldState | None = None
    retrieved_evidence: list[RetrievedEvidence] = field(default_factory=list)
    ranked_evidence: list[RetrievedEvidence] = field(default_factory=list)
    context: str = ""
    creative_plan: CreativePlan | None = None
    story_plan: StoryPlan | None = None
    story_draft: StoryDraft | None = None
    claims: list[StoryClaim] = field(default_factory=list)
    verification_results: list[VerificationResult] = field(default_factory=list)
    evaluation: EvaluationResult | None = None
    runtime: dict[str, float] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "manifest": self.manifest.to_dict(),
            "observations": [obs.__dict__ for obs in self.observations],
            "world_state": self.world_state.to_dict() if self.world_state else None,
            "retrieved_evidence": [e.to_dict() for e in self.retrieved_evidence],
            "ranked_evidence": [e.to_dict() for e in self.ranked_evidence],
            "context": self.context,
            "creative_plan": self.creative_plan.to_dict() if self.creative_plan else None,
            "story_plan": self.story_plan.to_dict() if self.story_plan else None,
            "story_draft": self.story_draft.__dict__ if self.story_draft else None,
            "claims": [c.to_dict() for c in self.claims],
            "verification_results": [v.to_dict() for v in self.verification_results],
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "runtime": self.runtime,
            "logs": self.logs,
        }