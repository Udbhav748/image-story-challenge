"""Enumerations for the Image → Story V2 system."""
from enum import Enum


class EvidenceType(str, Enum):
    """Types of visual evidence."""
    ENTITY = "entity"
    OBJECT = "object"
    PERSON = "person"
    ACTION = "action"
    RELATIONSHIP = "relationship"
    LOCATION = "location"
    SPATIAL_FACT = "spatial_fact"
    OCR = "ocr"
    VISUAL_ATTRIBUTE = "visual_attribute"
    EVENT = "event"


class EvidenceConfidence(str, Enum):
    """Confidence classification for evidence."""
    HIGH = "high"       # >= 0.85
    MEDIUM = "medium"   # 0.5 - 0.85
    LOW = "low"         # < 0.5


class InformationClass(str, Enum):
    """Classification of information for story generation."""
    HARD_FACT = "hard_fact"           # Directly supported by visual evidence
    SOFT_INFERENCE = "soft_inference" # Reasonable interpretation, not directly proven
    CREATIVE_SPACE = "creative_space" # Narrative invention allowed


class SourceModel(str, Enum):
    """Source vision models."""
    FLORENCE2 = "florence2"
    GROUNDING_DINO = "grounding_dino"
    OCR = "ocr"
    HUMAN = "human"


class PipelineMode(str, Enum):
    """Pipeline execution modes."""
    FAST = "fast"
    STANDARD = "standard"
    FULL = "full"


class VisionTask(str, Enum):
    """Florence-2 vision tasks."""
    DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
    OBJECT_DETECTION = "<OD>"
    DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    OCR = "<OCR>"
    CAPTION = "<CAPTION>"
    REGION_PROPOSAL = "<REGION_PROPOSAL>"


class GroundingDINOModel(str, Enum):
    """GroundingDINO model variants."""
    BASE = "IDEA-Research/grounding-dino-base"
    TINY = "IDEA-Research/grounding-dino-tiny"


class EmbeddingModel(str, Enum):
    """Sentence transformer embedding models."""
    MINILM_L6 = "sentence-transformers/all-MiniLM-L6-v2"
    MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"
    E5_SMALL = "intfloat/e5-small-v2"


class StoryGenre(str, Enum):
    """Story genre presets."""
    COMEDY = "comedy"
    MYSTERY = "mystery"
    ADVENTURE = "adventure"
    EMOTIONAL = "emotional"
    WHIMSICAL = "whimsical"
    THRILLER = "thriller"
    ABSURD = "absurd"
    ROMANTIC = "romantic"


class StoryTone(str, Enum):
    """Story tone presets."""
    COMEDIC = "comedic"
    SERIOUS = "serious"
    LIGHTHEARTED = "lighthearted"
    DARK = "dark"
    WHIMSICAL = "whimsical"
    SUSPENSEFUL = "suspenseful"
    HEARTWARMING = "heartwarming"
    SARCASTIC = "sarcastic"


class BeatType(str, Enum):
    """Story beat types."""
    SETUP = "setup"
    GOAL = "goal"
    CONFLICT = "conflict"
    ESCALATION = "escalation"
    SURPRISE = "surprise"
    TWIST = "twist"
    RESOLUTION = "resolution"
    CALLBACK = "callback"
    FORESHADOWING = "foreshadowing"


class ClaimStatus(str, Enum):
    """Claim verification status."""
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class ConflictType(str, Enum):
    """Types of narrative conflict."""
    MISUNDERSTANDING = "misunderstanding"
    MISSING_OBJECT = "missing_object"
    TIME_PRESSURE = "time_pressure"
    SOCIAL_AWKWARDNESS = "social_awkwardness"
    UNEXPECTED_DISCOVERY = "unexpected_discovery"
    COMPETING_GOALS = "competing_goals"
    MYSTERY = "mystery"
    EMBARRASSMENT = "embarrassment"
    SURPRISING_COINCIDENCE = "surprising_coincidence"
    ENVIRONMENTAL_OBSTACLE = "environmental_obstacle"


class HumorStyle(str, Enum):
    """Humor styles for story generation."""
    DEADPAN = "deadpan"
    AWKWARD = "awkward"
    ABSURD = "absurd"
    SITUATIONAL = "situational"
    MISUNDERSTANDING = "misunderstanding"
    PERSONIFICATION = "personification"
    UNEXPECTED_PAYOFF = "unexpected_payoff"
    CALLBACK = "callback"


class EvaluationMetric(str, Enum):
    """Evaluation metrics."""
    GROUNDING = "grounding"
    CONTINUITY = "continuity"
    NARRATIVE_QUALITY = "narrative_quality"
    CONTRADICTIONS = "contradictions"
    REPETITION = "repetition"
    CLAIM_SUPPORT = "claim_support"
    VISUAL_SUPPORT = "visual_support"