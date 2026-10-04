"""Pipeline orchestrator - coordinates all stages."""
import os
import time
import uuid
from typing import Any
from PIL import Image

from ..domain.schemas import (
    PipelineConfig,
    PipelineArtifacts,
    ExperimentManifest,
    VisualObservations,
    WorldState,
    CreativePlan,
    StoryPlan,
    StoryDraft,
    RetrievedEvidence,
)
from ..domain.enums import PipelineMode
from ..vision.florence import Florence2Model
from ..vision.grounding import GroundingDINOModel, create_grounding_dino
from ..vision.ocr import OCRModel, create_ocr
from ..vision.verifier import VisualVerifier
from ..memory.embeddings import create_embedding_model
from ..memory.faiss_store import FAISSVectorStore
from ..memory.retrieval import EvidenceRetriever
from ..context.world_state import WorldStateBuilder
from ..context.ranker import EvidenceRanker, RankingWeights
from ..context.builder import ContextBuilder
from ..context.sequence import SequenceContextBuilder
from ..narrative.planner import CreativePlanner
from ..generation.base import create_generator, StoryGenerationPipeline
from ..evaluation.evaluator import ComprehensiveEvaluator


class PipelineOrchestrator:
    """Main pipeline orchestrator for Image → Story V2."""
    
    def __init__(self, config: PipelineConfig):
        self._config = config
        self._runtime: dict[str, float] = {}
        self._logs: list[str] = []
        
        # Components (initialized lazily)
        self._florence: Florence2Model | None = None
        self._grounding: GroundingDINOModel | None = None
        self._ocr: OCRModel | None = None
        self._embedding_model = None
        self._vector_store = None
        self._retriever = None
        self._world_builder = None
        self._ranker = None
        self._context_builder = None
        self._sequence_builder = None
        self._creative_planner = None
        self._generator = None
        self._gen_pipeline = None
        self._evaluator = None
        self._verifier = None
    
    def _log(self, message: str) -> None:
        self._logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        device = self._config.device
        
        # Vision models
        self._florence = Florence2Model(device=device)
        self._florence.load()
        
        if self._config.use_grounding_dino:
            self._grounding = create_grounding_dino(device=device, enabled=True)
            self._grounding.load()
        
        if self._config.use_ocr:
            self._ocr = create_ocr(device=device, enabled=True)
            self._ocr.load()
        
        # Memory
        self._embedding_model = create_embedding_model(device=device)
        self._embedding_model.load()
        self._vector_store = FAISSVectorStore(self._embedding_model, device=device)
        self._retriever = EvidenceRetriever(self._embedding_model, self._vector_store, device=device)
        
        # Context
        self._world_builder = WorldStateBuilder()
        self._ranker = EvidenceRanker()
        self._context_builder = ContextBuilder(max_words=self._config.context_max_words)
        self._sequence_builder = SequenceContextBuilder(max_total_words=self._config.context_max_words)
        
        # Narrative
        self._creative_planner = CreativePlanner(
            seed=self._config.seed,
            creativity_config=self._config.creativity_config,
            genre=self._config.genre,
            tone=self._config.tone,
        )
        
        # Generation
        self._generator = create_generator("qwen", device=device)
        self._gen_pipeline = StoryGenerationPipeline(self._generator, self._config)
        
        # Verification
        self._verifier = VisualVerifier(
            florence_model=self._florence,
            grounding_model=self._grounding,
            device=device,
        )
        
        # Evaluation
        self._evaluator = ComprehensiveEvaluator(device=device)
    
    def run_single_image(
        self,
        image_path: str,
        evaluate: bool = True,
    ) -> PipelineArtifacts:
        """Run pipeline on a single image."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        manifest = ExperimentManifest(
            run_id=run_id,
            config=self._config.to_dict(),
            seed=self._config.seed,
            device=self._config.device,
        )
        
        artifacts = PipelineArtifacts(run_id=run_id, manifest=manifest)
        
        # Initialize components if not already done
        if self._florence is None:
            self._initialize_components()
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Stage 1: Visual Perception
        t0 = time.perf_counter()
        observations = self._florence.analyze(image, frame_id=0)
        
        # GroundingDINO detections
        if self._grounding and self._grounding.enabled:
            # Use detected entities as prompts for grounding
            phrases = observations.od_labels[:10] + observations.characters[:5]
            if phrases:
                grounding_evidence = self._grounding.detect_with_phrases(image, phrases, frame_id=0)
                observations.evidence_records.extend(grounding_evidence)
                observations.grounding_detections = [e.to_dict() for e in grounding_evidence]
        
        # OCR
        if self._ocr and self._ocr.enabled:
            ocr_evidence = self._ocr.extract_text(image, frame_id=0)
            observations.evidence_records.extend(ocr_evidence)
            if ocr_evidence:
                observations.ocr_text = " ".join([e.evidence_text.replace("OCR text: ", "") for e in ocr_evidence])
        
        artifacts.observations = [observations]
        self._runtime["vision_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 2: World State
        t0 = time.perf_counter()
        world_state = self._world_builder.add_observations(observations)
        artifacts.world_state = world_state
        self._runtime["world_state_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 3: Memory & Retrieval
        t0 = time.perf_counter()
        if self._config.use_faiss:
            self._retriever.initialize()
            self._retriever.index_observations([observations])
            
            # Retrieve for context
            query = f"{observations.scene} {' '.join(observations.characters)} {' '.join(observations.od_labels)}"
            retrieved = self._retriever.retrieve(query, top_k=self._config.faiss_top_k)
            artifacts.retrieved_evidence = retrieved
        self._runtime["retrieval_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 4: Evidence Ranking
        t0 = time.perf_counter()
        entity_recurrence = {}
        for entity in world_state.get_all_entities():
            if entity.is_recurring:
                entity_recurrence[entity.label] = len(entity.frames_present)
        
        ranked = self._ranker.rank(
            artifacts.retrieved_evidence,
            query=query if 'query' in locals() else "",
            current_frame=0,
            entity_recurrence=entity_recurrence,
        )
        ranked = self._ranker.select_for_context(ranked, max_tokens=self._config.context_max_words * 1.3)
        artifacts.ranked_evidence = ranked
        self._runtime["ranking_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 5: Creative Planning
        t0 = time.perf_counter()
        creative_plan = self._creative_planner.create_creative_plan(
            world_state, [observations], ranked
        )
        story_plan = self._creative_planner.create_story_plan(
            creative_plan, self._config.target_story_words
        )
        artifacts.creative_plan = creative_plan
        artifacts.story_plan = story_plan
        self._runtime["planning_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 6: Context Building
        t0 = time.perf_counter()
        context = self._context_builder.build_context(
            [observations], world_state, ranked, creative_plan
        )
        prompt = self._context_builder.build_story_prompt(
            context, creative_plan, self._config.target_story_words
        )
        artifacts.context = context
        self._runtime["context_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 7: Story Generation
        t0 = time.perf_counter()
        story_draft = self._gen_pipeline.generate_story(prompt, story_plan)
        story_draft.generation_time_s = round(time.perf_counter() - t0, 2)
        artifacts.story_draft = story_draft
        self._runtime["generation_s"] = story_draft.generation_time_s
        
        # Stage 8: Evaluation
        if evaluate:
            t0 = time.perf_counter()
            eval_result = self._evaluator.evaluate(artifacts, image)
            artifacts.evaluation = eval_result
            self._runtime["evaluation_s"] = round(time.perf_counter() - t0, 2)
        
        artifacts.runtime = self._runtime
        artifacts.logs = self._logs
        
        return artifacts
    
    def run_multi_image(
        self,
        image_paths: list[str],
        evaluate: bool = True,
    ) -> PipelineArtifacts:
        """Run pipeline on multiple images (sequence)."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        manifest = ExperimentManifest(
            run_id=run_id,
            config=self._config.to_dict(),
            seed=self._config.seed,
            device=self._config.device,
        )
        
        artifacts = PipelineArtifacts(run_id=run_id, manifest=manifest)
        
        # Initialize components if not already done
        if self._florence is None:
            self._initialize_components()
        
        # Load images
        images = [Image.open(p).convert("RGB") for p in image_paths]
        
        # Stage 1: Visual Perception (all frames)
        t0 = time.perf_counter()
        all_observations = []
        all_evidence = []
        
        for frame_id, image in enumerate(images):
            observations = self._florence.analyze(image, frame_id=frame_id)
            
            # GroundingDINO
            if self._grounding and self._grounding.enabled:
                phrases = observations.od_labels[:10] + observations.characters[:5]
                if phrases:
                    grounding_evidence = self._grounding.detect_with_phrases(image, phrases, frame_id=frame_id)
                    observations.evidence_records.extend(grounding_evidence)
                    observations.grounding_detections = [e.to_dict() for e in grounding_evidence]
            
            # OCR
            if self._ocr and self._ocr.enabled:
                ocr_evidence = self._ocr.extract_text(image, frame_id=frame_id)
                observations.evidence_records.extend(ocr_evidence)
                if ocr_evidence:
                    observations.ocr_text = " ".join([e.evidence_text.replace("OCR text: ", "") for e in ocr_evidence])
            
            all_observations.append(observations)
            all_evidence.extend(observations.evidence_records)
        
        artifacts.observations = all_observations
        self._runtime["vision_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 2: World State
        t0 = time.perf_counter()
        world_state = self._world_builder.add_observations_batch(all_observations)
        artifacts.world_state = world_state
        self._runtime["world_state_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 3: Memory & Retrieval
        t0 = time.perf_counter()
        if self._config.use_faiss:
            self._retriever.initialize()
            self._retriever.index_observations(all_observations)
            
            # Retrieve for story planning
            query = " ".join([
                obs.scene for obs in all_observations if obs.scene
            ] + [
                " ".join(obs.characters) for obs in all_observations
            ] + [
                " ".join(obs.od_labels) for obs in all_observations
            ])
            
            retrieved = self._retriever.retrieve_for_story_planning(
                query, len(all_observations) - 1, top_k=self._config.faiss_top_k
            )
            artifacts.retrieved_evidence = retrieved
        self._runtime["retrieval_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 4: Evidence Ranking
        t0 = time.perf_counter()
        entity_recurrence = {}
        for entity in world_state.get_all_entities():
            if entity.is_recurring:
                entity_recurrence[entity.label] = len(entity.frames_present)
        
        ranked = self._ranker.rank_for_story_planning(
            artifacts.retrieved_evidence,
            query,
            len(all_observations) - 1,
            entity_recurrence,
        )
        ranked = self._ranker.select_for_context(ranked, max_tokens=self._config.context_max_words * 1.3)
        artifacts.ranked_evidence = ranked
        self._runtime["ranking_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 5: Creative Planning
        t0 = time.perf_counter()
        creative_plan = self._creative_planner.create_creative_plan(
            world_state, all_observations, ranked
        )
        story_plan = self._creative_planner.create_story_plan(
            creative_plan, self._config.target_story_words
        )
        artifacts.creative_plan = creative_plan
        artifacts.story_plan = story_plan
        self._runtime["planning_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 6: Context Building
        t0 = time.perf_counter()
        context = self._sequence_builder.build_sequence_context(
            all_observations, world_state, ranked, creative_plan
        )
        prompt = self._sequence_builder.build_multi_image_prompt(
            context, len(all_observations), creative_plan, self._config.target_story_words
        )
        artifacts.context = context
        self._runtime["context_s"] = round(time.perf_counter() - t0, 2)
        
        # Stage 7: Story Generation
        t0 = time.perf_counter()
        story_draft = self._gen_pipeline.generate_story(prompt, story_plan)
        story_draft.generation_time_s = round(time.perf_counter() - t0, 2)
        artifacts.story_draft = story_draft
        self._runtime["generation_s"] = story_draft.generation_time_s
        
        # Stage 8: Evaluation
        if evaluate:
            t0 = time.perf_counter()
            # Use first image for CLIP evaluation
            eval_result = self._evaluator.evaluate(artifacts, images[0] if images else None)
            artifacts.evaluation = eval_result
            self._runtime["evaluation_s"] = round(time.perf_counter() - t0, 2)
        
        artifacts.runtime = self._runtime
        artifacts.logs = self._logs
        
        return artifacts
    
    def cleanup(self) -> None:
        """Clean up all models."""
        for model in [self._florence, self._grounding, self._ocr, self._embedding_model, self._generator]:
            if model and hasattr(model, 'unload'):
                model.unload()


def create_orchestrator(config: PipelineConfig) -> PipelineOrchestrator:
    """Factory function to create pipeline orchestrator."""
    return PipelineOrchestrator(config)