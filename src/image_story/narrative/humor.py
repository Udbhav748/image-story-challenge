"""Humor engine for narrative generation."""
from typing import Any
import random

from ..domain.schemas import WorldEntity
from ..domain.enums import HumorStyle


class HumorEngine:
    """Generate humor elements grounded in visual evidence."""
    
    HUMOR_TEMPLATES = {
        HumorStyle.DEADPAN: [
            "{char} stared at the {object}. The {object} stared back. Neither blinked.",
            "It was a {adjective} day for a {noun}. {char} would know - they'd seen {number} of them.",
            "{char} sighed. '{dialogue}', they said to the {object}. The {object} remained unimpressed.",
        ],
        HumorStyle.AWKWARD: [
            "{char} waved at {other_char}, realized they were waving at a {object}, and kept waving anyway.",
            "The silence stretched. {char} cleared their throat. '{dialogue}', they offered. The {object} said nothing.",
            "{char} tried to {action} casually. The {object} fell over. {char} pretended it was intentional.",
        ],
        HumorStyle.ABSURD: [
            "The {object} began reciting poetry. {char} took notes, nodding seriously.",
            "{char} discovered the {object} was actually a {absurd_identity} in disguise. Naturally.",
            "The {object} had opinions about {topic}. {char} disagreed, and a debate ensued.",
        ],
        HumorStyle.SITUATIONAL: [
            "{char} needed a {tool} but only had a {wrong_tool}. It worked. Somehow.",
            "The {object} was exactly where {char} left it. This was suspicious.",
            "Everything was going according to plan. This worried {char} more than chaos would.",
        ],
        HumorStyle.MISUNDERSTANDING: [
            "{char} heard '{phrase}' and prepared for {wrong_preparation}. It was actually about {actual_topic}.",
            "{char} and {other_char} agreed perfectly on {topic}. They meant completely different things.",
            "The sign said '{sign_text}'. {char} interpreted this as '{misinterpretation}'.",
        ],
        HumorStyle.PERSONIFICATION: [
            "The {object} sighed. It had a long day of being a {object}.",
            "{char} swore the {object} rolled its eyes. {object}s don't have eyes. Probably.",
            "The {object} judged {char}'s life choices. It was a {object}; judgment was its purpose.",
        ],
        HumorStyle.UNEXPECTED_PAYOFF: [
            "{char} spent hours {effort}. The result: {mundane_result}. Perfect.",
            "The {object} turned out to be {unexpected_identity}. {char} celebrated with {celebration}.",
            "After all that, {char} realized the {object} was exactly what it claimed to be. Disappointing.",
        ],
        HumorStyle.CALLBACK: [
            "Just like the {previous_object} from {previous_frame}, this {object} also {behavior}.",
            "Remember when {char} said '{previous_quote}'? The {object} proved them right. Again.",
            "The {object} had the same {trait} as the {previous_object}. Coincidence? {char} thought not.",
        ],
    }
    
    def __init__(self, seed: int = 0, humor_level: float = 0.5):
        self._seed = seed
        self._rng = random.Random(seed)
        self._humor_level = humor_level
    
    def generate_humor_moments(
        self,
        world_state: Any,
        conflict: dict[str, Any],
        style: HumorStyle = HumorStyle.SITUATIONAL,
        count: int = 2,
    ) -> list[str]:
        """Generate humor moments for the story."""
        if self._rng.random() > self._humor_level:
            return []
        
        characters = [e for e in world_state.get_all_entities() if e.entity_type == "character"]
        objects = [e for e in world_state.get_all_entities() if e.entity_type == "object"]
        
        if not characters:
            return []
        
        char = self._rng.choice(characters)
        other_char = self._rng.choice(characters) if len(characters) > 1 else None
        obj = self._rng.choice(objects) if objects else None
        
        templates = self.HUMOR_TEMPLATES.get(style, self.HUMOR_TEMPLATES[HumorStyle.SITUATIONAL])
        moments = []
        
        for _ in range(min(count, len(templates))):
            template = self._rng.choice(templates)
            moment = self._fill_humor_template(template, char, other_char, obj)
            moments.append(moment)
        
        return moments
    
    def _fill_humor_template(
        self,
        template: str,
        char: WorldEntity,
        other_char: WorldEntity | None,
        obj: WorldEntity | None,
    ) -> str:
        fillers = {
            "char": char.label,
            "other_char": other_char.label if other_char else "someone",
            "object": obj.label if obj else "thing",
            "adjective": "perfectly ordinary",
            "noun": "adventure",
            "number": "seventeen",
            "dialogue": "Well, that happened",
            "action": "pick it up",
            "absurd_identity": "secret agent",
            "topic": "the meaning of existence",
            "tool": "hammer",
            "wrong_tool": "banana",
            "phrase": "the eagle has landed",
            "wrong_preparation": "an alien invasion",
            "actual_topic": "a bird",
            "sign_text": "wet paint",
            "misinterpretation": "paint is wet, touch it",
            "previous_object": "red balloon",
            "previous_frame": "earlier",
            "behavior": "floated away",
            "previous_quote": "it'll come back",
            "trait": "mysterious disappearance habit",
            "effort": "building a complex machine",
            "mundane_result": "a sandwich",
            "unexpected_identity": "a key",
            "celebration": "tea",
        }
        
        for key, value in fillers.items():
            template = template.replace(f"{{{key}}}", value)
        
        return template
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._rng.seed(seed)
    
    def set_humor_level(self, level: float) -> None:
        self._humor_level = max(0.0, min(1.0, level))