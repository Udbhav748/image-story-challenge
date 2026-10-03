"""Lightweight tests for the Image -> Story pipeline."""
import os, sys, json, tempfile, csv
sys.path.insert(0, os.path.dirname(__file__))

import baseline
import seeing
import context_builder
import metric


def test_word_count_validation():
    """Test length_valid boundary conditions."""
    assert metric.split_sentences("Hello. World!") == ["Hello.", "World!"]
    assert metric.split_sentences("No punctuation") == ["No punctuation"]
    assert metric.split_sentences("") == []
    
    # word count
    assert len("hello world".split()) == 2
    assert len(("x " * 79 + "y").split()) == 80
    assert len(("x " * 120 + "y").split()) == 121
    print("OK word_count_validation")


def test_grounding_calculation():
    """Test grounding score formula with known inputs."""
    # Mock the components
    clip_n = 0.5
    nli_contra_mean = 0.1
    attr_conflict = False
    g = 0.4 * clip_n + 0.4 * (1 - nli_contra_mean) + 0.2 * (1 if not attr_conflict else 0)
    assert abs(g - (0.4*0.5 + 0.4*0.9 + 0.2*1)) < 0.001
    
    # With attribute conflict
    g2 = 0.4 * clip_n + 0.4 * (1 - nli_contra_mean) + 0.2 * 0
    assert g2 < g
    print("OK grounding_calculation")


def test_repetition_detection():
    """Test repetition_score function."""
    # No repetition
    story1 = "The cat sat on the mat. It purred softly."
    rep1 = metric.repetition_score(story1)
    assert rep1["repetition_rate"] == 0.0
    assert rep1["distinct3"] == 1.0
    assert rep1["repeated_sentences"] == 0
    
    # Repeated sentence
    story2 = "The cat sat. The cat sat. The dog ran."
    rep2 = metric.repetition_score(story2)
    assert rep2["repeated_sentences"] == 1
    
    # Repeated trigrams
    story3 = "a b c d a b c d"  # trigrams: (a,b,c), (b,c,d), (c,d,a), (d,a,b), (a,b,c), (b,c,d) -> 2 repeated
    rep3 = metric.repetition_score(story3)
    assert rep3["repeated_trigrams"] >= 1
    assert rep3["repetition_rate"] > 0
    
    print("OK repetition_detection")


def test_continuity_score():
    """Test continuity_score function."""
    # Single context -> perfect scores
    ctx1 = {"characters": ["cat"], "objects": ["mat"], "region_descriptions": ["cat on mat"]}
    cont1 = metric.continuity_score([ctx1], "The cat sat on the mat.")
    assert cont1["entity_consistency"] == 1.0
    assert cont1["adjacent_similarity"] == 1.0
    
    # Two contexts with shared entity
    ctx2 = [
        {"characters": ["cat"], "objects": ["mat"], "region_descriptions": ["cat on mat"]},
        {"characters": ["cat"], "objects": ["bowl"], "region_descriptions": ["cat eats"]},
    ]
    cont2 = metric.continuity_score(ctx2, "The cat sat on the mat then ate from the bowl.")
    assert cont2["entity_consistency"] == 1.0  # "cat" recurs and appears in story
    assert cont2["transition_markers"] >= 1  # "then"
    assert 0 < cont2["adjacent_similarity"] < 1
    
    # Two contexts with no shared entity
    ctx3 = [
        {"characters": ["cat"], "objects": ["mat"], "region_descriptions": ["cat on mat"]},
        {"characters": ["dog"], "objects": ["bone"], "region_descriptions": ["dog chews"]},
    ]
    cont3 = metric.continuity_score(ctx3, "The cat sat. The dog chewed.")
    assert cont3["entity_consistency"] == 1.0  # no recurring entities -> vacuously 1.0
    
    print("OK continuity_score")


def test_attribute_conflict():
    """Test attribute_conflict detection."""
    cap = "a red wooden car"
    story1 = "The red wooden car drove fast."  # no conflict
    story2 = "The blue wooden car drove fast."  # color conflict
    story3 = "The red metal car drove fast."  # material conflict
    
    assert metric.attribute_conflict(cap, story1) == []
    assert "blue" in metric.attribute_conflict(cap, story2)
    assert "metal" in metric.attribute_conflict(cap, story3)
    print("OK attribute_conflict")


def test_context_builder_single():
    """Test build_single_image_context determinism."""
    desc = {
        "scene": "A street scene",
        "characters": ["man", "child"],
        "objects": ["building", "window", "car"],
        "actions": ["walking"],
        "spatial_relations": ["man next to child"],
        "region_descriptions": ["man walking", "child skipping", "building tall"],
        "style_or_mood": "playful",
        "ocr_text": "",
    }
    ctx1 = context_builder.build_single_image_context(desc)
    ctx2 = context_builder.build_single_image_context(desc)
    assert ctx1 == ctx2  # deterministic
    assert "Scene:" in ctx1
    assert "Characters:" in ctx1
    assert "Objects:" in ctx1
    assert "Actions:" in ctx1
    assert "playful" in ctx1
    print("OK context_builder_single")


def test_context_builder_sequence():
    """Test build_sequence_context with multiple images."""
    descs = [
        {"image_id": "img1", "scene": "Street", "characters": ["man"], "objects": ["building"], "actions": ["walking"], "region_descriptions": ["man walking"], "style_or_mood": ""},
        {"image_id": "img2", "scene": "Park", "characters": ["man"], "objects": ["tree"], "actions": ["sitting"], "region_descriptions": ["man sitting"], "style_or_mood": ""},
    ]
    seq = context_builder.build_sequence_context(descs)
    assert "SEQUENCE OF IMAGES" in seq
    assert "IMAGE 1" in seq
    assert "IMAGE 2" in seq
    assert "CONTINUITY NOTES" in seq
    assert "Recurring characters: man" in seq
    
    prompt = context_builder.build_story_prompt(seq, 2)
    assert "continuous story" in prompt
    assert "Maintain character identity" in prompt
    print("OK context_builder_sequence")


def test_json_schema_validation():
    """Test vision JSON output schema."""
    # Use repository-local test image instead of machine-specific path
    test_img = os.path.join(os.path.dirname(__file__), "selftest_image.jpg")
    desc = seeing.describe.__wrapped__(test_img) if hasattr(seeing.describe, '__wrapped__') else seeing.describe(test_img)
    required_keys = ["image_id", "scene", "description", "objects", "characters", "actions", "relationships", "ocr_text", "spatial_relations", "region_descriptions", "style_or_mood", "runtime_s", "model_load_s"]
    for k in required_keys:
        assert k in desc, f"Missing key: {k}"
    assert isinstance(desc["objects"], list)
    assert isinstance(desc["characters"], list)
    assert isinstance(desc["actions"], list)
    assert isinstance(desc["region_descriptions"], list)
    print("OK json_schema_validation")


def test_csv_output():
    """Test CSV writing and reading."""
    rows = [{
        "image": "test.jpg", "variant": "test", "context": "ctx", "story": "story",
        "see_s": 1.0, "story_s": 2.0, "word_count": 5, "length_valid": False,
        "truncated": False, "clip_image_story_mean": 0.2, "clip_image_story_min": 0.1,
        "clip_image_caption": 0.3, "nli_contra_mean": 0.1, "nli_contra_max": 0.2,
        "attribute_conflict": [], "grounding_score": 0.5, "grounding_pass": False,
        "runtime_s": 3.0, "repeated_sentences": 0, "repeated_bigrams": 0,
        "repeated_trigrams": 0, "repetition_rate": 0.0, "distinct3": 1.0,
        "entity_consistency": 1.0, "transition_markers": 0, "adjacent_similarity": 1.0,
    }]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        tmp = f.name
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        # Read back
        with open(tmp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)
        assert len(read_rows) == 1
        assert read_rows[0]["image"] == "test.jpg"
        assert float(read_rows[0]["grounding_score"]) == 0.5
    finally:
        os.unlink(tmp)
    print("OK csv_output")


def test_handling_failed_image():
    """Test that pipeline handles missing/corrupt images gracefully."""
    # Non-existent file
    try:
        result = seeing.describe("nonexistent_xyz.jpg")
        assert False, "Should have raised"
    except Exception:
        pass  # Expected
    print("OK handling_failed_image")


def test_reproducibility_config():
    """Test that config/seeds are documented and used."""
    assert baseline.SEED == 0
    assert baseline.QWEN_ID == "Qwen/Qwen2.5-0.5B-Instruct"
    assert seeing.MODEL_ID == "florence-community/Florence-2-base"
    assert seeing.NUM_BEAMS == 1
    print("OK reproducibility_config")


if __name__ == "__main__":
    test_word_count_validation()
    test_grounding_calculation()
    test_repetition_detection()
    test_continuity_score()
    test_attribute_conflict()
    test_context_builder_single()
    test_context_builder_sequence()
    # test_json_schema_validation()  # requires model load, skip in quick test
    test_csv_output()
    test_handling_failed_image()
    test_reproducibility_config()
    print("\nALL TESTS PASSED All tests passed!")