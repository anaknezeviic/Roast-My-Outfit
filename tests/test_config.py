"""Cover configuration loading and the shipped perception config."""

from __future__ import annotations

from enum import Enum

import pytest

from rmo import paths
from rmo.config import ConfigError, load_perception_config
from rmo.perception.postprocess import _normalise
from rmo.schemas import (
    ColorName,
    Fabric,
    GarmentSlot,
    LowerLength,
    Neckline,
    Pattern,
    SleeveLength,
)

VOCABULARIES: dict[str, type[Enum]] = {
    "slot": GarmentSlot,
    "color": ColorName,
    "pattern": Pattern,
    "fabric": Fabric,
    "sleeve_length": SleeveLength,
    "length": LowerLength,
    "neckline": Neckline,
}

# DeepFashion-MultiModal parses 24 classes; 0 background, 13 hair, 14 face and 15 skin are not worn
PARSING_CLASSES = frozenset(range(24))
NON_GARMENT_CLASSES = frozenset({0, 13, 14, 15})


def spelled(enum_type: type[Enum]) -> set[str]:
    return {_normalise(str(member.value)) for member in enum_type}


@pytest.fixture()
def shipped():
    return load_perception_config()


def test_missing_file_raises_config_error(tmp_path) -> None:
    with pytest.raises(ConfigError, match="Could not read configuration"):
        load_perception_config(tmp_path / "absent.yaml")


def test_non_mapping_payload_raises_config_error(tmp_path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_perception_config(path)


def test_invalid_yaml_raises_config_error(tmp_path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("a: [1,\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_perception_config(path)


def test_default_path_is_the_repo_config(shipped) -> None:
    assert isinstance(shipped, dict)
    assert (paths.configs_dir() / "perception.yaml").is_file()


def test_result_is_cached_per_path(tmp_path) -> None:
    path = tmp_path / "cached.yaml"
    path.write_text("prompt: hello\n", encoding="utf-8")
    assert load_perception_config(path) is load_perception_config(path)


def test_cache_clear_reloads_from_disk(tmp_path) -> None:
    path = tmp_path / "reloaded.yaml"
    path.write_text("prompt: first\n", encoding="utf-8")
    assert load_perception_config(path)["prompt"] == "first"
    load_perception_config.cache_clear()
    path.write_text("prompt: second\n", encoding="utf-8")
    assert load_perception_config(path)["prompt"] == "second"


def test_shipped_config_has_every_required_key(shipped) -> None:
    assert {"prompt", "generation", "synonyms", "mask_labels"} <= set(shipped)


def test_shipped_prompt_asks_about_clothing(shipped) -> None:
    prompt = shipped["prompt"].lower()
    assert "wearing" in prompt or "outfit" in prompt


@pytest.mark.parametrize("name", sorted(VOCABULARIES))
def test_shipped_prompt_does_not_enumerate_a_vocabulary(shipped, name: str) -> None:
    known = spelled(VOCABULARIES[name])
    for line in shipped["prompt"].splitlines():
        listed = {_normalise(part) for part in line.split(",")} & known
        assert len(listed) < 3, f"{name} vocabulary listed in prompt: {sorted(listed)}"


def test_shipped_prompt_has_no_template_placeholders(shipped) -> None:
    assert "<" not in shipped["prompt"]


def test_generation_is_deterministic(shipped) -> None:
    assert shipped["generation"]["do_sample"] is False


def test_mask_labels_use_known_slots(shipped) -> None:
    assert set(shipped["mask_labels"]) <= {slot.value for slot in GarmentSlot}


def test_mask_labels_are_disjoint(shipped) -> None:
    ids = [label for labels in shipped["mask_labels"].values() for label in labels]
    assert len(ids) == len(set(ids))


def test_mask_labels_claim_every_worn_parsing_class(shipped) -> None:
    claimed = {label for labels in shipped["mask_labels"].values() for label in labels}
    assert claimed == PARSING_CLASSES - NON_GARMENT_CLASSES


def test_synonym_tables_use_known_attribute_names(shipped) -> None:
    assert set(shipped["synonyms"]) == set(VOCABULARIES)


@pytest.mark.parametrize("name", sorted(VOCABULARIES))
def test_synonym_targets_are_vocabulary_members(shipped, name: str) -> None:
    known = spelled(VOCABULARIES[name])
    assert {_normalise(str(value)) for value in shipped["synonyms"][name].values()} <= known


@pytest.mark.parametrize("name", sorted(VOCABULARIES))
def test_synonym_keys_do_not_shadow_vocabulary_members(shipped, name: str) -> None:
    known = spelled(VOCABULARIES[name])
    assert not known & {str(key) for key in shipped["synonyms"][name]}
