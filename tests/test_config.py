"""Cover configuration loading and the shipped perception config."""

from __future__ import annotations

import string
from enum import Enum

import pytest

from rmo import paths
from rmo.config import (
    ConfigError,
    load_perception_config,
    load_roast_config,
    load_scoring_config,
)
from rmo.perception.postprocess import _normalise
from rmo.schemas import (
    ColorName,
    Fabric,
    GarmentSlot,
    LowerLength,
    Neckline,
    OutfitDescription,
    Pattern,
    SleeveLength,
    SubScores,
)
from rmo.scoring.rules import category_key

VOCABULARIES: dict[str, type[Enum]] = {
    "slot": GarmentSlot,
    "color": ColorName,
    "pattern": Pattern,
    "fabric": Fabric,
    "sleeve_length": SleeveLength,
    "length": LowerLength,
    "neckline": Neckline,
}

SCORING_SECTIONS = frozenset(
    {
        "palette",
        "color",
        "formality",
        "season",
        "harmony",
        "proportion",
        "weights",
        "penalties",
        "defaults",
    }
)

# DeepFashion-MultiModal parses 24 classes; 0 background, 13 hair, 14 face and 15 skin are not worn
PARSING_CLASSES = frozenset(range(24))
NON_GARMENT_CLASSES = frozenset({0, 13, 14, 15})


def spelled(enum_type: type[Enum]) -> set[str]:
    return {_normalise(str(member.value)) for member in enum_type}


@pytest.fixture()
def shipped():
    return load_perception_config()


@pytest.fixture()
def shipped_roast():
    return load_roast_config()


LOADERS = (load_perception_config, load_roast_config, load_scoring_config)


@pytest.mark.parametrize("loader", LOADERS, ids=lambda fn: fn.__name__)
def test_missing_file_raises_config_error(loader, tmp_path) -> None:
    with pytest.raises(ConfigError, match="Could not read configuration"):
        loader(tmp_path / "absent.yaml")


@pytest.mark.parametrize("loader", LOADERS, ids=lambda fn: fn.__name__)
def test_non_mapping_payload_raises_config_error(loader, tmp_path) -> None:
    path = tmp_path / f"{loader.__name__}_list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        loader(path)


@pytest.mark.parametrize("loader", LOADERS, ids=lambda fn: fn.__name__)
def test_invalid_yaml_raises_config_error(loader, tmp_path) -> None:
    path = tmp_path / f"{loader.__name__}_broken.yaml"
    path.write_text("a: [1,\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        loader(path)


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


def test_shipped_prompt_requires_a_pattern_for_every_item(shipped) -> None:
    prompt = shipped["prompt"].lower()
    assert "every item" in prompt
    assert "pattern" in prompt
    assert "plain" in prompt


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


def test_default_roast_path_is_the_repo_config(shipped_roast) -> None:
    assert isinstance(shipped_roast, dict)
    assert (paths.configs_dir() / "roast.yaml").is_file()


def test_roast_result_is_cached_per_path(tmp_path) -> None:
    path = tmp_path / "roast_cached.yaml"
    path.write_text("persona: hello\n", encoding="utf-8")
    assert load_roast_config(path) is load_roast_config(path)


def test_roast_cache_clear_reloads_from_disk(tmp_path) -> None:
    path = tmp_path / "roast_reloaded.yaml"
    path.write_text("persona: first\n", encoding="utf-8")
    assert load_roast_config(path)["persona"] == "first"
    load_roast_config.cache_clear()
    path.write_text("persona: second\n", encoding="utf-8")
    assert load_roast_config(path)["persona"] == "second"


def test_shipped_roast_config_has_exactly_two_keys(shipped_roast) -> None:
    assert set(shipped_roast) == {"persona", "prompt_template"}


@pytest.mark.parametrize(
    "placeholder", ["{tone}", "{caption}", "{garments}", "{issues}"]
)
def test_shipped_roast_template_carries_every_placeholder(
    shipped_roast, placeholder: str
) -> None:
    assert placeholder in shipped_roast["prompt_template"]


def test_shipped_roast_template_has_no_other_braces(shipped_roast) -> None:
    fields = [
        name
        for _, name, _, _ in string.Formatter().parse(shipped_roast["prompt_template"])
        if name is not None
    ]
    assert fields == ["tone", "caption", "garments", "issues"]


def test_shipped_persona_has_no_braces(shipped_roast) -> None:
    assert "{" not in shipped_roast["persona"]


def test_shipped_persona_forbids_commenting_on_the_person(shipped_roast) -> None:
    persona = shipped_roast["persona"]
    assert "Roast the clothes and styling only, never the person wearing them." in persona
    assert "Do not comment on:" in persona


def test_scoring_default_path_is_the_repo_config() -> None:
    assert isinstance(load_scoring_config(), dict)
    assert (paths.configs_dir() / "scoring.yaml").is_file()


def test_every_loader_caches_independently(tmp_path) -> None:
    path = tmp_path / "independent.yaml"
    path.write_text("value: first\n", encoding="utf-8")
    perception = load_perception_config(path)
    roast = load_roast_config(path)
    scoring = load_scoring_config(path)
    assert len({id(perception), id(roast), id(scoring)}) == 3

    path.write_text("value: second\n", encoding="utf-8")
    load_scoring_config.cache_clear()
    assert load_scoring_config(path)["value"] == "second"
    assert load_perception_config(path) is perception
    assert load_roast_config(path) is roast


def test_shipped_scoring_config_has_a_palette_section() -> None:
    palette = load_scoring_config()["palette"]
    assert {"seed", "n_clusters", "n_init", "min_area_fraction", "max_fit_pixels"} <= set(palette)


def test_shipped_scoring_config_has_every_scoring_section() -> None:
    assert SCORING_SECTIONS <= set(load_scoring_config())


def test_shipped_weights_cover_exactly_the_subscore_axes() -> None:
    assert set(load_scoring_config()["weights"]) == set(SubScores.model_fields)


def test_shipped_formality_keys_are_normalised_and_in_range() -> None:
    levels = load_scoring_config()["formality"]["levels"]
    assert all(key == category_key(key) for key in levels)
    assert all(isinstance(value, int) and 0 <= value <= 4 for value in levels.values())


def test_shipped_formality_table_covers_every_fixture_category(
    descriptions: dict[str, OutfitDescription],
) -> None:
    levels = load_scoring_config()["formality"]["levels"]
    used = {
        category_key(garment.category)
        for description in descriptions.values()
        for garment in description.garments
    }
    assert used - set(levels) == {"unknown"}


@pytest.mark.parametrize(
    ("key", "enum_type"),
    [
        ("fabric_warmth", Fabric),
        ("sleeve_exposure", SleeveLength),
        ("length_exposure", LowerLength),
    ],
)
def test_shipped_season_tables_score_real_members_other_than_na(
    key: str, enum_type: type[Enum]
) -> None:
    scored = set(load_scoring_config()["season"][key])
    assert scored <= {str(member.value) for member in enum_type}
    assert "na" not in scored


@pytest.mark.parametrize(
    "key", ["accessory_slots", "dominant_slots", "core_slots"]
)
def test_shipped_proportion_lists_name_real_slots(key: str) -> None:
    listed = load_scoring_config()["proportion"][key]
    assert listed
    assert set(listed) <= {slot.value for slot in GarmentSlot}


def test_shipped_relation_bands_do_not_overlap() -> None:
    color = load_scoring_config()["color"]
    assert color["analogous_max"] < 120.0 - color["triadic_tolerance"]
    assert 120.0 + color["triadic_tolerance"] < 180.0 - color["complementary_tolerance"]
