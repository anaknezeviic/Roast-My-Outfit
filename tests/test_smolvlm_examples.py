from pathlib import Path

from rmo.data.smolvlm_examples import SmolVLMExampleBuilder


def test_build_training_example() -> None:
    builder = SmolVLMExampleBuilder(
        raw_dir=Path("data/raw"),
    )

    example = builder.build(
        "MEN-Denim-id_00000080-01_7_additional"
    )

    assert example["image_id"] == "MEN-Denim-id_00000080-01_7_additional"
    assert Path(example["image_path"]).exists()

    garments = example["answer"]["garments"]

    assert len(garments) == 2

    upper = garments[0]
    assert upper["slot"] == "upper"
    assert upper["fabric"] == "cotton"
    assert upper["pattern"] == "pure_color"
    assert upper["sleeve_length"] == "na"
    assert upper["neckline"] == "round"

    lower = garments[1]
    assert lower["slot"] == "lower"
    assert lower["fabric"] == "cotton"
    assert lower["pattern"] == "lattice"
    assert lower["length"] == "long"