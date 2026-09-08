from pathlib import Path

from rmo.data.parse_annotations import (
    parse_fabric,
    parse_pattern,
    parse_shape,
)


class SmolVLMExampleBuilder:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir

        self.shape = parse_shape(
            raw_dir / "labels" / "shape" / "shape_anno_all.txt"
        ).set_index("image_id")

        self.fabric = parse_fabric(
            raw_dir / "labels" / "texture" / "fabric_ann.txt"
        ).set_index("image_id")

        self.pattern = parse_pattern(
            raw_dir / "labels" / "texture" / "pattern_ann.txt"
        ).set_index("image_id")

    def build(self, image_id: str) -> dict:
        image_path = self.raw_dir / "images" / f"{image_id}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        shape_row = self.shape.loc[image_id]
        fabric_row = self.fabric.loc[image_id]
        pattern_row = self.pattern.loc[image_id]

        prompt = (
            "Describe the visible garments in the image.\n\n"
            "For each garment, return:\n"
            "- slot\n"
            "- fabric\n"
            "- pattern\n"
            "- sleeve_length\n"
            "- length\n"
            "- neckline"
        )

        garments = []

        if (
            fabric_row["upper_fabric"] != "na"
            or pattern_row["upper_pattern"] != "na"
        ):
            garments.append(
                {
                    "slot": "upper",
                    "fabric": fabric_row["upper_fabric"],
                    "pattern": pattern_row["upper_pattern"],
                    "sleeve_length": shape_row["sleeve_length"],
                    "neckline": shape_row["neckline"],
                }
            )

        if (
            fabric_row["lower_fabric"] != "na"
            or pattern_row["lower_pattern"] != "na"
        ):
            garments.append(
                {
                    "slot": "lower",
                    "fabric": fabric_row["lower_fabric"],
                    "pattern": pattern_row["lower_pattern"],
                    "length": shape_row["lower_length"],
                }
            )

        if (
            fabric_row["outer_fabric"] != "na"
            or pattern_row["outer_pattern"] != "na"
        ):
            garments.append(
                {
                    "slot": "outer",
                    "fabric": fabric_row["outer_fabric"],
                    "pattern": pattern_row["outer_pattern"],
                }
            )

        return {
            "image_id": image_id,
            "image_path": str(image_path),
            "prompt": prompt,
            "answer": {
                "garments": garments,
            },
        }