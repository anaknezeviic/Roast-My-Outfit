"""Pydantic records exchanged between the pipeline stages.

Enum members are ``str`` subclasses: interpolate ``.value``, never the member itself.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "SCHEMA_VERSION",
    "model_config",
    "Provenance",
    "GarmentSlot",
    "Pattern",
    "Fabric",
    "SleeveLength",
    "LowerLength",
    "Neckline",
    "ColorName",
    "ColorLabSource",
    "NEUTRAL_COLORS",
    "Garment",
    "OutfitDescription",
]

SCHEMA_VERSION = "1.0.0"

model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)


class Provenance(str, Enum):
    """Where the attributes on a record came from."""

    gt = "gt"
    predicted = "predicted"
    fixture = "fixture"


class GarmentSlot(str, Enum):
    """Body region or accessory class a garment occupies."""

    upper = "upper"
    outer = "outer"
    lower = "lower"
    dress = "dress"
    romper = "romper"
    footwear = "footwear"
    headwear = "headwear"
    neckwear = "neckwear"
    eyewear = "eyewear"
    bag = "bag"
    belt = "belt"
    socks = "socks"
    gloves = "gloves"
    jewelry = "jewelry"
    other = "other"


class Pattern(str, Enum):
    """Surface pattern printed or woven into a garment."""

    floral = "floral"
    graphic = "graphic"
    striped = "striped"
    pure_color = "pure_color"
    lattice = "lattice"
    color_block = "color_block"
    other = "other"
    na = "na"


class Fabric(str, Enum):
    """Material a garment is made from."""

    denim = "denim"
    cotton = "cotton"
    leather = "leather"
    furry = "furry"
    knitted = "knitted"
    chiffon = "chiffon"
    other = "other"
    na = "na"


class SleeveLength(str, Enum):
    """Sleeve length of an upper, outer or dress garment."""

    sleeveless = "sleeveless"
    short = "short"
    medium = "medium"
    long = "long"
    not_long = "not_long"
    na = "na"


class LowerLength(str, Enum):
    """Hem length of a lower garment or dress."""

    three_point = "three_point"
    medium_short = "medium_short"
    three_quarter = "three_quarter"
    long = "long"
    na = "na"


class Neckline(str, Enum):
    """Neckline shape of an upper, outer or dress garment."""

    v_shape = "v_shape"
    square = "square"
    round = "round"
    standing = "standing"
    lapel = "lapel"
    suspenders = "suspenders"
    na = "na"


class ColorName(str, Enum):
    """Coarse colour vocabulary that CIELAB centroids are quantised onto."""

    red = "red"
    orange = "orange"
    yellow = "yellow"
    chartreuse = "chartreuse"
    green = "green"
    spring_green = "spring_green"
    cyan = "cyan"
    azure = "azure"
    blue = "blue"
    violet = "violet"
    magenta = "magenta"
    rose = "rose"
    black = "black"
    white = "white"
    gray = "gray"
    beige = "beige"
    brown = "brown"
    navy = "navy"
    unknown = "unknown"


ColorLabSource = Literal["mask", "wholeimage"]

NEUTRAL_COLORS: frozenset[ColorName] = frozenset(
    {
        ColorName.black,
        ColorName.white,
        ColorName.gray,
        ColorName.beige,
        ColorName.brown,
        ColorName.navy,
    }
)


class Garment(BaseModel):
    """One garment or accessory seen in an outfit."""

    model_config = model_config

    ref: str = ""
    slot: GarmentSlot
    category: str = Field(min_length=1, max_length=64)
    color: ColorName = ColorName.unknown
    color_lab: tuple[float, float, float] | None = None
    color_lab_source: ColorLabSource | None = None
    area_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    pattern: Pattern = Pattern.na
    fabric: Fabric = Fabric.na
    sleeve_length: SleeveLength | None = None
    length: LowerLength | None = None
    neckline: Neckline | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class OutfitDescription(BaseModel):
    """Everything the pipeline knows about one outfit photograph."""

    model_config = model_config

    image_id: str = Field(min_length=1)
    image_path: str = ""
    garments: list[Garment] = Field(min_length=1)
    caption: str = Field(default="", max_length=2000)
    provenance: Provenance = Provenance.predicted
    source_model: str = Field(min_length=1)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _resolve_refs(self) -> OutfitDescription:
        """Number blank refs per slot, then reject duplicates."""
        seen_per_slot: Counter[GarmentSlot] = Counter()
        for garment in self.garments:
            index = seen_per_slot[garment.slot]
            seen_per_slot[garment.slot] += 1
            if not garment.ref:
                garment.ref = f"{garment.slot.value}_{index}"

        duplicates = sorted(ref for ref, count in Counter(self.refs()).items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate garment refs: {', '.join(duplicates)}.")
        return self

    def refs(self) -> list[str]:
        """Return every garment ref, in garment order."""
        return [garment.ref for garment in self.garments]

    def by_slot(self, slot: GarmentSlot) -> list[Garment]:
        """Return the garments filling one slot, in garment order."""
        return [garment for garment in self.garments if garment.slot == slot]
