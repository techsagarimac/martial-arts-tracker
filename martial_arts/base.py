"""Martial-arts style plugins. Rules stay in the detector; styles only enable techniques."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TechniqueSpec:
    id: str
    name: str
    category: str  # punch | kick | other


class StylePlugin(Protocol):
    slug: str
    display_name: str

    def enabled_techniques(self) -> list[TechniqueSpec]:
        ...


PUNCHES = (
    TechniqueSpec("jab", "Jab", "punch"),
    TechniqueSpec("cross", "Cross", "punch"),
    TechniqueSpec("hook", "Hook", "punch"),
    TechniqueSpec("straight_punch", "Straight punch", "punch"),
)

KICKS = (
    TechniqueSpec("front_kick", "Front kick", "kick"),
    TechniqueSpec("roundhouse_kick", "Roundhouse-style kick", "kick"),
    TechniqueSpec("side_kick", "Side kick", "kick"),
)


class BoxingStyle:
    slug = "boxing"
    display_name = "Boxing"

    def enabled_techniques(self) -> list[TechniqueSpec]:
        return list(PUNCHES)


class KickboxingStyle:
    slug = "kickboxing"
    display_name = "Kickboxing"

    def enabled_techniques(self) -> list[TechniqueSpec]:
        return list(PUNCHES) + list(KICKS)


class KarateStyle:
    slug = "karate"
    display_name = "Karate"

    def enabled_techniques(self) -> list[TechniqueSpec]:
        return list(PUNCHES) + list(KICKS)


class TaekwondoStyle:
    slug = "taekwondo"
    display_name = "Taekwondo"

    def enabled_techniques(self) -> list[TechniqueSpec]:
        return list(KICKS) + list(PUNCHES)


class MuayThaiStyle:
    slug = "muay_thai"
    display_name = "Muay Thai"

    def enabled_techniques(self) -> list[TechniqueSpec]:
        # Elbows/knees are intentionally not classified yet.
        return list(PUNCHES) + list(KICKS)


class MmaStyle:
    slug = "mma"
    display_name = "MMA"

    def enabled_techniques(self) -> list[TechniqueSpec]:
        return list(PUNCHES) + list(KICKS)


class KungFuStyle:
    slug = "kung_fu"
    display_name = "Kung Fu"

    def enabled_techniques(self) -> list[TechniqueSpec]:
        return list(PUNCHES) + list(KICKS)


STYLES: dict[str, StylePlugin] = {
    "boxing": BoxingStyle(),
    "kickboxing": KickboxingStyle(),
    "karate": KarateStyle(),
    "taekwondo": TaekwondoStyle(),
    "muay_thai": MuayThaiStyle(),
    "mma": MmaStyle(),
    "kung_fu": KungFuStyle(),
}

STYLE_LABELS = [style.display_name for style in STYLES.values()]


def get_style(name: str) -> StylePlugin:
    key = (name or "kickboxing").strip().lower().replace(" ", "_")
    aliases = {
        "muaythai": "muay_thai",
        "kungfu": "kung_fu",
        "tae_kwon_do": "taekwondo",
    }
    key = aliases.get(key, key)
    return STYLES.get(key, STYLES["kickboxing"])


def slug_from_display(display: str) -> str:
    for slug, style in STYLES.items():
        if style.display_name.lower() == display.lower():
            return slug
    return get_style(display).slug
