"""Append-only category IDs with explicit, cycle-checked synonyms."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from contracts import Category
from storage import atomic_json, read_json


def normalize(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("Category names must be strings")
    name = " ".join(unicodedata.normalize("NFKC", name).strip().casefold().split())
    if not name or len(name) > 80 or not re.fullmatch(r"[\w][\w /'()\-]*", name):
        raise ValueError(f"Invalid short category name: {name!r}")
    return name


class CategoryRegistry:
    def __init__(self, path: Path, synonyms: dict[str, str], stuff_classes: list[str]):
        self.path = path
        self.synonyms = {normalize(k): normalize(v) for k, v in synonyms.items()}
        for key in self.synonyms:
            self.canonical(key)
        self.stuff = {self.canonical(x) for x in stuff_classes}
        self.categories = [
            Category.model_validate(x) for x in read_json(path, {"categories": []})["categories"]
        ]
        names = [c.name for c in self.categories]
        ids = [c.id for c in self.categories]
        if len(names) != len(set(names)) or len(ids) != len(set(ids)):
            raise ValueError("Category registry contains duplicate names or IDs")
        if any(
            self.canonical(c.name) != c.name or c.isthing != int(c.name not in self.stuff)
            for c in self.categories
        ):
            raise ValueError("Category normalization/type changed; use a new job")

    def canonical(self, name: str) -> str:
        name, seen = normalize(name), set()
        while name in self.synonyms:
            if name in seen:
                raise ValueError(f"Synonym cycle at {name}")
            seen.add(name)
            replacement = self.synonyms[name]
            if replacement == name:
                break
            name = replacement
        return name

    def names(self, names: list[str]) -> list[str]:
        return sorted({self.canonical(name) for name in names})

    def register(self, names: list[str]) -> None:
        existing = {category.name for category in self.categories}
        next_id = max((c.id for c in self.categories), default=0) + 1
        for name in self.names(names):
            if name not in existing:
                self.categories.append(
                    Category(id=next_id, name=name, isthing=int(name not in self.stuff))
                )
                next_id += 1
        atomic_json(self.path, {"categories": [c.model_dump() for c in self.categories]})

    @property
    def by_name(self) -> dict[str, int]:
        return {category.name: category.id for category in self.categories}
