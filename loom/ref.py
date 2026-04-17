from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict as TypingDict


@dataclass(frozen=True)
class Ref:
    """Ephemeral runtime reference to a record stored in a Dataset.

    This is NOT intended to be persisted.
    """

    dataset: Any
    addr: int

    def get(self) -> TypingDict[str, Any]:
        return self.dataset[self.addr]

    def set(self, record: TypingDict[str, Any]) -> None:
        self.dataset[self.addr] = record

    def update(self, **fields: Any) -> None:
        # Fast path: update individual fields when possible
        for field_name, value in fields.items():
            self.dataset.write_field(self.addr, field_name, value)

    def __repr__(self) -> str:
        ds_name = getattr(self.dataset, "name", "<unknown>")
        return f"Ref(dataset={ds_name!r}, addr={self.addr})"
