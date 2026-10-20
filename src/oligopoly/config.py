"""Model and experiment configuration (`config/models.toml`, `config/experiment.toml`)."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .llm import DEFAULT_BYTES_PER_TOKEN_MIN

ROOT = Path(__file__).resolve().parents[2]
# part orderings: name -> (part 1 has chat, part 2 has chat)
ORDERINGS: dict[str, tuple[bool, bool]] = {"NT-T": (False, True), "T-NT": (True, False), "NT-NT": (False, False)}
CONFIG_DIR = ROOT / "config"
MOCK_PROVIDER = "mock"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    name: str  # table name: the short name used on the command line and in file names
    model: str
    provider: str
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    extra: dict[str, Any] | None = None  # other sampling parameters sent as-is (e.g. presence_penalty)
    max_tokens: int | None = None
    reasoning: dict[str, Any] | None = None
    require_parameters: bool = True
    cache_control: bool = True
    # OpenRouter structured outputs (json_schema, strict) for chat and price calls; needs require_parameters so
    # the pinned provider must honour it (DECISIONS.md, 2026-10-20). Part of the model's identity (fingerprints).
    structured_outputs: bool = False
    pricing: dict[str, float] | None = None  # USD per million tokens, dry-run only
    timeout_s: float | None = None  # HTTP timeout per call (client default 900 s)
    # Context limits (DECISIONS.md, 2026-10-20): cells this model skips, as (ordering, n) pairs, and the lowest
    # UTF-8 bytes per prompt token assumed by the runtime context guard (llm.prompt_token_bound).
    exclude_cells: tuple[tuple[str, int], ...] = ()
    bytes_per_token_min: float = DEFAULT_BYTES_PER_TOKEN_MIN

    # settings that do not affect play; left out of fingerprints
    OPERATIONAL = ("pricing", "timeout_s", "exclude_cells", "bytes_per_token_min")

    def __post_init__(self) -> None:
        cells = []
        for c in self.exclude_cells if isinstance(self.exclude_cells, (list, tuple)) else [self.exclude_cells]:
            ok = (isinstance(c, (list, tuple)) and len(c) == 2 and c[0] in ORDERINGS and isinstance(c[1], int)
                  and not isinstance(c[1], bool) and c[1] >= 2)
            if not ok:
                raise ConfigError(f"model {self.name!r}: bad exclude_cells entry {c!r}; use [ordering, n] with "
                                  f"ordering in {list(ORDERINGS)} and integer n >= 2")
            cells.append((c[0], c[1]))
        object.__setattr__(self, "exclude_cells", tuple(cells))  # TOML gives lists
        if self.structured_outputs and not self.require_parameters:
            raise ConfigError(f"model {self.name!r}: structured_outputs needs require_parameters = true")
        b = self.bytes_per_token_min
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not b > 0:
            raise ConfigError(f"model {self.name!r}: bytes_per_token_min must be a positive number, got {b!r}")

    def excludes(self, n: int, ordering: str) -> bool:
        return (ordering, n) in self.exclude_cells

    @property
    def is_mock(self) -> bool:
        return self.provider == MOCK_PROVIDER

    @property
    def provider_pending(self) -> bool:
        return self.provider.strip().upper() in ("", "TODO")

    def check_live(self) -> None:
        """Refuse a live run before the provider probe has pinned an endpoint."""
        if self.provider_pending:
            raise ConfigError(f"model {self.name!r} has provider = {self.provider!r}; pin an endpoint first")

    def chat_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `OpenRouterClient.chat` (top_k and `extra` go through extra_body)."""
        extra_body = {**({"top_k": self.top_k} if self.top_k is not None else {}), **(self.extra or {})}
        return {
            "model": self.model, "provider": self.provider, "temperature": self.temperature, "top_p": self.top_p,
            "max_tokens": self.max_tokens, "reasoning": self.reasoning,
            "require_parameters": self.require_parameters,
            "extra_body": extra_body or None,
            "timeout_s": self.timeout_s,
            "bytes_per_token_min": self.bytes_per_token_min,
        }

    def identity(self) -> dict[str, Any]:
        """The settings that determine play (everything but OPERATIONAL), for run and market fingerprints."""
        return {k: v for k, v in asdict(self).items() if k not in self.OPERATIONAL}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    part_lengths: tuple[int, int] = (29, 24)
    min_periods: int = 20
    chat_rounds: int = 3
    char_limit: int = 300
    max_reasks: int = 2
    notes: bool = True
    # post-game comprehension probe after the final period (DECISIONS.md, 2026-10-20); adds calls and the
    # record's "probe" entry, never changes play
    post_game_probe: bool = True
    # completed periods each request shows verbatim; older play survives only in the firm's own notes
    # (DECISIONS.md §6 item 3, 2026-10-20)
    memory_window: int = 3
    # hard cap on the notes a firm carries as memory and in its windowed answers (DECISIONS.md engine details
    # 12): longer notes are cut at a word boundary and marked " [truncated]"; the instructions ask for ~150 words
    notes_max_words: int = 200
    notes_max_chars: int = 1400

    def __post_init__(self) -> None:
        for name, low in (("memory_window", 1), ("notes_max_words", 1), ("notes_max_chars", 50)):
            w = getattr(self, name)
            if isinstance(w, bool) or not isinstance(w, int) or w < low:
                raise ConfigError(f"{name} must be an integer >= {low}, got {w!r}")

    @property
    def pilot(self) -> bool:
        """Parts shorter than min_periods: the die schedule adapts, the instructions do not (pilot only)."""
        return min(self.part_lengths) < self.min_periods

    def part_min_periods(self, part: int) -> int:
        return min(self.min_periods, self.part_lengths[part - 1])

    def with_part_lengths(self, lengths: tuple[int, int]) -> ExperimentConfig:
        if len(lengths) != 2 or min(lengths) < 1:
            raise ConfigError(f"need two positive part lengths, got {lengths}")
        return replace(self, part_lengths=tuple(lengths))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_models(path: str | Path = CONFIG_DIR / "models.toml") -> dict[str, ModelConfig]:
    data = tomllib.loads(Path(path).read_text())
    known = set(ModelConfig.__dataclass_fields__) - {"name"}
    out = {}
    for name, t in data.items():
        if unknown := set(t) - known:
            raise ConfigError(f"model {name!r}: unknown keys {sorted(unknown)}")
        if "model" not in t or "provider" not in t:
            raise ConfigError(f"model {name!r} needs 'model' and 'provider'")
        out[name] = ModelConfig(name=name, **t)
    return out


def load_experiment(path: str | Path = CONFIG_DIR / "experiment.toml") -> ExperimentConfig:
    t = tomllib.loads(Path(path).read_text())
    known = set(ExperimentConfig.__dataclass_fields__)
    if unknown := set(t) - known:
        raise ConfigError(f"experiment config: unknown keys {sorted(unknown)}")
    if "part_lengths" in t:
        t["part_lengths"] = tuple(t["part_lengths"])
    return ExperimentConfig(**t)
