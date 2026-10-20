"""Model and experiment configuration (`config/models.toml`, `config/experiment.toml`)."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
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
    max_tokens: int | None = None
    reasoning: dict[str, Any] | None = None
    require_parameters: bool = True
    cache_control: bool = True
    pricing: dict[str, float] | None = None  # USD per million tokens, dry-run only
    timeout_s: float | None = None  # HTTP timeout per call (client default 900 s)

    OPERATIONAL = ("pricing", "timeout_s")  # settings that do not affect play; left out of fingerprints

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
        """Keyword arguments for `OpenRouterClient.chat` (top_k goes through extra_body)."""
        return {
            "model": self.model, "provider": self.provider, "temperature": self.temperature, "top_p": self.top_p,
            "max_tokens": self.max_tokens, "reasoning": self.reasoning,
            "require_parameters": self.require_parameters,
            "extra_body": {"top_k": self.top_k} if self.top_k is not None else None,
            "timeout_s": self.timeout_s,
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
