"""Token and cost accounting from the API's ``usage`` fields.

citecheck ships no prices. Model prices change, differ by contract and differ
between platforms, so a hard-coded table would quietly go stale. Instead you
pass a small TOML file with the prices you pay (per million tokens)::

    currency = "USD"

    [models."claude-opus-5-5"]
    input_per_mtok = ...         # uncached input tokens
    output_per_mtok = ...        # output tokens (thinking included)
    cache_write_per_mtok = ...   # tokens written to the 5-minute prompt cache
    cache_read_per_mtok = ...    # tokens read from the prompt cache

Amounts are handled as :class:`decimal.Decimal`, so per-token prices such as
0.000004 add up exactly. A model missing from the table yields no total at all
(and names the missing model) rather than a silently low one.
"""

import math
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from citecheck.errors import PricingError
from citecheck.llm import ModelResponse, TokenUsage

PRICE_FIELDS: Final = (
    "input_per_mtok",
    "output_per_mtok",
    "cache_write_per_mtok",
    "cache_read_per_mtok",
)
_MILLION: Final = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelPrice:
    """Prices for one model, per million tokens."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_write_per_mtok: Decimal
    cache_read_per_mtok: Decimal


@dataclass(frozen=True)
class PriceTable:
    """User-supplied prices for the models that may serve requests."""

    currency: str
    models: Mapping[str, ModelPrice]
    source: str


def _price(value: object, where: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PricingError(f"{where} must be a number")
    if not math.isfinite(value) or value < 0:
        raise PricingError(f"{where} must be a finite, non-negative number")
    return Decimal(str(value))


def parse_price_table(data: Mapping[str, Any], *, source: str) -> PriceTable:
    """Validate a parsed price table (for example the result of ``tomllib``).

    Raises:
        PricingError: On a missing, unknown or invalid field.
    """
    unknown_top = sorted(set(data) - {"currency", "models"})
    if unknown_top:
        raise PricingError(f"{source}: unknown top-level keys: {', '.join(unknown_top)}")
    currency = data.get("currency", "USD")
    if not isinstance(currency, str) or not currency.strip():
        raise PricingError(f"{source}: currency must be a non-empty string")
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise PricingError(f'{source}: add at least one [models."<model id>"] table')
    prices: dict[str, ModelPrice] = {}
    for name, entry in models.items():
        where = f'{source}: [models."{name}"]'
        if not isinstance(entry, dict):
            raise PricingError(f"{where} must be a table")
        if missing := [f for f in PRICE_FIELDS if f not in entry]:
            raise PricingError(f"{where} is missing {', '.join(missing)}")
        if unknown := sorted(set(entry) - set(PRICE_FIELDS)):
            raise PricingError(f"{where} has unknown keys: {', '.join(unknown)}")
        prices[name] = ModelPrice(*(_price(entry[f], f"{where}.{f}") for f in PRICE_FIELDS))
    return PriceTable(currency=currency.strip(), models=prices, source=source)


def load_price_table(path: Path) -> PriceTable:
    """Read and validate a TOML price table.

    Raises:
        PricingError: If the file is missing, is not valid TOML, or is invalid.
    """
    if not path.is_file():
        raise PricingError(f"price table not found: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise PricingError(f"{path.name}: invalid TOML ({exc})") from exc
    return parse_price_table(data, source=path.name)


@dataclass(frozen=True)
class UsageTotals:
    """Token totals over a set of responses.

    Attributes:
        calls: Number of Messages API calls (responses).
        input_tokens: Uncached input tokens.
        output_tokens: Output tokens.
        cache_creation_input_tokens: Input tokens written to the prompt cache.
        cache_read_input_tokens: Input tokens served from the prompt cache.
        estimated: True if any count is a fixture-mode estimate.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    estimated: bool = False

    @classmethod
    def from_responses(cls, responses: Iterable[ModelResponse]) -> "UsageTotals":
        """Sum the usage of every attempt in ``responses``."""
        responses = list(responses)
        entries = [entry for response in responses for entry in response.usage]
        return cls(
            calls=len(responses),
            input_tokens=sum(e.input_tokens for e in entries),
            output_tokens=sum(e.output_tokens for e in entries),
            cache_creation_input_tokens=sum(e.cache_creation_input_tokens for e in entries),
            cache_read_input_tokens=sum(e.cache_read_input_tokens for e in entries),
            estimated=any(response.usage_estimated for response in responses),
        )

    def __add__(self, other: "UsageTotals") -> "UsageTotals":
        """Combine totals from two sets of responses."""
        return UsageTotals(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            estimated=self.estimated or other.estimated,
        )


@dataclass(frozen=True)
class CostEstimate:
    """The cost of a set of model attempts under a :class:`PriceTable`.

    Attributes:
        currency: The table's currency.
        total: The total, or None if any model was missing from the table.
        by_model: Cost per model that was priced.
        missing_models: Models that served requests but have no price.
        estimated_usage: True if the token counts were fixture-mode estimates,
            in which case the amount demonstrates the arithmetic only.
    """

    currency: str
    total: Decimal | None
    by_model: Mapping[str, Decimal]
    missing_models: tuple[str, ...]
    estimated_usage: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary (amounts as strings)."""
        return {
            "currency": self.currency,
            "total": None if self.total is None else str(self.total),
            "by_model": {model: str(amount) for model, amount in self.by_model.items()},
            "missing_models": list(self.missing_models),
            "estimated_usage": self.estimated_usage,
        }


def cost_of(usage: TokenUsage, price: ModelPrice) -> Decimal:
    """Cost of one attempt: each token class times its per-million price."""
    return (
        usage.input_tokens * price.input_per_mtok
        + usage.output_tokens * price.output_per_mtok
        + usage.cache_creation_input_tokens * price.cache_write_per_mtok
        + usage.cache_read_input_tokens * price.cache_read_per_mtok
    ) / _MILLION


def estimate_cost(responses: Iterable[ModelResponse], table: PriceTable) -> CostEstimate:
    """Price every attempt in ``responses`` by the model that actually ran it."""
    by_model: dict[str, Decimal] = {}
    missing: dict[str, None] = {}
    estimated = False
    for response in responses:
        estimated = estimated or response.usage_estimated
        for usage in response.usage:
            price = table.models.get(usage.model)
            if price is None:
                missing.setdefault(usage.model, None)
                continue
            by_model[usage.model] = by_model.get(usage.model, Decimal(0)) + cost_of(usage, price)
    total = None if missing else sum(by_model.values(), Decimal(0))
    return CostEstimate(
        currency=table.currency,
        total=total,
        by_model=by_model,
        missing_models=tuple(missing),
        estimated_usage=estimated,
    )
