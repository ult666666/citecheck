from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from citecheck.costs import (
    ModelPrice,
    PriceTable,
    UsageTotals,
    estimate_cost,
    load_price_table,
    parse_price_table,
)
from citecheck.errors import PricingError
from citecheck.llm import ModelResponse, TokenUsage

ENTRY = {
    "input_per_mtok": 3.0,
    "output_per_mtok": 15,
    "cache_write_per_mtok": 3.75,
    "cache_read_per_mtok": 0.3,
}


def table_data(**model_overrides: Any) -> dict[str, Any]:
    return {"currency": "USD", "models": {"m1": {**ENTRY, **model_overrides}}}


def response(*usage: TokenUsage, estimated: bool = False) -> ModelResponse:
    return ModelResponse(
        model=usage[0].model,
        stop_reason="tool_use",
        tool_calls=(),
        text="",
        usage=usage,
        usage_estimated=estimated,
    )


def test_parse_a_valid_table_into_decimals() -> None:
    table = parse_price_table(table_data(), source="p.toml")
    assert table.currency == "USD"
    assert table.models["m1"] == ModelPrice(
        Decimal("3.0"), Decimal("15"), Decimal("3.75"), Decimal("0.3")
    )
    assert table.source == "p.toml"


def test_currency_defaults_to_usd() -> None:
    assert parse_price_table({"models": {"m1": ENTRY}}, source="p").currency == "USD"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"models": {"m1": ENTRY}, "extra": 1}, "unknown top-level keys: extra"),
        ({"currency": "", "models": {"m1": ENTRY}}, "currency"),
        ({"currency": "USD"}, "at least one"),
        ({"models": {}}, "at least one"),
        ({"models": {"m1": 5}}, "must be a table"),
        ({"models": {"m1": {"input_per_mtok": 1}}}, "is missing output_per_mtok"),
        (table_data(discount=0.1), "unknown keys: discount"),
        (table_data(input_per_mtok=-1), "non-negative"),
        (table_data(input_per_mtok=float("inf")), "non-negative"),
        (table_data(input_per_mtok=True), "must be a number"),
        (table_data(input_per_mtok="3"), "must be a number"),
    ],
)
def test_invalid_tables_are_rejected(data: dict[str, Any], message: str) -> None:
    with pytest.raises(PricingError, match=message):
        parse_price_table(data, source="p.toml")


def test_load_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "prices.toml"
    path.write_text(
        'currency = "EUR"\n[models."m1"]\ninput_per_mtok = 1\noutput_per_mtok = 2\n'
        "cache_write_per_mtok = 1.25\ncache_read_per_mtok = 0.1\n",
        encoding="utf-8",
    )
    table = load_price_table(path)
    assert table.currency == "EUR"
    assert table.source == "prices.toml"


def test_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(PricingError, match="not found"):
        load_price_table(tmp_path / "nope.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("models = [unclosed", encoding="utf-8")
    with pytest.raises(PricingError, match="invalid TOML"):
        load_price_table(bad)


def test_cost_is_exact_and_split_by_model() -> None:
    table = PriceTable(
        currency="USD",
        models={
            "m1": parse_price_table(table_data(), source="p").models["m1"],
            "m2": ModelPrice(Decimal(1), Decimal(1), Decimal(1), Decimal(1)),
        },
        source="p",
    )
    cost = estimate_cost(
        [
            response(TokenUsage("m1", 1_000_000, 0)),
            response(TokenUsage("m1", 0, 1000, 2000, 10_000), TokenUsage("m2", 500, 500)),
        ],
        table,
    )
    # m1: 3.00 + (1000*15 + 2000*3.75 + 10000*0.3) / 1e6 = 3.0255; m2: 1000 / 1e6 = 0.001
    assert cost.by_model == {"m1": Decimal("3.0255"), "m2": Decimal("0.001")}
    assert cost.total == Decimal("3.0265")
    assert cost.missing_models == ()
    assert cost.to_dict()["total"] == "3.0265"


def test_a_model_missing_from_the_table_means_no_total() -> None:
    table = parse_price_table(table_data(), source="p")
    cost = estimate_cost(
        [response(TokenUsage("m1", 10, 10)), response(TokenUsage("x", 1, 1))], table
    )
    assert cost.total is None
    assert cost.missing_models == ("x",)
    assert cost.to_dict()["total"] is None


def test_estimated_usage_is_flagged() -> None:
    table = parse_price_table(table_data(), source="p")
    assert estimate_cost([response(TokenUsage("m1", 1, 1), estimated=True)], table).estimated_usage


def test_usage_totals_sum_attempts_and_combine() -> None:
    first = UsageTotals.from_responses(
        [response(TokenUsage("m1", 10, 2, 3, 4), TokenUsage("m2", 1, 1))]
    )
    assert (first.calls, first.input_tokens, first.output_tokens) == (1, 11, 3)
    second = UsageTotals.from_responses([response(TokenUsage("m1", 5, 5), estimated=True)])
    combined = first + second
    assert combined == UsageTotals(
        calls=2,
        input_tokens=16,
        output_tokens=8,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=4,
        estimated=True,
    )
