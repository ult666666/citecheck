"""Claim-level checks: split a sentence into claims and compare each with its source clause.

The lexical checker asks whether a sentence's words appear somewhere in the
passages it cites. That cannot see a flipped negation, a swapped "before" and
"after", or "30 days" moved from one rule to another. This module splits a
sentence into claims and compares each claim with its *aligned clause*: the
clause of the cited passages that shares the most content words with it
(numbers and negation words do not count toward the match, so changing them
cannot move the alignment). Four checks then run:

1. **Polarity.** Exactly one of the claim and its aligned clause carries a
   negation ("not", "no", "never", "without", "cannot", ...). Comparisons such
   as "no more than" are not negations and are removed first; a comparison
   that only one side states with "no" ("more than" against "no more than")
   is flagged separately.
2. **Modality.** The claim says "must" (or "required", "need") where the clause
   only says "may" (or "can", "optional"), or the reverse. Two negated sides
   ("must not", "may not") are both prohibitions and are not compared.
3. **Opposites.** The claim uses one word of a fixed pair ("before"/"after",
   "domestic"/"international") where the clause uses the other.
4. **Quantities.** Every quantity in the claim (a number with its unit, an
   amount with its rate such as "$60 per month", a percentage, a ratio, a clock
   time, a date, a month, a weekday, a frequency such as "every quarter", or an
   identifier such as "SEV1") must appear in the cited passages. The claim is
   flagged if its aligned clause gives a different value in the same slot
   ("30 days" against "90 days"), if only the unit differs ("72 days" against
   "72 hours"), or if the value is nowhere. A number that is not read as a
   quantity ("9am", "15th") must still appear in the cited passages. A number
   copied from the question passes only inside a comparison that holds, for
   example "$5,000 falls in the $1,000 to $10,000 range".

Sentences are split at semicolons and at a comma followed by "and", "but",
"while", "whereas", "so", "yet" or "plus", when both sides keep at least three
content words, so a supported half cannot carry an unsupported one. Every rule
is a lookup or a comparison that a reader can redo by hand; docs/DESIGN.md lists
what the rules still miss.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from citecheck.textnorm import (
    STOPWORDS,
    content_terms,
    normalize_number,
    numbers_in,
    stem_lite,
    tokenize,
)

NEGATIONS: Final[frozenset[str]] = frozenset(
    {"not", "no", "never", "without", "none", "nobody", "nothing", "neither", "nor", "cannot"}
)
OBLIGATION: Final[frozenset[str]] = frozenset(
    {"must", "required", "require", "requires", "mandatory", "shall", "need", "needs"}
)
PERMISSION: Final[frozenset[str]] = frozenset(
    {"may", "can", "allowed", "permitted", "optional", "acceptable"}
)
#: Word pairs whose members contradict each other in policy text. Compared after
#: stem_lite, so "approved" matches "approve".
OPPOSITES: Final[tuple[tuple[str, str], ...]] = (
    ("before", "after"),
    ("earlier", "later"),
    ("earliest", "latest"),
    ("first", "last"),
    ("minimum", "maximum"),
    ("least", "most"),
    ("more", "less"),
    ("more", "fewer"),
    ("above", "below"),
    ("over", "under"),
    ("higher", "lower"),
    ("increase", "decrease"),
    ("full", "half"),
    ("full", "part"),
    ("domestic", "international"),
    ("internal", "external"),
    ("public", "private"),
    ("paid", "unpaid"),
    ("temporary", "permanent"),
    ("open", "closed"),
    ("required", "optional"),
    ("mandatory", "optional"),
    ("mandatory", "voluntary"),
    ("allowed", "prohibited"),
    ("allowed", "forbidden"),
    ("permitted", "prohibited"),
    ("permitted", "forbidden"),
    ("accept", "decline"),
    ("accept", "reject"),
    ("approve", "decline"),
    ("approve", "reject"),
    ("include", "exclude"),
    ("immediately", "gradually"),
    ("own", "another"),
    ("inside", "outside"),
    ("enable", "disable"),
    ("eligible", "ineligible"),
)
_OPPOSITE_STEMS: Final[tuple[tuple[str, str, str, str], ...]] = tuple(
    (stem_lite(a), stem_lite(b), a, b) for a, b in OPPOSITES
)
_MIN_CLAIM_TERMS: Final = 3
_MIN_ALIGNMENT: Final = 2
_CLAIM_BREAK: Final = re.compile(r";\s+|,\s+(?=(?:and|but|while|whereas|so|yet|plus)\s)", re.I)
_CLAUSE_BREAK: Final = re.compile(
    r";\s+|,\s+(?=(?:and|but|while|whereas|so|yet|plus|which|except|unless)\s)", re.I
)
_SENTENCE_BREAK: Final = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\"'(])|\n+")
_COMPARISON_WORDS: Final = "more|less|fewer|later|earlier|longer|shorter|greater"
_COMPARATIVE: Final = re.compile(
    rf"\b(?:no|not)\s{{1,3}}(?:{_COMPARISON_WORDS})\s{{1,3}}than\b",
    re.I,
)
_THAN: Final = re.compile(
    rf"\b(?:(?P<negation>no|not)\s{{1,3}})?(?P<word>{_COMPARISON_WORDS})\s{{1,3}}than\b",
    re.I,
)
_COMPARISON_LOW: Final = re.compile(
    r"\b(?:under|below|within|less than|fewer than|shorter than|up to|at most|no more than)\b"
)
_COMPARISON_HIGH: Final = re.compile(
    r"\b(?:over|above|exceeds?|beyond|more than|greater than|longer than|at least)\b"
)
_COMPARISON_RANGE: Final = re.compile(r"\b(?:range|between|falls? in|from)\b")
_MONTHS: Final = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_WEEKDAYS: Final = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
#: Each opposite word maps to its pairs, so swapping it for its opposite cannot move a
#: claim's alignment while the word still helps to locate the source.
_PAIR_KEYS: Final[dict[str, frozenset[str]]] = {
    stem: frozenset(f"{a}|{b}" for a, b, _, _ in _OPPOSITE_STEMS if stem in (a, b))
    for pair in _OPPOSITE_STEMS
    for stem in pair[:2]
}
_WORD_NUMBERS: Final = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "sixty": 60,
    "ninety": 90,
}
_PERIODS: Final = frozenset({"minute", "hour", "day", "night", "week", "month", "quarter", "year"})
#: Words between a number and its unit that do not change the unit.
_UNIT_MODIFIERS: Final = frozenset(
    {"consecutive", "business", "working", "calendar", "unused", "paid", "sick", "changed"}
    | {"competing", "additional", "extra", "full", "more", "new", "approving"}
)
_NUMBER: Final = r"\d+(?:,\d{3})*(?:\.\d+)?"
# A ratio never starts inside a run of digits: without that anchor, every position of a
# long run followed by a letter retried the whole run, which took quadratic time.
_QUANTITY: Final = re.compile(
    rf"(?P<ratio>(?<!\d)(?P<rl>\d+(?:\.\d+)?)\s{{0,2}}:\s{{0,2}}(?P<rr>\d)(?!\d))"
    rf"|(?P<time>\b(?P<hh>\d{{1,2}}):(?P<mm>\d{{2}})\b)"
    rf"|(?P<money>\$\s?(?P<amount>{_NUMBER}))"
    rf"|(?P<ident>\b(?P<family>[a-z]+)(?P<code>\d+)\b)"
    rf"|(?P<number>\b{_NUMBER}\b)"
    rf"|(?P<word>\b(?:{'|'.join(_WORD_NUMBERS)})\b)"
)
# Whitespace inside these patterns is bounded, so matching stays linear on long runs of
# spaces in untrusted text; a number and its unit are never more than a few spaces apart.
_RATE: Final = re.compile(
    r"\s{0,3}(?:per|a|an|each|every|for each|for every)\s{1,3}(?:calendar\s{1,3})?"
    r"(?P<period>[a-z]{1,20})"
)
_AFTER_NUMBER: Final = re.compile(r"\s{0,3}(?:-\s{0,3})?(?P<rest>[a-z%][a-z-]{0,40})")
_LIST_GAP: Final = re.compile(r"\s{0,3}(?:,|and|or|to|-)\s{0,3}(?:(?:and|or)\s{1,3})?")
_MAX_LIST_GAP: Final = 12
_FREQUENCY: Final = re.compile(
    r"\b(?:every|each|per)\s{1,3}(?:calendar\s{1,3})?"
    r"(minute|hour|day|night|week|month|quarter|year)s?\b"
)


@dataclass(frozen=True, order=True)
class Quantity:
    """A value with what it counts, normalized so equal amounts compare equal.

    Attributes:
        kind: "count", "money", "percent", "ratio", "time", "date", "month",
            "weekday", "frequency" or "id".
        value: The normalized value ("1500", "4.5:1", "10:30", "march 31").
        unit: For counts, the stemmed unit word ("day", "item"), or "" for a
            bare number; for money, "per <period>" or ""; for identifiers, the
            letter prefix ("sev").
        text: The quantity as written, for messages.
    """

    kind: str
    value: str
    unit: str
    text: str

    def same_as(self, other: "Quantity") -> bool:
        """Whether ``other`` states this quantity.

        A bare number in the claim matches any count, amount or percentage with
        the same value, and an amount of money matches when only one side
        states a rate ("$325" and "$325 per night").
        """
        if self.kind == "count" and not self.unit:  # "1500" matches "$1,500" and "1,500 days"
            return self.value == other.value and other.kind in ("count", "money", "percent")
        if (self.kind, self.value) != (other.kind, other.value):
            return False
        return self.unit == other.unit or (self.kind == "money" and not (self.unit and other.unit))

    def same_slot(self, other: "Quantity") -> bool:
        """Whether ``other`` fills the same slot, so a different value contradicts this one.

        Counts need the same unit word and identifiers the same prefix; every
        other kind is one slot per clause (two amounts of money, two dates).
        """
        if self.kind != other.kind:
            return False
        if self.kind in ("count", "id"):
            return bool(self.unit) and self.unit == other.unit
        return True


@dataclass(frozen=True)
class Finding:
    """What the checks found for one claim.

    Attributes:
        text: The claim.
        aligned: The cited clause that shares the most content words with the
            claim, or "" when none shares any.
        problems: One human-readable sentence per failed check.
        missing_numbers: Normalized numbers of quantities found nowhere in the
            cited passages.
        question_numbers: Normalized numbers copied from the question that
            passed the comparison check; they are not claims about the passages.
    """

    text: str
    aligned: str
    problems: tuple[str, ...]
    missing_numbers: tuple[str, ...]
    question_numbers: tuple[str, ...] = ()


def _content_count(text: str) -> int:
    return len(content_terms(text))


def _split(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split at ``pattern``, merging any part with too few content words into its neighbor."""
    parts = [part.strip(" ,;") for part in pattern.split(text) if part and part.strip(" ,;")]
    merged: list[str] = []
    for part in parts:
        if merged and (
            _content_count(part) < _MIN_CLAIM_TERMS
            or _content_count(merged[-1]) < _MIN_CLAIM_TERMS
        ):
            merged[-1] = f"{merged[-1]}, {part}"
        else:
            merged.append(part)
    return merged or [text.strip()]


_ANSWER_PARTICLE: Final = re.compile(r"^\s*(?:yes|no)\s*[,:;]\s*", re.I)


def split_claims(sentence: str) -> list[str]:
    """Split an answer sentence into claims (see the module docstring).

    A leading "Yes," or "No," answers the question rather than stating a claim,
    so it is dropped first.

    Examples:
        >>> split_claims("Submit receipts within 30 days, and late receipts are rejected.")
        ['Submit receipts within 30 days', 'and late receipts are rejected.']
        >>> split_claims("Courses, books, and workshops are covered.")
        ['Courses, books, and workshops are covered.']
    """
    return _split(_ANSWER_PARTICLE.sub("", sentence), _CLAIM_BREAK)


def passage_clauses(passages: Iterable[str]) -> list[tuple[str, str]]:
    """Split cited passages into (clause, its whole sentence) pairs, in reading order."""
    clauses: list[tuple[str, str]] = []
    for passage in passages:
        for sentence in _SENTENCE_BREAK.split(passage):
            if sentence.strip():
                clauses.extend(
                    (part, sentence.strip()) for part in _split(sentence.strip(), _CLAUSE_BREAK)
                )
    return clauses


def _alignment_terms(text: str) -> frozenset[str]:
    """Content words that locate a claim's source.

    Numbers and negations are left out and opposite words become their pairs,
    because those are what the checks compare.
    """
    terms: set[str] = set()
    for term in content_terms(text):
        if term[0].isdigit() or term in NEGATIONS:
            continue
        terms.update(_PAIR_KEYS.get(term, {term}))
    return frozenset(terms)


def align(claim: str, clauses: Sequence[tuple[str, str]]) -> tuple[str, str]:
    """Return the (clause, sentence) pair sharing the most content words with ``claim``.

    The alignment must be unambiguous: the best clause shares at least two
    words and strictly more than any other clause. Otherwise ("", "") is
    returned and no clause comparison runs, because a contradiction can only be
    claimed against the rule the claim restates.

    Examples:
        >>> align("Backups are purged within 30 days.", [("Data is deleted within 30 days.", ""),
        ...       ("Backups are purged within 90 days.", "")])[0]
        'Backups are purged within 90 days.'
    """
    terms = _alignment_terms(claim)
    best, best_score, runner_up = ("", ""), 0, 0
    for clause, sentence in clauses:
        score = len(terms & _alignment_terms(clause))
        if score > best_score:
            best, best_score, runner_up = (clause, sentence), score, best_score
        elif score > runner_up:
            runner_up = score
    return best if best_score >= _MIN_ALIGNMENT and best_score > runner_up else ("", "")


def negated(text: str) -> bool:
    """Whether ``text`` carries a negation, ignoring comparisons such as "no more than".

    Examples:
        >>> negated("Sick days do not carry over."), negated("at most 72 hours")
        (True, False)
        >>> negated("no more than 72 hours")
        False
    """
    return any(token in NEGATIONS for token in tokenize(_COMPARATIVE.sub(" ", text)))


def modality(text: str) -> frozenset[str]:
    """The modal classes in ``text``: "obligation", "permission", both or neither."""
    tokens = set(tokenize(text))
    classes = set()
    if tokens & OBLIGATION:
        classes.add("obligation")
    if tokens & PERMISSION:
        classes.add("permission")
    return frozenset(classes)


def _stems(text: str) -> frozenset[str]:
    return frozenset(stem_lite(token) for token in tokenize(text))


def opposite_conflicts(claim: str, clause: str) -> list[str]:
    """Opposite pairs where the claim uses one word and the clause only the other.

    Examples:
        >>> opposite_conflicts("Book it after you arrive.", "Book it before you arrive.")
        ["the claim says 'after' where the cited text says 'before'"]
    """
    left, right = _stems(claim), _stems(clause)
    found = []
    for first, second, first_word, second_word in _OPPOSITE_STEMS:
        for mine, theirs, mine_word, theirs_word in (
            (first, second, first_word, second_word),
            (second, first, second_word, first_word),
        ):
            if mine in left and theirs in right and mine not in right and theirs not in left:
                found.append(
                    f"the claim says '{mine_word}' where the cited text says '{theirs_word}'"
                )
    return found


def comparisons(text: str) -> frozenset[str]:
    """The comparisons in ``text`` ("more than", "no more than"), with "not" written as "no".

    Examples:
        >>> sorted(comparisons("Not more than 3 days, and later than Friday."))
        ['later than', 'no more than']
    """
    return frozenset(
        ("no " if match.group("negation") else "") + match.group("word").casefold() + " than"
        for match in _THAN.finditer(text)
    )


def comparison_conflicts(claim: str, clause: str) -> list[str]:
    """Comparisons that the claim states with the other polarity: "more than" for "no more than".

    The polarity check removes "no more than" first, because it means "at most" and is
    not a negation. This check sees a "no" added to or dropped from a comparison.

    Examples:
        >>> comparison_conflicts("Notes for more than 3 days.", "Notes for no more than 3 days.")
        ["the claim says 'more than' where the cited text says 'no more than'"]
    """
    left, right = comparisons(claim), comparisons(clause)
    found = []
    for phrase in sorted(left - right):
        other = phrase.removeprefix("no ") if phrase.startswith("no ") else f"no {phrase}"
        if other in right:
            found.append(f"the claim says '{phrase}' where the cited text says '{other}'")
    return found


def _unit_after(folded: str, original: str, end: int) -> tuple[str, str]:
    """The stemmed unit after a number and the unit word as written, or ("", "").

    Modifiers such as "business" and all-capital acronyms such as "PTO" are
    skipped, so "10 business days" and "5 unused PTO days" both count days.
    """
    position = end
    while match := _AFTER_NUMBER.match(folded, position):
        word = match.group("rest").strip("-").split("-")[0]
        written = original[match.start("rest") : match.start("rest") + len(word)]
        if word in ("percent", "%"):
            return "%", written
        if word in STOPWORDS:
            return "", ""  # "60 and 90 days", "3 to 5 peers": the unit comes later
        if word in _UNIT_MODIFIERS or (len(written) > 1 and written.isupper()):
            position = match.end()
            continue
        return stem_lite(word), written
    return "", ""


def _scan(text: str) -> list[tuple[int, int, Quantity]]:
    """First pass: every quantity with its span; counts keep only a directly following unit."""
    folded = text.casefold()
    found: list[tuple[int, int, Quantity]] = []
    for match in _QUANTITY.finditer(folded):
        start, end = match.span()
        written = text[start:end]
        if match.group("ratio"):
            value = f"{normalize_number(match.group('rl'))}:{match.group('rr')}"
            found.append((start, end, Quantity("ratio", value, "", written)))
        elif match.group("time"):
            value = f"{int(match.group('hh'))}:{match.group('mm')}"
            found.append((start, end, Quantity("time", value, "", written)))
        elif match.group("money"):
            unit = ""
            rate = _RATE.match(folded, end)
            if rate and stem_lite(rate.group("period")) in _PERIODS | {"person"}:
                unit, end = f"per {stem_lite(rate.group('period'))}", rate.end()
            amount = normalize_number(match.group("amount"))
            found.append((start, end, Quantity("money", amount, unit, text[start:end])))
        elif match.group("ident"):
            value = f"{match.group('family')}{int(match.group('code'))}"
            found.append((start, end, Quantity("id", value, match.group("family"), written)))
        else:
            raw = match.group("number") or match.group("word")
            value = str(_WORD_NUMBERS[raw]) if raw in _WORD_NUMBERS else normalize_number(raw)
            unit, word = _unit_after(folded, text, end)
            if unit == "%":
                found.append((start, end, Quantity("percent", value, "", f"{written} percent")))
            elif unit or raw not in _WORD_NUMBERS:  # a lone "one" is usually a pronoun
                found.append(
                    (start, end, Quantity("count", value, unit, f"{written} {word}".strip()))
                )
    return found


def quantities(text: str) -> list[Quantity]:
    """Every quantity in ``text`` (see :class:`Quantity`), in order of appearance.

    Numbers in a list share the unit of the last one ("30, 60 and 90 days").

    Examples:
        >>> [q.text for q in quantities("$60 per month, 30, 60 and 90 days, 4.5:1 and SEV1")]
        ['$60 per month', '30 days', '60 days', '90 days', '4.5:1', 'SEV1']
    """
    found = _scan(text)
    folded = text.casefold()
    # Second pass, right to left: a bare count before a list separator takes the next unit.
    for position in range(len(found) - 2, -1, -1):
        start, end, quantity = found[position]
        _, _, following = found[position + 1]
        gap = folded[end : found[position + 1][0]]
        if (
            quantity.kind == following.kind == "count"
            and not quantity.unit
            and len(gap) <= _MAX_LIST_GAP
            and _LIST_GAP.fullmatch(gap)
        ):
            word = following.text.split(" ", 1)[1] if " " in following.text else ""
            found[position] = (
                start,
                end,
                Quantity(
                    "count", quantity.value, following.unit, f"{quantity.text} {word}".strip()
                ),
            )
    result = [quantity for _, _, quantity in found]
    for match in re.finditer(r"\b(" + "|".join(_MONTHS) + r")\b(?:\s{1,3}(\d{1,2})\b)?", folded):
        month, day = match.groups()
        if month == "may" and not day:
            continue  # the modal verb
        written = text[match.start() : match.end()]
        if day:
            result = [
                q
                for q in result
                if not (q.kind == "count" and q.value == str(int(day)) and not q.unit)
            ]
            result.append(Quantity("date", f"{month} {int(day)}", "", written))
        else:
            result.append(Quantity("month", month, "", written))
    for match in re.finditer(r"\b(" + "|".join(_WEEKDAYS) + r")s?\b", folded):
        result.append(Quantity("weekday", match.group(1), "", text[match.start() : match.end()]))
    for match in _FREQUENCY.finditer(folded):
        if "$" in folded[max(0, match.start() - 12) : match.start()]:
            continue  # "$60 per month" is already a rate of the amount
        written = text[match.start() : match.end()]
        result.append(Quantity("frequency", stem_lite(match.group(1)), "", written))
    return result


def _comparison_holds(value: float, others: Sequence[float], claim: str) -> bool:
    """Whether the claim places ``value`` correctly against stated bounds."""
    folded = claim.casefold()
    if len(others) >= 2 and _COMPARISON_RANGE.search(folded):
        return min(others) <= value <= max(others)
    if len(others) != 1:
        return False
    low, high = bool(_COMPARISON_LOW.search(folded)), bool(_COMPARISON_HIGH.search(folded))
    if low == high:
        return False
    return value <= others[0] if low else value >= others[0]


def _numeric(quantity: Quantity) -> float | None:
    try:
        return float(quantity.value)
    except ValueError:
        return None


def _quantity_problems(
    claim: str, aligned: tuple[str, str], clauses: Sequence[tuple[str, str]], question: str
) -> tuple[list[str], list[str], list[str]]:
    mine = quantities(claim)
    near = quantities(aligned[0]) if aligned[0] else []
    same_sentence = quantities(aligned[1]) if aligned[1] else []
    every = [q for clause, _ in clauses for q in quantities(clause)]
    asked = quantities(question) if question else []
    problems: list[str] = []
    missing: list[str] = []
    accepted: list[str] = []
    for quantity in mine:
        if any(quantity.same_as(other) for other in near):
            continue
        if any(quantity.same_as(other) for other in asked):
            bounds = [
                number
                for other in mine
                if other is not quantity
                and other.kind == quantity.kind
                and any(other.same_as(p) for p in every)
                for number in [_numeric(other)]
                if number is not None
            ]
            number = _numeric(quantity)
            if number is not None and _comparison_holds(number, bounds, claim):
                accepted.append(quantity.value)
                continue
        in_sentence = any(quantity.same_as(other) for other in same_sentence)
        unit_swap = [o for o in near if (o.kind, o.value) == (quantity.kind, quantity.value)]
        if unit_swap and not in_sentence:
            problems.append(
                f"the claim says {quantity.text} where the cited text says {unit_swap[0].text}"
            )
            continue
        slot = [other for other in near if quantity.same_slot(other)]
        if slot and not in_sentence:
            problems.append(
                f"the claim says {quantity.text} where the cited text says {slot[0].text}"
            )
            continue
        if any(quantity.same_as(other) for other in every):
            continue
        swapped = [o for o in every if (o.kind, o.value) == (quantity.kind, quantity.value)]
        if swapped:
            problems.append(
                f"the claim says {quantity.text} where the cited text says {swapped[0].text}"
            )
            continue
        problems.append(f"{quantity.text} is not in the cited passages")
        if quantity.kind in ("count", "money", "percent"):
            missing.append(quantity.value)
    # A number the quantity reader cannot read, such as "9am", "15th" or "48h", keeps the
    # lexical checker's number rule: it must appear somewhere in the cited passages.
    read = {number for quantity in mine for number in numbers_in(quantity.text)}
    cited = numbers_in(" ".join(clause for clause, _ in clauses))
    for token in sorted(numbers_in(claim) - read - cited):
        problems.append(f"{token} is not in the cited passages")
        missing.append(token)
    return problems, missing, accepted


def analyze(sentence: str, passages: Sequence[str], question: str = "") -> tuple[Finding, ...]:
    """Split ``sentence`` into claims and run every check against ``passages``.

    Args:
        sentence: One answer sentence.
        passages: The texts of the passages the sentence validly cites.
        question: The user's question, whose numbers may be repeated inside a
            comparison that the passages support.
    """
    clauses = passage_clauses(passages)
    findings = []
    for claim in split_claims(sentence):
        pair = align(claim, clauses)
        aligned, source = pair
        problems: list[str] = []
        if aligned:
            # Polarity and modality must disagree with the aligned clause and with
            # its whole sentence, so a claim that restates several clauses is not
            # judged by one of them alone. Opposite words are compared with the
            # whole sentence.
            if negated(claim) != negated(aligned) and negated(claim) != negated(source):
                which = "the claim" if negated(claim) else "the cited text"
                problems.append(f"only {which} is negated: '{aligned}'")
            claim_modes, clause_modes = modality(claim), modality(aligned)
            both_negated = negated(claim) and negated(aligned)  # two prohibitions agree
            if (
                claim_modes
                and clause_modes
                and not claim_modes & clause_modes
                and not claim_modes & modality(source)
                and not both_negated
            ):
                problems.append(
                    f"the claim states {'/'.join(sorted(claim_modes))} where the cited text "
                    f"states {'/'.join(sorted(clause_modes))}: '{aligned}'"
                )
            problems.extend(opposite_conflicts(claim, source))
            problems.extend(comparison_conflicts(claim, source))
        number_problems, missing, accepted = _quantity_problems(claim, pair, clauses, question)
        problems.extend(number_problems)
        findings.append(Finding(claim, aligned, tuple(problems), tuple(missing), tuple(accepted)))
    return tuple(findings)
