"""Text normalization for search queries and for the support checker.

The verifier's whole method is "which content words of a sentence also appear in
the passages it cites", so the rules in this module *are* the method. They are
kept small enough to check by hand: lowercase the text, split it into word and
number tokens, drop a fixed list of function words, and strip common English
inflections ("stemming-lite"). Predictability matters more than linguistic
accuracy here: a reader should be able to work out why a sentence got its score.

There are two tokenizers on purpose:

* :func:`query_terms` splits text the way SQLite's ``unicode61`` tokenizer does
  (on every non-alphanumeric character), so each term it returns can match an
  indexed token. Stemming is left to SQLite's built-in Porter stemmer.
* :func:`content_terms` is number-aware ("$1,500.00" and "1500" become the same
  token) and applies :func:`stem_lite`. It is only used for verification.
"""

import re
from typing import Final

# A number (digits with optional internal "," or "." groups) or a word made of
# letters, optionally with one internal apostrophe ("employee's", "don't").
_CONTENT_TOKEN_RE: Final = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:'[^\W\d_]+)?")
# What SQLite's unicode61 tokenizer treats as a token: a run of letters/digits.
_QUERY_TOKEN_RE: Final = re.compile(r"[^\W_]+")
# Curly apostrophes (U+2019, U+2018) are mapped to the ASCII apostrophe.
_APOSTROPHES: Final = str.maketrans({"\u2019": "'", "\u2018": "'"})

#: Function words that carry little meaning on their own. Negations ("not",
#: "no", "never", "without") and comparison words ("before", "after", "over",
#: "under", "within", "least") are deliberately *not* stopwords: in policy text
#: they are often the whole point of a rule. Modal verbs ("must", "may") *are*
#: stopwords, so the checker cannot tell "must" from "may"; docs/DESIGN.md lists
#: this among the method's limits.
STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        # articles, determiners and quantifiers
        *("a", "an", "the", "this", "that", "these", "those"),
        *("each", "every", "some", "any", "all", "both", "such"),
        # pronouns
        *("i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself"),
        *("yourselves", "he", "him", "his", "himself", "she", "her", "hers"),
        *("herself", "it", "its", "itself", "we", "us", "our", "ours", "ourselves"),
        *("they", "them", "their", "theirs", "themselves"),
        *("who", "whom", "whose", "which", "what"),
        # auxiliary and modal verbs
        *("am", "is", "are", "was", "were", "be", "been", "being"),
        *("do", "does", "did", "doing", "have", "has", "had", "having"),
        *("will", "would", "shall", "should", "can", "could", "may", "might", "must"),
        # prepositions
        *("of", "in", "on", "at", "to", "for", "from", "by", "with", "about"),
        *("into", "onto", "through", "during", "via", "per", "as"),
        # conjunctions
        *("and", "or", "but", "if", "then", "so", "because", "while", "than"),
        *("though", "although", "whether"),
        # question words and filler adverbs
        *("how", "when", "where", "why", "also", "just", "very", "too", "here"),
        *("there", "etc"),
    }
)

# Suffix rules for stem_lite, tried in order; the first match wins. Each rule is
# (suffix, replacement). Only inflections are handled (plurals, -ed, -ing), not
# derivations such as -al or -ment, so "approval" and "approve" stay different.
_SUFFIX_RULES: Final[tuple[tuple[str, str], ...]] = (
    ("eed", "eed"),  # "agreed", "speed": keep, so "speed" does not become "spe"
    ("ies", "y"),  # policies -> policy
    ("ied", "y"),  # applied -> apply
    ("sses", "ss"),  # addresses -> address
    ("xes", "x"),  # taxes -> tax
    ("ches", "ch"),  # matches -> match
    ("shes", "sh"),  # wishes -> wish
    ("ing", ""),  # booking -> book
    ("ed", ""),  # approved -> approv (and "approve" -> "approv" via the e rule)
    ("s", ""),  # receipts -> receipt (but not "ss", "us", "is": see stem_lite)
)
_MIN_STEM: Final = 3
# Doubled final consonants are undoubled after -ed/-ing ("submitted" -> "submit"),
# except l, s and z, following the Porter stemmer ("called" stays "call").
_UNDOUBLE_EXCEPTIONS: Final = frozenset("lsz")
_VOWELS: Final = frozenset("aeiouy")


def normalize_number(raw: str) -> str:
    """Return a canonical form of a number token so equal amounts compare equal.

    Commas are treated as thousands separators and removed, leading zeros are
    dropped, and a decimal part keeps no trailing zeros. Tokens with more than
    one dot (version numbers such as "2.4.1") are returned unchanged.

    Examples:
        >>> normalize_number("1,500.00")
        '1500'
        >>> normalize_number("007")
        '7'
        >>> normalize_number("2.50")
        '2.5'
    """
    text = raw.replace(",", "")
    if text.count(".") > 1:
        return text
    whole, _, fraction = text.partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def _word_token(raw: str) -> str:
    """Normalize an apostrophe-bearing word: possessives and n't contractions."""
    if raw.endswith("n't"):
        return "not"  # "don't", "can't", "isn't": keep only the negation
    if raw.endswith("'s"):
        return raw[:-2]  # "employee's" -> "employee"
    return raw.replace("'", "")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word and normalized number tokens.

    Stopwords are kept; :func:`content_terms` removes them.

    Examples:
        >>> tokenize("Receipts over $1,500 aren't optional.")
        ['receipts', 'over', '1500', 'not', 'optional']
    """
    folded = text.casefold().translate(_APOSTROPHES)
    tokens: list[str] = []
    for raw in _CONTENT_TOKEN_RE.findall(folded):
        tokens.append(normalize_number(raw) if raw[0].isdigit() else _word_token(raw))
    return tokens


def _undouble(stem: str) -> str:
    """Drop a doubled final consonant ("submitt" -> "submit"), except l, s, z."""
    if (
        len(stem) > _MIN_STEM
        and stem[-1] == stem[-2]
        and stem[-1] not in _VOWELS
        and stem[-1] not in _UNDOUBLE_EXCEPTIONS
    ):
        return stem[:-1]
    return stem


def stem_lite(token: str) -> str:
    """Strip common English inflections so word forms of one word compare equal.

    The rules are intentionally few: plural and verb endings, then a final
    silent "e". Both sides of every comparison (sentence and passage) go through
    the same function, so consistency matters more than producing real words.

    Examples:
        >>> [stem_lite(w) for w in ["policies", "approved", "approves", "approve"]]
        ['policy', 'approv', 'approv', 'approv']
        >>> [stem_lite(w) for w in ["status", "address", "days", "submitted"]]
        ['status', 'address', 'day', 'submit']
    """
    if len(token) <= _MIN_STEM or token[0].isdigit():
        return token
    stem = token
    for suffix, replacement in _SUFFIX_RULES:
        if not stem.endswith(suffix):
            continue
        if suffix == "s" and stem.endswith(("ss", "us", "is")):
            break  # "process", "bonus", "analysis" are not plurals
        candidate = stem[: -len(suffix)] + replacement
        if len(candidate) < _MIN_STEM:
            break  # "lies", "used", "thing": too short to be an inflected form
        stem = candidate
        if suffix in ("ing", "ed"):
            stem = _undouble(stem)
        break
    if stem.endswith("e") and len(stem) > _MIN_STEM + 1:
        stem = stem[:-1]  # "expense" -> "expens", matching "expenses" -> "expens"
    return stem


def content_terms(text: str) -> frozenset[str]:
    """Return the set of normalized content terms in ``text``.

    Content terms are tokens that are not stopwords and not single letters,
    passed through :func:`stem_lite`. Numbers are kept in normalized form.

    Examples:
        >>> sorted(content_terms("Employees must submit receipts within 30 days."))
        ['30', 'day', 'employe', 'receipt', 'submit', 'within']
    """
    terms: set[str] = set()
    for token in tokenize(text):
        if token in STOPWORDS or (len(token) == 1 and not token.isdigit()):
            continue
        terms.add(stem_lite(token))
    return frozenset(terms)


def numbers_in(text: str) -> frozenset[str]:
    """Return the normalized number tokens in ``text``.

    Examples:
        >>> sorted(numbers_in("Up to $1,500 per year, or 2.50 per day."))
        ['1500', '2.5']
    """
    return frozenset(token for token in tokenize(text) if token[0].isdigit())


def query_terms(text: str) -> list[str]:
    """Return search terms for SQLite FTS5, in order of first appearance.

    Text is split exactly where SQLite's ``unicode61`` tokenizer splits it, so
    every term can match an indexed token; stopwords and single letters are
    dropped and duplicates removed. Terms are not stemmed here because the
    index's Porter tokenizer stems query terms itself.

    Examples:
        >>> query_terms("How much is the 401(k) match, and when does it vest?")
        ['much', '401', 'match', 'vest']
    """
    seen: dict[str, None] = {}
    for token in _QUERY_TOKEN_RE.findall(text.casefold()):
        if token in STOPWORDS or (len(token) == 1 and not token.isdigit()):
            continue
        seen.setdefault(token, None)
    return list(seen)
