"""Turn a question into a safe FTS5 query and apply the abstention threshold.

User questions contain characters that mean something in FTS5's query language
(quotes, ``*``, ``-``, ``:``, parentheses, and the words AND, OR, NOT, NEAR), so a
question is never passed to ``MATCH`` as-is. Instead it is reduced to plain
terms (see :func:`citecheck.textnorm.query_terms`), each term is wrapped in
double quotes, and the terms are joined with ``OR``. Quoting makes every term a
literal; ``OR`` lets a chunk match any of them, and BM25 then ranks chunks that
match more, and rarer, terms higher.

The threshold is what lets citecheck abstain *without a model call*: only chunks
whose score reaches ``min_score`` are offered to the model, and if none do, the
pipeline answers "not enough evidence" on its own.
"""

from dataclasses import dataclass
from typing import Final

from citecheck.errors import QueryError
from citecheck.index import Index, RetrievedChunk
from citecheck.textnorm import query_terms

MAX_QUESTION_CHARS: Final = 2000
#: Very long questions are cut to their first 32 distinct terms; BM25 over an
#: OR of hundreds of terms is slow and ranks poorly anyway.
MAX_QUERY_TERMS: Final = 32


@dataclass(frozen=True)
class RetrievalResult:
    """The outcome of searching for one question.

    Attributes:
        query: The FTS5 query that was run, or None if the question had no
            searchable terms (only stopwords or punctuation).
        candidates: The top-k chunks, best first, before the threshold.
        passages: The candidates whose score is at least ``min_score``; these,
            and only these, are offered to the model.
    """

    query: str | None
    candidates: tuple[RetrievedChunk, ...]
    passages: tuple[RetrievedChunk, ...]

    @property
    def below_threshold(self) -> tuple[RetrievedChunk, ...]:
        """Candidates that were retrieved but scored under the threshold."""
        offered = {passage.chunk_id for passage in self.passages}
        return tuple(c for c in self.candidates if c.chunk_id not in offered)


def validate_question(question: str) -> str:
    """Return the question stripped of surrounding whitespace.

    Raises:
        QueryError: If the question is empty or longer than MAX_QUESTION_CHARS.
    """
    stripped = question.strip()
    if not stripped:
        raise QueryError("the question is empty")
    if len(stripped) > MAX_QUESTION_CHARS:
        raise QueryError(
            f"the question is {len(stripped)} characters long; the limit is {MAX_QUESTION_CHARS}"
        )
    return stripped


def build_match_query(question: str) -> str | None:
    """Build an FTS5 query (``"term" OR "term" ...``), or None if nothing is searchable.

    Examples:
        >>> build_match_query('What does "NOT" mean in a -- query?')
        '"not" OR "mean" OR "query"'
        >>> build_match_query("What is it?") is None
        True
    """
    terms = query_terms(question)[:MAX_QUERY_TERMS]
    if not terms:
        return None
    # query_terms only returns letters and digits, so no term can contain a
    # double quote; quoting turns AND/OR/NOT/NEAR into ordinary words.
    return " OR ".join(f'"{term}"' for term in terms)


def retrieve(index: Index, question: str, *, k: int, min_score: float) -> RetrievalResult:
    """Search the index for ``question`` and split results at ``min_score``."""
    query = build_match_query(validate_question(question))
    if query is None:
        return RetrievalResult(query=None, candidates=(), passages=())
    candidates = tuple(index.search(query, limit=k))
    passages = tuple(chunk for chunk in candidates if chunk.score >= min_score)
    return RetrievalResult(query=query, candidates=candidates, passages=passages)
