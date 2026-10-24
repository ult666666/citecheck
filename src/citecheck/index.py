"""A SQLite FTS5 search index over chunks, ranked with BM25.

Why SQLite FTS5: it ships inside Python's standard library (no search server and
no extra dependency), keeps the whole index in one file, and ranks matches with
Okapi BM25 through its built-in ``bm25()`` function. BM25 scores a chunk higher
when it contains more of the query's terms, when those terms are rare across the
corpus (inverse document frequency), and when the chunk is short (length
normalization).

Layout: a plain ``chunks`` table holds each chunk's ID, line span and text, and
an *external-content* FTS5 table (``chunks_fts``) indexes the ``title`` and
``text`` columns of that table without storing the text a second time. Matches
in the title (the heading path) count twice as much as matches in the body.

SQLite's ``bm25()`` returns *lower is better* (negative) values. citecheck flips
the sign, so a :class:`RetrievedChunk` score is *higher is better*. Scores are
only comparable within one index: they depend on corpus statistics.
"""

import os
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from citecheck import __version__
from citecheck.chunking import DEFAULT_MAX_CHUNK_CHARS, Document, load_corpus
from citecheck.errors import IndexFormatError, IndexNotFoundError, QueryError

SCHEMA_VERSION: Final = "1"
#: Porter stemming on top of Unicode-aware tokenization, with accents folded.
FTS_TOKENIZER: Final = "porter unicode61 remove_diacritics 2"
TITLE_WEIGHT: Final = 2.0
TEXT_WEIGHT: Final = 1.0
DEFAULT_INDEX_PATH: Final = Path(".citecheck") / "index.sqlite"

_SCHEMA: Final = f"""
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE documents (
    path TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    chunk_count INTEGER NOT NULL
);
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    doc_path TEXT NOT NULL REFERENCES documents (path),
    title TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    title,
    text,
    content = 'chunks',
    content_rowid = 'id',
    tokenize = '{FTS_TOKENIZER}'
);
"""

_SEARCH_SQL: Final = """
SELECT c.chunk_id, c.doc_path, c.title, c.start_line, c.end_line, c.text,
       bm25(chunks_fts, ?, ?) AS rank
FROM chunks_fts
JOIN chunks AS c ON c.id = chunks_fts.rowid
WHERE chunks_fts MATCH ?
ORDER BY rank, c.chunk_id
LIMIT ?
"""


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a search, with its BM25 score (higher is better)."""

    chunk_id: str
    doc_path: str
    title: str
    start_line: int
    end_line: int
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        data = asdict(self)
        data["score"] = round(self.score, 4)
        return data


@dataclass(frozen=True)
class IndexStats:
    """How much an index holds."""

    documents: int
    chunks: int


def _rows_for(
    documents: Iterable[Document],
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    document_rows: list[tuple[Any, ...]] = []
    chunk_rows: list[tuple[Any, ...]] = []
    for document in documents:
        document_rows.append(
            (document.path, document.title, document.sha256, len(document.chunks))
        )
        chunk_rows.extend(
            (c.chunk_id, c.doc_path, c.title, c.start_line, c.end_line, c.text)
            for c in document.chunks
        )
    return document_rows, chunk_rows


def build_index(
    corpus_dir: Path,
    index_path: Path = DEFAULT_INDEX_PATH,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> IndexStats:
    """Chunk every document under ``corpus_dir`` and write a fresh index file.

    The index is written to a temporary file next to ``index_path`` and moved
    into place only when complete, so an interrupted build never leaves a
    half-written index behind (and readers never see one).

    Raises:
        CorpusError: If the documents cannot be loaded.
        IndexFormatError: If this Python's SQLite build lacks FTS5.
    """
    documents = load_corpus(corpus_dir, max_chunk_chars=max_chunk_chars)
    document_rows, chunk_rows = _rows_for(documents)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_name(index_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "citecheck_version": __version__,
        "fts_tokenizer": FTS_TOKENIZER,
        "max_chunk_chars": str(max_chunk_chars),
        "documents": str(len(document_rows)),
        "chunks": str(len(chunk_rows)),
    }
    try:
        with closing(sqlite3.connect(tmp_path)) as conn:
            try:
                conn.executescript(_SCHEMA)
            except sqlite3.OperationalError as exc:  # pragma: no cover - depends on the build
                raise IndexFormatError(f"SQLite FTS5 is not available: {exc}") from exc
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())
            conn.executemany("INSERT INTO documents VALUES (?, ?, ?, ?)", document_rows)
            conn.executemany(
                "INSERT INTO chunks (chunk_id, doc_path, title, start_line, end_line, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                chunk_rows,
            )
            # Populate the full-text index from the content table in one pass.
            conn.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild')")
            conn.commit()
        os.replace(tmp_path, index_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return IndexStats(documents=len(document_rows), chunks=len(chunk_rows))


class Index:
    """Read-only access to an index file built by :func:`build_index`.

    A new SQLite connection is opened (read-only) for every query. That keeps
    the object safe to share between the threads of a web server, and opening
    a local SQLite file costs well under a millisecond.
    """

    def __init__(self, path: Path = DEFAULT_INDEX_PATH) -> None:
        """Open an existing index and check that its format is supported.

        Raises:
            IndexNotFoundError: If ``path`` does not exist.
            IndexFormatError: If ``path`` is not a citecheck index of this version.
        """
        self.path = path
        if not path.is_file():
            raise IndexNotFoundError(
                f"no index found at {path}; build one first with: citecheck index DIR"
            )
        self._meta = self._read_meta()
        version = self._meta.get("schema_version")
        if version != SCHEMA_VERSION:
            raise IndexFormatError(
                f"{path} uses index format {version!r}, but this citecheck reads format "
                f"{SCHEMA_VERSION!r}; rebuild it with: citecheck index DIR"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)

    def _read_meta(self) -> dict[str, str]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute("SELECT key, value FROM meta").fetchall()
        except sqlite3.DatabaseError as exc:
            raise IndexFormatError(f"{self.path} is not a citecheck index ({exc})") from exc
        return {str(key): str(value) for key, value in rows}

    @property
    def stats(self) -> IndexStats:
        """Document and chunk counts recorded when the index was built."""
        return IndexStats(
            documents=int(self._meta.get("documents", "0")),
            chunks=int(self._meta.get("chunks", "0")),
        )

    def search(self, match_query: str, limit: int) -> list[RetrievedChunk]:
        """Run an FTS5 ``MATCH`` query and return the best ``limit`` chunks.

        Args:
            match_query: An FTS5 query string, normally built by
                :func:`citecheck.retrieval.build_match_query`.
            limit: Maximum number of chunks to return.

        Raises:
            QueryError: If SQLite rejects the query syntax.
        """
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    _SEARCH_SQL, (TITLE_WEIGHT, TEXT_WEIGHT, match_query, limit)
                ).fetchall()
        except sqlite3.OperationalError as exc:
            raise QueryError(f"search query rejected by SQLite: {exc}") from exc
        return [
            RetrievedChunk(
                chunk_id=row[0],
                doc_path=row[1],
                title=row[2],
                start_line=row[3],
                end_line=row[4],
                text=row[5],
                score=-float(row[6]),
            )
            for row in rows
        ]

    def chunk_ids(self) -> frozenset[str]:
        """Return every chunk ID in the index (used to validate gold labels)."""
        with closing(self._connect()) as conn:
            return frozenset(row[0] for row in conn.execute("SELECT chunk_id FROM chunks"))
