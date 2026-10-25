"""Load Markdown and plain-text files and split them into citable chunks.

A chunk is the unit the model cites, so its ID must be stable and readable:
``<relative path>#<heading slug>`` (for example ``people/time-off.md#carryover``),
with a ``:2``, ``:3``... suffix when a long section is split into several chunks.
IDs depend only on the file's relative path and its headings, never on when or
where the index was built, so the gold labels in an evaluation set stay valid
after a rebuild. Renaming a heading or a file does change its IDs.

Chunking rules:

* Markdown files are split at ATX headings (``#`` to ``######``). Headings inside
  fenced code blocks are ignored. YAML front matter (a ``---`` block on the first
  line) is skipped, which is how the bundled corpus labels itself fictional
  without that label being indexed.
* Plain-text files have no headings; the whole file is one section.
* Within a section, paragraphs (separated by blank lines) are packed into chunks
  of at most ``max_chunk_chars`` characters. A paragraph is never split, so a
  single paragraph longer than the limit becomes its own oversized chunk.
* Every chunk records the 1-based, inclusive line span it came from, so a reader
  can open the original file at the cited lines.
"""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Final

from citecheck.errors import ConfigError, CorpusError

MARKDOWN_SUFFIXES: Final = frozenset({".md", ".markdown"})
TEXT_SUFFIXES: Final = frozenset({".txt"})
SUPPORTED_SUFFIXES: Final = MARKDOWN_SUFFIXES | TEXT_SUFFIXES
DEFAULT_MAX_CHUNK_CHARS: Final = 1200
MIN_MAX_CHUNK_CHARS: Final = 100
MAX_MAX_CHUNK_CHARS: Final = 20_000
#: Files larger than this are rejected rather than silently indexed; a handbook
#: page is a few kilobytes, so a 5 MB "document" is almost certainly a mistake.
MAX_FILE_BYTES: Final = 5_000_000

_HEADING_RE: Final = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_CLOSING_HASHES_RE: Final = re.compile(r"(?:^|[ \t]+)#+$")
_FENCE_RE: Final = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


@dataclass(frozen=True)
class Chunk:
    """A citable piece of a document.

    Attributes:
        chunk_id: Stable ID, ``<relative path>#<slug>[:<part>]``.
        doc_path: Path of the source file relative to the corpus root, with
            forward slashes on every platform.
        title: Heading path from the document title down to this section, for
            example ``"Paid Time Off > Carryover"``.
        start_line: First line of the chunk in the source file (1-based).
        end_line: Last line of the chunk in the source file (1-based, inclusive).
        text: The chunk's paragraphs, separated by blank lines. The section
            heading is not repeated here; it is part of ``title``.
    """

    chunk_id: str
    doc_path: str
    title: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class Document:
    """A loaded source file and the chunks cut from it."""

    path: str
    title: str
    sha256: str
    chunks: tuple[Chunk, ...]


@dataclass
class _Paragraph:
    start_line: int
    end_line: int
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class _Section:
    heading_path: tuple[str, ...]
    heading: str | None  # None for text before the first heading
    heading_line: int | None
    paragraphs: list[_Paragraph] = field(default_factory=list)


def validate_max_chunk_chars(max_chunk_chars: int) -> None:
    """Raise :class:`ConfigError` unless the chunk size limit is sensible."""
    if not MIN_MAX_CHUNK_CHARS <= max_chunk_chars <= MAX_MAX_CHUNK_CHARS:
        raise ConfigError(
            f"max_chunk_chars must be between {MIN_MAX_CHUNK_CHARS} and "
            f"{MAX_MAX_CHUNK_CHARS}, got {max_chunk_chars}"
        )


def slugify(heading: str) -> str:
    """Turn a heading into a GitHub-style anchor slug.

    Examples:
        >>> slugify("Receipts & Documentation")
        'receipts-documentation'
        >>> slugify("Who is eligible?")
        'who-is-eligible'
        >>> slugify("***")
        'section'
    """
    text = re.sub(r"[^\w\s-]", "", heading.casefold())
    text = re.sub(r"[\s_]+", "-", text.strip())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "section"


def _clean_heading(raw: str) -> str:
    """Remove optional closing hashes ("## Title ##") and surrounding spaces."""
    return _CLOSING_HASHES_RE.sub("", raw.strip()).strip()


def _front_matter_length(lines: list[str], doc_path: str) -> int:
    """Return how many leading lines belong to YAML front matter (0 if none)."""
    if not lines or lines[0].strip() != "---":
        return 0
    for index in range(1, len(lines)):
        if lines[index].strip() in ("---", "..."):
            return index + 1
    raise CorpusError(f"{doc_path}: front matter starting on line 1 is never closed with '---'")


def _split_sections(lines: list[str], first_line: int, *, markdown: bool) -> list[_Section]:
    """Group lines into sections (by heading) and sections into paragraphs."""
    sections = [_Section(heading_path=(), heading=None, heading_line=None)]
    stack: list[tuple[int, str]] = []
    paragraph: list[tuple[int, str]] = []
    fence: str | None = None

    def flush() -> None:
        if paragraph:
            sections[-1].paragraphs.append(
                _Paragraph(paragraph[0][0], paragraph[-1][0], [text for _, text in paragraph])
            )
            paragraph.clear()

    for offset, line in enumerate(lines):
        line_number = first_line + offset
        if markdown:
            fence_match = _FENCE_RE.match(line)
            if fence is not None:
                paragraph.append((line_number, line.rstrip()))
                if (
                    fence_match
                    and fence_match.group(1)[0] == fence[0]
                    and len(fence_match.group(1)) >= len(fence)
                    and not fence_match.group(2).strip(" \t")
                ):
                    fence = None
                continue
            if fence_match:
                marker, info = fence_match.groups()
                # CommonMark forbids backticks in a backtick fence's info string.
                if marker[0] == "~" or "`" not in info:
                    fence = marker
                    paragraph.append((line_number, line.rstrip()))
                    continue
            heading = _HEADING_RE.match(line)
            if heading:
                flush()
                level = len(heading.group(1))
                text = _clean_heading(heading.group(2) or "")
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
                sections.append(
                    _Section(
                        heading_path=tuple(t for _, t in stack),
                        heading=text,
                        heading_line=line_number,
                    )
                )
                continue
        if not line.strip():
            flush()
            continue
        paragraph.append((line_number, line.rstrip()))
    flush()
    return sections


def _pack(paragraphs: list[_Paragraph], max_chunk_chars: int) -> list[list[_Paragraph]]:
    """Greedily pack consecutive paragraphs into groups of at most max_chunk_chars."""
    groups: list[list[_Paragraph]] = []
    current: list[_Paragraph] = []
    size = 0
    for paragraph in paragraphs:
        length = len(paragraph.text)
        added = length + 2 if current else length  # +2 for the blank-line separator
        if current and size + added > max_chunk_chars:
            groups.append(current)
            current, size, added = [], 0, length
        current.append(paragraph)
        size += added
    if current:
        groups.append(current)
    return groups


def _unique_slug(base: str, used: set[str]) -> str:
    """Return ``base`` or ``base-2``, ``base-3``... whichever is unused yet."""
    slug, counter = base, 2
    while slug in used:
        slug, counter = f"{base}-{counter}", counter + 1
    used.add(slug)
    return slug


def chunk_text(
    text: str,
    doc_path: str,
    *,
    markdown: bool,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> tuple[str, tuple[Chunk, ...]]:
    """Split one document's text into chunks.

    Args:
        text: The full file contents.
        doc_path: The document's path relative to the corpus root (used in IDs).
        markdown: Whether to interpret headings, code fences and front matter.
        max_chunk_chars: Upper bound on chunk length, except for oversized
            single paragraphs.

    Returns:
        The document title (its first heading, else the file name) and its chunks.
    """
    validate_max_chunk_chars(max_chunk_chars)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    skip = _front_matter_length(lines, doc_path) if markdown else 0
    sections = _split_sections(lines[skip:], skip + 1, markdown=markdown)
    first_heading = next((s.heading for s in sections if s.heading), None)
    title = first_heading or PurePath(doc_path).stem
    used: set[str] = set()
    chunks: list[Chunk] = []
    for section in sections:
        if not section.paragraphs:
            continue  # a heading directly followed by a subheading has no text
        base = "top" if section.heading is None else slugify(section.heading)
        slug = _unique_slug(base, used)
        section_title = " > ".join(h for h in section.heading_path if h) or title
        for part, group in enumerate(_pack(section.paragraphs, max_chunk_chars), start=1):
            # The first chunk of a section starts at its heading line, so the
            # cited span shows the reader which section the text came from.
            start_line = group[0].start_line
            if part == 1 and section.heading_line is not None:
                start_line = section.heading_line
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_path}#{slug}" if part == 1 else f"{doc_path}#{slug}:{part}",
                    doc_path=doc_path,
                    title=section_title,
                    start_line=start_line,
                    end_line=group[-1].end_line,
                    text="\n\n".join(paragraph.text for paragraph in group),
                )
            )
    return title, tuple(chunks)


def iter_corpus_files(root: Path) -> Iterator[Path]:
    """Yield supported files under ``root`` in a stable (sorted) order.

    Hidden files and folders (names starting with ".") and symbolic links are
    skipped, so a stray ``.git`` folder or a link pointing outside the corpus is
    never indexed.
    """
    if not root.exists():
        raise CorpusError(f"document folder not found: {root}")
    if not root.is_dir():
        raise CorpusError(f"not a folder: {root}")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def load_document(
    path: Path, root: Path, *, max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
) -> Document:
    """Read one file (UTF-8) and chunk it."""
    doc_path = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raise CorpusError(f"{doc_path}: file is larger than {MAX_FILE_BYTES} bytes")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CorpusError(
            f"{doc_path}: not valid UTF-8 ({exc.reason} at byte {exc.start})"
        ) from exc
    title, chunks = chunk_text(
        text,
        doc_path,
        markdown=path.suffix.lower() in MARKDOWN_SUFFIXES,
        max_chunk_chars=max_chunk_chars,
    )
    return Document(
        path=doc_path, title=title, sha256=hashlib.sha256(raw).hexdigest(), chunks=chunks
    )


def load_corpus(root: Path, *, max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[Document]:
    """Load and chunk every supported file under ``root``.

    Raises:
        CorpusError: If the folder is missing, holds no supported files, holds
            no text at all, or a file cannot be read.
    """
    validate_max_chunk_chars(max_chunk_chars)
    documents = [
        load_document(path, root, max_chunk_chars=max_chunk_chars)
        for path in iter_corpus_files(root)
    ]
    if not documents:
        raise CorpusError(f"no .md, .markdown or .txt files found under {root}")
    if not any(document.chunks for document in documents):
        raise CorpusError(f"the files under {root} contain no text to index")
    return documents
