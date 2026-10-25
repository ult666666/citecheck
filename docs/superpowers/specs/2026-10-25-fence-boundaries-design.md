# Preserve Markdown code fences when assigning citations

The section scanner stores only a fence character. With a four-backtick opener, an internal three-backtick line closes the block early and a subsequent code line such as `## Override Checklist` becomes a false section title and chunk ID. A fence-looking line with trailing text has the same failure.

Apply the boundary rules from [CommonMark 0.31.2 §4.5](https://spec.commonmark.org/0.31.2/#fenced-code-blocks): an opener is at least three matching backticks or tildes, indented zero to three spaces; backtick info strings cannot contain backticks. A closer uses the same character, contains at least the opener's marker count, is indented zero to three spaces, and has only spaces/tabs after the marker. No closer means code continues to document end.

Keep the small existing top-level ATX scanner and raw source line handling. Store the complete opening marker so character and length are retained; capture trailing text separately for opener/closer validation. Code contents remain searchable under their real enclosing section. This is a boundary fix, not a claim of full CommonMark rendering, container parsing or code exclusion.

Alternatives: adding a full Markdown parser would introduce an unnecessary runtime dependency and source-map migration; fixing length alone would still close on trailing text and accept invalid backtick info strings. The selected change handles the boundary rules together without changing the index format or touching provider behavior.

Prove the behavior through public chunking APIs and real SQLite indexing/retrieval, including valid/invalid marker lengths, characters, indentation, trailing text, info strings, line spans, and genuine headings after a valid closer. Preserve existing handbook IDs and offline evaluation output. Work only in the task-owned Citecheck clone and publish a scoped PR for parent review; do not merge.
