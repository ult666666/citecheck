# citecheck design notes

This document explains how citecheck works and why each piece is built the way it is: the
pipeline, the prompt and schema, the verification method and its limits, the evaluation
methodology and the trade-offs. Every number quoted here comes from running the code in this
repository, and most are pinned by tests (named where it matters).

## 1. Goals and non-goals

**Goals**

1. Every answer sentence must point at the evidence for it, in a form a program can check.
2. The check itself must be transparent: a person should be able to recompute any verdict by
   hand from the sentence and the cited passages.
3. The whole system must run offline and deterministically, so that tests, CI and the README
   need no API key and produce the same output every time.
4. The evaluation must say plainly what it measures and what it does not.

**Non-goals**

- Producing the best possible answers. citecheck is about verifiability; a stronger retriever or
  model would plug into the same interfaces.
- Scale. The demo corpus has 25 documents and 103 chunks.
- Production hardening (authentication, rate limiting, multi-tenant indexes).

## 2. The pipeline at a glance

| Stage | Module | Input -> output | Key decision |
|---|---|---|---|
| Chunk | `chunking.py` | files -> chunks with IDs and line spans | split at headings, then pack paragraphs |
| Index | `index.py` | chunks -> one SQLite file | FTS5 external-content table, Porter stemming |
| Retrieve | `retrieval.py` | question -> top-k chunks with BM25 scores | quoted OR query, score threshold |
| Answer | `answer.py`, `prompts.py`, `llm.py` | passages + question -> sentences with citations | strict `submit_answer` tool |
| Check | `verify.py` | sentences + offered passages -> labels | citation validity, word overlap, number rule |
| Judge (optional) | `judge.py` | checked sentences -> entailment verdicts | reported beside, never instead of, the label |
| Account | `costs.py`, `pipeline.py` | responses -> tokens, cost, timings | price by the model that ran each attempt |
| Evaluate | `evaluation.py` | labeled questions -> metrics | fresh temporary index, deterministic report |

`pipeline.ask()` runs the stages in order and returns a `CheckedAnswer` that carries all of the
evidence: offered passages and scores, per-sentence verdicts with matched and missing words,
model calls, usage, cost and timings.

## 3. Ingestion and chunking

**Why headings and paragraphs.** A citation is only useful if a person can follow it, so chunks
follow the document's own structure. Each Markdown file is split at ATX headings (`#` to
`######`); headings inside fenced code blocks are ignored. Within a section, paragraphs
(separated by blank lines) are packed greedily into chunks of at most 1,200 characters. A
paragraph is never split, so a single paragraph longer than the limit becomes one oversized
chunk; that keeps every chunk a contiguous, readable run of the source. Plain-text files have no
headings, so the whole file is one section.

**Fence boundaries.** The scanner retains the opener's marker length and validates the
closing suffix, following [CommonMark §4.5](https://spec.commonmark.org/0.31.2/#fenced-code-blocks).
Shorter markers and fence-looking lines with trailing text stay literal code, so they cannot
create false citation sections. Code contents remain searchable under their enclosing heading.
The scanner handles top-level ATX boundaries; it is not a full Markdown renderer.

**Chunk IDs.** An ID is `<relative path>#<heading slug>`, for example
`people/time-off.md#carryover`. Slugs follow GitHub's anchor style (lowercase, punctuation
removed, spaces to hyphens). A repeated heading in one file gets `-2`, `-3`; a section split into
several chunks gets `:2`, `:3` (slugs never contain a colon, so the two schemes cannot collide).
Text before the first heading uses the slug `top`. IDs depend only on the relative path and the
headings, never on when or where the index was built, which is what lets the evaluation's gold
labels survive a rebuild. Renaming a heading or a file does change its IDs.

**Line spans.** Each chunk records a 1-based, inclusive line span in the original file. The first
chunk of a section starts at its heading line, so the cited span shows which section the text
came from. Front matter lines still count, so spans match what an editor shows.

**Front matter.** A YAML block at the top of a Markdown file (`---` ... `---`) is skipped. The demo
corpus uses it to label every document `fictional: true` without putting that label into the
index, where the same words in every document would add noise to retrieval
(`tests/test_bundled_data.py` checks that searching for the label finds nothing).

**Defensive loading.** Files are read as strict UTF-8 (a bad byte names the file), files over
5 MB are rejected rather than silently indexed, hidden files and folders are skipped (no
accidental `.git` indexing), and symbolic links are skipped (no indexing outside the folder).

## 4. Index and retrieval

**Why SQLite FTS5.** It ships with Python's standard library, needs no server, keeps the index in
one file, and ranks with BM25 through the built-in `bm25()` function. The schema is a plain
`chunks` table (ID, path, heading path, line span, text) plus an *external-content* FTS5 table
that indexes the `title` (heading path) and `text` columns without storing the text twice. The
index is built in a temporary file and moved into place with `os.replace`, so readers never see a
half-built index and a failed build leaves the old one intact.

**Tokenizer.** `porter unicode61 remove_diacritics 2`: Unicode-aware splitting on
non-alphanumeric characters, accent folding, and Porter stemming, so "reimbursements" matches
"reimbursed".

**BM25.** For a query Q and chunk D, FTS5 computes

    score(D, Q) = sum over terms t in Q of
                  IDF(t) * f(t, D) * (k1 + 1) / (f(t, D) + k1 * (1 - b + b * |D| / avgdl))

with k1 = 1.2 and b = 0.75, where f(t, D) is the term's frequency in the chunk, |D| the chunk
length and avgdl the average length. IDF(t) = ln((N - n + 0.5) / (n + 0.5)), where N is the number
of chunks and n the number containing t, so rare terms count more. In words: more query terms,
rarer terms and shorter chunks all raise the score. citecheck weights title matches 2.0 and body
matches 1.0. SQLite returns lower-is-better values; citecheck flips the sign so higher is better.
Scores depend on corpus statistics and are only comparable within one index.

**Query building.** A question is never passed to `MATCH` as-is, because FTS5 gives meaning to
quotes, `*`, `-`, `:`, parentheses and the words AND, OR, NOT and NEAR. The question is split the
way the `unicode61` tokenizer splits text, stopwords and single letters are dropped, every term is
wrapped in double quotes and the terms are joined with OR:

    receipts AND hotels NEAR(x) "quoted" -minus col:val *
    -> "receipts" OR "hotels" OR "near" OR "quoted" OR "minus" OR "col" OR "val"

Quoting makes every term literal, and OR lets BM25 rank chunks that match more (and rarer) terms
higher. Questions are capped at 2,000 characters and 32 distinct terms. A Hypothesis property test
feeds arbitrary Unicode strings through this path and checks that SQLite accepts every query.

**The abstention threshold.** Only chunks scoring at least `min_score` are offered to the model;
if none do, citecheck abstains without a model call. The default, 3.0, was chosen by looking at
score distributions on the demo corpus (pinned in
`tests/test_bundled_data.py::test_the_numbers_behind_the_default_threshold`):

- a chunk that matches only the company name, which appears in many documents, scores about 2
  (the best such match for "What is Larkmoor's stock ticker symbol?" is 2.07);
- a chunk that matches one rare word scores about 4 ("Pittsburgh" alone: 4.17);
- every one of the 31 gold chunks for the 28 answerable questions is in the top 5, and the
  weakest scores 5.49.

So 3.0 removes name-only matches while keeping every gold chunk. This is circular: the threshold
was tuned on the same questions it is evaluated on, which is one reason the evaluation numbers
are not a generalization claim. On your own documents, look at `citecheck search` output and
recalibrate.

**Quirks worth knowing.** Porter stemming conflates some unrelated words: "international" and
"internal" both become "intern", so a question about international travel retrieves the "Focus
Thursdays" chunk, which mentions internal meetings (pinned by a test). And FTS5 clamps a
non-positive IDF to a tiny positive number, so in a corpus with very few chunks every match
scores almost zero; the unit tests use corpora with several chunks for that reason.

**Concurrency.** `Index` opens a fresh read-only SQLite connection per query. That keeps it safe
to share across the threads of the web server; opening a local SQLite file is cheap.

## 5. Prompt and schema design

### Why a tool call

The answer has to come back as data (sentences, each with citation IDs) that code can check.
Three options were considered:

1. Free text with inline markers such as `[2]`: easy to read, fragile to parse, and nothing
   enforces that every sentence has a marker.
2. A tool call with a JSON schema: the model "calls" `submit_answer` and its arguments are the
   answer. With `strict: true`, the API guarantees the arguments match the schema.
3. Structured outputs (`output_config.format` with a JSON schema): the same schema guarantee for
   the reply itself, and no possibility of the model skipping the tool.

citecheck uses option 2, as the project brief asked. Option 3 would work equally well and would
remove the retry described below; it is the first alternative to try.

### Forced tool choice is not available

The usual way to guarantee a tool call is `tool_choice: {"type": "tool", "name": ...}`. Claude
Opus 5.5 and Claude Sonnet 5.5 reject forced tool choice with HTTP 400. citecheck therefore:

- sets `tool_choice: {"type": "auto", "disable_parallel_tool_use": true}` (at most one call);
- names the tool in the system prompt and in the tool description;
- keeps `strict: true`, so a call that does happen has schema-valid arguments;
- retries once if a reply has no `submit_answer` call. The retry is a fresh, self-contained
  request with a one-line reminder appended, not a continuation of the conversation, so no
  earlier assistant output (including thinking blocks) has to be replayed. A second miss raises
  `ModelResponseError`.

citecheck validates the arguments itself as well (`answer.parse_answer_input`), because fixtures,
older models and future schema changes should fail loudly at the boundary.

### The schema

```json
{
  "sentences": [{"text": "string", "citations": ["chunk id", "..."]}],
  "insufficient_evidence": true
}
```

Both objects set `additionalProperties: false` and list every property as required, as strict
mode demands; the schema avoids keywords strict mode does not support, such as `minItems` (a test
walks both tool schemas to check this). How the two fields combine:

| `insufficient_evidence` | `sentences` | Status | Meaning |
|---|---|---|---|
| false | non-empty | answered | normal answer |
| true | non-empty | partial | answered what the passages support, flagged the rest |
| true | empty | abstained | the passages do not contain the answer |
| false | empty | abstained | protocol slip; treated as no answer |

**A considered alternative:** putting the retrieved chunk IDs into the schema as an `enum` would
make invalid citations impossible to emit. It was rejected because the tool definition would then
change with every question. Tools are the first part of the prompt, so the prompt cache would
never hit, and the API compiles each new strict schema before use, adding latency. The checker
already rejects invalid citations, and keeping them visible is useful for measuring how often a
model gets IDs wrong.

### The prompt

The system prompt is a constant. It tells the model to answer only by calling the tool, to write
one claim per sentence, to cite only passages that directly state the claim, to copy numbers and
names exactly, to use `insufficient_evidence` instead of guessing, and to treat passage text as
data rather than instructions. It also says that each sentence will be checked automatically and
that answers close to the passages' wording are easiest to verify. That last line trades some
fluency for verifiability, on purpose.

The user message puts the passages first and the question last, the layout Anthropic recommends
for document question answering:

```text
<passages>
<passage id="workplace/expenses.md#submitting-expenses" title="Expense Reimbursement &gt; Submitting expenses">
Submit business expenses in the expense tool within 30 days of the purchase date. ...
</passage>
...
</passages>

<question>What is the deadline for submitting an expense?</question>
```

**Prompt-injection defenses.** Passage text has `<` and `>` escaped, so a document cannot close
the `<passage>` element or forge a new one (tested); attributes are fully HTML-escaped; and the
system prompt says instructions inside passages are not to be followed. Only retrieved passages
are ever sent, and the checker independently verifies whatever comes back, so an injected
instruction that changes the answer still has to survive the citation and support checks.

### Request parameters

`ModelRequest.to_api_params()` is the single source of truth for what is sent; `ask --dry-run`
prints it, and a test checks the live adapter sends exactly that.

- **Model:** `claude-opus-5-5` by default, configurable.
- **Effort:** `output_config.effort = "medium"`, sent explicitly because defaults differ between
  models; `--effort none` omits it for models that do not accept it.
- **No `temperature`, `top_p`, `top_k` or `thinking`:** current models reject sampling
  parameters, and the default model thinks adaptively when `thinking` is omitted.
- **`max_tokens = 16000`:** thinking tokens count toward the limit, and a reply cut off in the
  middle of a tool call is an error (citecheck never uses a truncated tool call), while unused
  headroom costs nothing.
- **Refusal fallback:** `fallbacks: "default"` with the `server-side-fallback-2026-07-01` beta
  header. If the requested model declines for policy reasons, the API can serve the request with
  another model and reports each attempt in `usage.iterations`. citecheck surfaces
  `fallback_used`, reports which model served the answer, and prices every attempt by its own
  model. Because the parameter lives in the SDK's beta namespace, live calls go through
  `client.beta.messages.create`; with `--no-fallback` no beta header is sent. A refusal that is
  not rescued becomes an abstention.

### Prompt caching

Prompt caching is a prefix match over `tools -> system -> messages`. citecheck puts one cache
breakpoint (`cache_control: {"type": "ephemeral"}`, the 5-minute cache) on the system prompt, so
the tool definition and system prompt, which are identical for every question, can be cached,
while the passages and question come after the breakpoint. Two invariants keep that prefix
byte-identical, and both are tested: the system prompts contain no dates, times or IDs, and two
requests for different questions render identical `tools` and `system` parameters.

Anthropic documents a minimum cacheable prefix length (listed as 512 tokens for the current
models when this was written).
citecheck's prefix is 2,785 characters, but its token count was not measured, because measuring
needs an API call (the token-counting endpoint); a character-based guess is not a token count. On
a live run, a non-zero `cache-read` figure from the second question on confirms the cache is
working.

## 6. Verification method

citecheck has two checkers. The lexical checker, described first, answers one question per
sentence: *do the passages this sentence cites contain its words and numbers?* The claim-level
checker, the default, keeps steps 1 to 3 and replaces the number rule with comparisons against
the clause a claim restates ([Claim-level checking](#claim-level-checking)). `--checker lexical`
selects the original.

**Step 1: citation validity.** Citations are deduplicated. A citation is valid only if it names a
passage that was offered to the model for this question; anything else (an invented ID, a typo,
a real chunk that was not in this question's top k) is rejected and reported. A sentence with no
valid citation is unsupported.

**Step 2: content terms.** The sentence and its validly cited passages (heading path plus text)
are reduced to sets of content terms (`textnorm.content_terms`):

- Lowercase, then split into words and numbers. Numbers are normalized: "$1,500.00" -> `1500`,
  "2.50" -> `2.5`, "007" -> `7`. Possessives lose "'s"; "n't" contractions become `not`.
- Drop stopwords: articles, pronouns, auxiliary and modal verbs, common prepositions and
  conjunctions, question words. Negations ("not", "no", "never", "without") and comparison words
  ("before", "after", "over", "under", "within") are kept, because in policy text they are often
  the whole rule. Modal verbs ("must", "may") are dropped.
- Drop single letters (such as the "k" in "401(k)").
- Apply "stemming-lite": strip one inflection, then a final silent "e".

| Rule | Example |
|---|---|
| `-ies` -> `-y`, `-ied` -> `-y` | policies -> policy, applied -> apply |
| `-sses`, `-xes`, `-ches`, `-shes` -> drop `es` | addresses -> address, taxes -> tax |
| `-ing`, `-ed` (then undouble a final consonant except l, s, z) | booking -> book, submitted -> submit |
| `-s`, except after `ss`, `us`, `is` | receipts -> receipt; status, bonus, analysis unchanged |
| `-eed` is left alone | speed stays speed |
| final `e` dropped when the stem stays longer than 3 letters | expense -> expens, approve -> approv |
| no change if the result would be shorter than 3 letters | lies, used unchanged |

Both sides of every comparison go through the same function, so consistency matters more than
producing real words: approve, approved and approves all become `approv`. Only inflections are
handled, so "approval" and "approve" stay different. The rules are deliberately few, so that a
person can predict every match.

**Step 3: support score.**

    support = |sentence terms found in the cited passages| / |sentence terms|

**Step 4: number rule.** Any number in the sentence that appears in none of its cited passages
makes the sentence unsupported, whatever its support score. A wrong figure is the most damaging
error in a policy answer, and one wrong digit barely moves an overlap score.

**Step 5: label.** support >= 0.80 is **supported**, support >= 0.50 is **weakly supported**,
anything lower is **unsupported**. A sentence with no content words at all ("It is what it is.")
is unsupported: there is nothing to check. Both thresholds are configurable.

**Worked example.** Fixture q20, sentence 2: "Change your password right away and notify your
manager." cites `security/phishing.md#if-you-clicked`, whose text says to "change your password
right away" and report in the security help channel.

- sentence terms: away, chang, manager, notify, password, right (6)
- found in the passage: away, chang, password, right (4)
- support = 4 / 6 = 0.67, no numbers -> **weakly supported**, with "manager, notify" reported
  as not found. The human label for this sentence is unsupported (the passage never mentions a
  manager), so this counts as a catch.

Fixture q13, sentence 2: "Expenses submitted after 60 days are rejected automatically." The
passage says 30 days. "60" is in no cited passage, so the sentence is **unsupported** by the
number rule.

**What the method catches and misses** (all from the bundled fixtures):

| Case | Fixture | Checker | Correct? |
|---|---|---|---|
| invented rule with a number | q13 | unsupported (number) | yes |
| wrong amount | q16 | unsupported (number) | yes |
| opinion with no citation | q17 | unsupported (no citation) | yes |
| only citation is a typo | q19 | unsupported (citation rejected) | yes |
| embellishment ("notify your manager") | q20 | weakly supported | yes (flagged) |
| negation flip with the same words | q03 | supported | **no** (miss) |
| domestic per diem stated as international | u02 | supported (5 of 6 words) | **no** (miss) |
| correct paraphrase in other words | q15 | unsupported (4 of 9 words) | **no** (false flag) |
| correct sentence repeating the question's $5,000 | q17 | unsupported (number) | **no** (false flag) |

With the claim-level checker, q03 and u02 are caught, q17's $5,000 passes its range comparison
(the sentence is still weakly supported because its first claim shares 3 of 5 words), and q15
remains a false flag.

### Claim-level checking

The lexical checker's misses share a cause: it compares a sentence with the union of every word
in its cited passages, so it cannot tell which rule the sentence restates. The claim-level
checker (`citecheck.claims`) adds that step and then asks sharper questions:

1. **Split into claims.** A sentence is split at semicolons and at a comma followed by "and",
   "but", "while", "whereas", "so", "yet" or "plus", when both sides keep at least three content
   words. A leading "Yes," or "No," is dropped, because it answers the question rather than
   stating a claim. Each claim gets its own label and the sentence gets the worst one, so a
   supported half cannot carry an unsupported half.
2. **Align.** Cited passages are split into sentences and those into clauses (the same comma
   rule, plus "which", "except" and "unless"). A claim's aligned clause is the clause sharing the
   most content words with it, where numbers and negations do not count and an opposite word
   counts as its pair, so changing those words cannot move the alignment. The best clause must
   share at least two words and strictly more than any other clause; on a tie no clause check
   runs, because a contradiction can only be claimed against the rule the claim restates.
3. **Polarity.** Flagged when exactly one side is negated ("not", "no", "never", "without",
   "cannot", ...), and the claim also disagrees with the clause's whole sentence. "No more
   than" and similar comparisons mean "at most", not a negation, so they are removed first.
   A comparison that one side states with "no" or "not" and the other without ("more than 3
   days" against "no more than 3 days") is flagged by its own check against the whole
   sentence.
4. **Modality.** Flagged when the claim only obliges ("must", "required", "need") where the
   clause and its sentence only permit ("may", "can", "optional"), or the reverse. Two negated
   sides are both prohibitions and agree.
5. **Opposites.** Flagged when the claim uses one word of 37 fixed pairs ("before"/"after",
   "minimum"/"maximum", "domestic"/"international", "full"/"part", "accept"/"decline", ...)
   and the aligned sentence uses only the other.
6. **Quantities.** Numbers are read with what they count: "30, 60 and 90 days" gives three day
   counts, "$60 per month" a monthly amount, "4.5:1" a ratio, "10:30" a clock time, "March 31"
   a date, "Fridays" a weekday, "every quarter" a frequency, "SEV1" an identifier; units skip
   modifiers such as "business" and acronyms such as "PTO". Every claim quantity must appear in
   the cited passages. It is flagged if the aligned clause has the same value with another unit
   ("72 days" against "72 hours"), a different value in the same slot ("30 days" against
   "90 days", "SEV1" against "SEV2", "$1,500" against "$75 per day"), or if it appears nowhere. A
   value present anywhere in the aligned clause's sentence is never a conflict. A number copied
   from the question passes only inside a comparison that holds: "$5,000 falls in the $1,000 to
   $10,000 range" or "a $700 purchase is under $1,000"; "a $12,000 purchase needs the department
   head" is still flagged. A number that is not read as a quantity because letters follow it
   ("9am", "15th", "48h") keeps the lexical number rule: it must appear somewhere in the cited
   passages.
7. **Label.** Any problem makes a claim unsupported, and the reason names it. Otherwise the
   claim's own word overlap decides its label, with the same thresholds as before.

**Labelled claim cases.** The 49 fixture sentences include only 7 unsupported ones, and the
lexical checker's four documented errors are among them, so they could not show whether a new
checker generalizes. `eval/claims.json` adds 140 cases. Each is a sentence written against one
or two real handbook chunks and labelled supported or unsupported by reading only those chunks.
Their categories, with counts: restatement 25, paraphrase 17, unit 13, swapped attribute 12,
negation 12, antonym 12, compound 11, number 9, contradiction 8, date 5, kept negation 5,
entity 3, question number 3, modality 2, inference 2 and multi-passage 1. Most categories hold
only one label; compound, question-number and multi-passage cases hold both. `claim_split` assigns a case to dev when the SHA-256 of its ID
is 0 or 1 modulo 5, which gave 63 dev and 77 test cases. The loader rejects a case whose split
does not match the rule, and a test pins the file's hash. The file was committed before any
checker change.

**Protocol and results.** Rules were adjusted only after reading dev results. Once frozen, the
checker was run once on test: 43 of 48 unsupported sentences flagged and 2 of 29 supported
sentences flagged. Two later changes were motivated by new corpus-wide tests, not by test
errors: every sentence of the handbook copied verbatim must pass, and swapping a word for its
opposite in an aligned sentence must fail. After them the dev and fixture results were
unchanged, and on test one case moved, c035, which is now flagged for a spurious reason (the verb
"owns" matched the pair "own"/"another"). Final figures (`uv run citecheck eval-claims`):

| Split | Checker | Unsupported flagged | Supported flagged | Accuracy |
|---|---|---|---|---|
| test (77) | lexical | 20 of 48 | 1 of 29 | 0.623 |
| test (77) | claim-level | 44 of 48 | 2 of 29 | 0.922 |
| dev (63) | lexical | 6 of 34 | 8 of 29 | 0.429 |
| dev (63) | claim-level | 28 of 34 | 7 of 29 | 0.794 |

On the test cases the claim-level checker alone is right 24 times and the lexical checker
alone once. Under no difference each such case would favor either checker with probability
1/2, so the exact two-sided McNemar p-value is 2 × (1 + 25) / 2^25 = 1.5e-06. By category on
test, the claim-level checker judges correctly 7 of 7 antonym cases (lexical 1), 5 of 5 negation
cases (1), 6 of 6 unit cases (2) and 5 of 9 swapped attributes (0). It misses four unsupported
test sentences, all swapped attributes: two values moved within one source sentence ("75
percent of the premium for employees", where 75 percent is the dependents' share, and "$500 per
month toward home internet", where $500 is the one-time stipend), a budget named for another
("paid from the learning budget" for "the team's travel budget") and a duty given to someone
else ("the promotion committee writes the promotion case"). Its two false flags on test are a
paraphrase with too few shared words and a compound sentence whose second claim ("it can pay
for books and workshops") shares 2 of 3 words. On dev it still misses a rule restated without
its restriction, a claim that merges two sentences and contradicts the one it does not align
with, two entity swaps, a license class given another class's rule and a value swapped within
one sentence. Its dev false flags are four paraphrases, two inferences ("60 days before" for
"less than 90 days") and one question-number sentence with low word overlap.

**What the numbers mean.** They show that the rules catch the error types they were designed
for, on sentences they were not tuned on, without raising false flags much. They are not an
accuracy estimate for real model output: the same author wrote the cases, labelled them and
wrote the rules, and had read every case while writing them, so the test figures are
optimistic. Real answers will contain errors outside these categories.

**Why a lexical checker at all.** It is deterministic, free, fast, needs no model or network,
and every verdict comes with the exact words that matched and the ones that did not, so it can be
audited and unit-tested. Its errors are predictable: it cannot see negation scope, word order,
modality or paraphrase, and citing more passages makes passing easier (support is measured
against their union). Lexical retrieval and lexical verification also share blind spots: a
synonym that BM25 misses is one the checker would miss too. That is why an entailment-style check
is the next step (section 11).

**The optional judge.** `--judge` sends the validly cited sentences and their passages to a model
with a `submit_verdicts` tool and asks, per sentence, for supported, partially supported or not
supported, with a one-sentence reason. Its verdict is shown beside the lexical label, never
instead of it: a model grading a model is useful evidence, not ground truth, and it costs one more
call. In the bundled demo the judge responses are hand-written fixtures too (for q03 and q15),
written to show the two methods' complementary failure modes.

## 7. Evaluation methodology

**The corpus.** `examples/handbook` holds 25 original documents (89 to 206 words each) for a
fictional company. They were written to be specific (numbers, deadlines, exceptions, approval
rules), with a few deliberate gaps (part-time PTO, an international per diem, a dental
deductible) so that some unanswerable questions sit close to real content.

**The questions.** `eval/questions.json` has 36 questions: 28 answerable, each with its gold chunk
IDs (31 gold chunks in total; three questions need two chunks, one across two documents), and 8
unanswerable: out of domain, in-domain gaps, and one with no searchable words. Each question has
a `notes` field saying what it tests. The evaluation rebuilds a fresh temporary index and refuses
to run if a gold ID does not exist, so a typo in the labels cannot silently lower recall.

**The fixtures.** Every question that gets past retrieval has a hand-written synthetic
`submit_answer` input. Most are faithful; some are wrong on purpose (listed in `eval/README.md`).
Each sentence also carries a human label (supported or unsupported). Fixtures exist so the whole
pipeline, metrics included, runs offline and deterministically; they test the mechanics, not a
model.

**Metrics.** With G(q) the gold chunks of question q, O(q) the chunks offered to the model and
C(q) every chunk ID cited in the answer (valid or rejected):

- **recall@k** = mean over answerable q of |G(q) ∩ O(q)| / |G(q)|; **hit@k** = share of answerable
  questions with G(q) ∩ O(q) non-empty. Retrieval quality, independent of the model.
- **citation precision** = TP / (TP + FP) and **citation recall** = TP / (TP + FN), pooled over
  questions, with TP = |C(q) ∩ G(q)|, FP = |C(q) − G(q)| (every citation in an answer to an
  unanswerable question is a false positive) and FN = |G(q) − C(q)| over answerable questions (an
  abstention on an answerable question loses all its gold chunks). Pooled here means summed
  counts, not an average of per-question ratios. Bundled result: 29 / 33 = 0.879 and
  29 / 31 = 0.935.
- **rejected citations**: citations to chunks that were not offered (2).
- **unsupported-sentence rate** = unsupported sentences / all answer sentences = 6 / 49 = 0.122.
- **abstention accuracy** = share of questions where the pipeline abstained exactly when the
  question is unanswerable = 34 / 36 = 0.944. A partial answer counts as an answer.
- **checker vs. human** (fixture mode only): a sentence is *flagged* if its label is anything but
  supported. The report counts flagged sentences among those a human labeled unsupported
  (7 of 7 with the claim-level checker, 5 of 7 with `--checker lexical`) and among those labeled
  supported (2 of 42 with either).

These numbers are pinned in `tests/test_evaluation.py`, and `tests/test_readme.py` checks that the
README shows exactly what `citecheck eval` prints. The report contains no timings, so it is
byte-for-byte reproducible.

**What they mean, and what they do not.** They show that the harness runs end to end and how the
checker behaves on known errors. They say nothing about any real model: the "model" is the
fixture set, the same author wrote the questions, labels and fixtures, the threshold was tuned on
the same questions, and 36 questions cannot support statistical claims. To measure a real model,
run `citecheck eval --live` (real API calls, one per question that clears retrieval, plus any
retries). Live mode reports everything except checker-vs-human agreement, since the human labels
describe the fixture sentences.

## 8. Cost and latency accounting

**Tokens.** Every response's `usage` is recorded: `input_tokens` (uncached input only),
`output_tokens` (thinking included), `cache_creation_input_tokens` and `cache_read_input_tokens`;
the full prompt size is the sum of the three input fields. When a refusal fallback runs, the API
reports each attempt in `usage.iterations` (a declined attempt plus the attempt that served the
answer), and citecheck records each with the model that ran it.

**Cost.** citecheck ships no prices: they change and depend on contract and platform. You pass a
TOML table with per-million-token prices for input, output, cache writes and cache reads per
model (`examples/prices.example.toml` shows the format with placeholder numbers). Amounts are
`decimal.Decimal`, so tiny per-token prices add up exactly. If any attempt was served by a model
missing from the table, no total is reported at all, rather than a misleadingly low one.

**Fixture mode.** The fixture client estimates tokens at about four characters per token and
every display says so. It is not a tokenizer; the only accurate count comes from the API
(including its token-counting endpoint), which offline mode cannot call. A cost computed from
these estimates demonstrates the arithmetic only.

**Latency.** `time.perf_counter()` timings are recorded per stage (retrieval, model, verification,
total) and shown by `ask` and the API. In fixture mode they mostly measure Python overhead. They
are left out of the evaluation report to keep it reproducible.

## 9. Interfaces and errors

The CLI defaults to fixture replay, so nothing calls the API unless `--live` is passed, and
`ask --dry-run` prints the exact request without creating any client. The FastAPI app (an optional
`[api]` extra) serves `POST /api/ask`, `POST /api/search`, `GET /healthz` and the HTML page; it
opens the index at startup, so a missing index fails immediately rather than on the first
request. The page uses a labeled form, an `aria-live` status region and text labels as well as
color; every text color has a contrast ratio of at least 6.6:1 against the background in both
light and dark mode; and server text is inserted with `textContent` only.

Every deliberate failure raises a subclass of `CiteCheckError`, and the CLI and API map them in
one place:

| Error | Raised when | CLI | HTTP |
|---|---|---|---|
| `QueryError` | empty or overlong question | exit 1 | 422 |
| `ConfigError` | invalid settings (k = 0, weak > supported, ...) | exit 1 | 422 |
| `CorpusError` | missing folder, bad UTF-8, oversized file | exit 1 | 500 |
| `IndexNotFoundError`, `IndexFormatError` | no index, or wrong format version | exit 1 | 503 |
| `FixtureNotFoundError` | fixture mode has no response for a question | exit 1 | 404 |
| `ModelResponseError` | no tool call after a retry, bad schema, `max_tokens` cut-off | exit 1 | 502 |
| `ModelAPIError` | API failure; `retryable` tells 429/5xx/network from 400/401/404 | exit 1 | 503 / 502 |
| `PricingError`, `EvalDataError`, `FixtureError` | malformed price table, question set, fixture | exit 1 | 500 |

SDK exceptions are caught from most to least specific (authentication, not found, rate limit,
bad request, other status, connection, other SDK errors), because a single broad `except` would
lose the difference between "try again later" and "this will never work". The SDK already retries
rate limits, overloads and network errors with backoff before citecheck sees them.

## 10. Testing strategy

- Over 300 tests, mostly fast unit tests with small synthetic corpora; integration tests run the
  bundled corpus through the CLI, the API and the evaluation.
- **No network, ever.** An autouse fixture replaces `anthropic.Anthropic` with a constructor that
  fails the test. The live adapter is tested with real SDK response types (`BetaMessage` built
  from API-shaped JSON, including a fallback-served response) and a stand-in client, and each
  branch of the SDK error mapping is covered by a parametrized test.
- **Property tests** (Hypothesis): any user string yields a query SQLite accepts; query terms are
  plain alphanumeric non-stopwords; content terms of joined passages equal the union of their
  terms, which is why joining cited passages cannot create or destroy a match.
- **Pinned facts:** the evaluation numbers, the threshold rationale, the Porter quirk, the
  README's evaluation block and quick-start output, and the integrity of the bundled data
  (fictional labels, fixture coverage, deliberately invalid citations listed explicitly).
- **Metamorphic tests over the whole handbook:** every sentence copied verbatim passes the
  claim-level checker; inserting "not" into any aligned sentence, changing any counted number
  to a value the chunk does not contain, or swapping a word for its opposite is flagged. A
  Hypothesis test feeds arbitrary text through the claim analysis. The held-out and dev figures,
  the claim file's hash and its split rule are pinned, and the exact McNemar function is checked
  against hand values and a SciPy cross-check.
- mypy in strict mode and ruff (including security lints) on source and tests; CI on Python 3.12
  and 3.13.

## 11. Trade-offs and alternatives

**BM25 instead of embeddings.** BM25 needs no model, no network and no vector store, ranks
transparently and is strong when questions use the documents' vocabulary, as policy questions
usually do. It misses synonyms and paraphrases. The natural upgrade is hybrid retrieval: BM25 plus
an embedding search, merged (for example by reciprocal rank fusion) and reranked.

**SQLite instead of a search server.** One file, no service to run, transactional builds. FTS5
handles far larger corpora than this one on a single machine; the limits would be concurrent
writes and incremental updates, which a rebuild-only design avoids.

**Lexical checker vs. entailment model vs. LLM judge.** The lexical checker is transparent and
free but shallow. A natural-language-inference model scores whether a premise entails a
hypothesis and would catch negation flips and accept paraphrases, at the cost of a model
dependency and less explainable scores. An LLM judge is the most flexible and the most expensive,
and it can be wrong in the same ways as the model it grades. citecheck keeps the lexical checker
as the always-on baseline and adds the judge as an optional second opinion.

**Claims, by rule.** One sentence can contain several claims, so the default checker splits
compound sentences and labels each part. The split and alignment rules are deliberately simple:
a split that leaves a short claim can lose context and lower its word overlap (one false flag on
test), and a claim that merges two source sentences is compared with only one of them.

**Fixed thresholds.** 0.80 and 0.50 are round numbers, not fitted values. A larger labeled set would
allow choosing them deliberately, for example by setting a target false-flag rate.

**Scaling up.** For far more documents: incremental indexing (an external-content FTS5 table can
be kept in sync row by row, typically with triggers), hybrid retrieval, a reranker, and batching evaluation calls through the
Message Batches API. Prompt caching matters more as the shared prefix grows.

## 12. Questions to expect

**Why does recall@5 come out at 1.000?** Because the questions use the handbook's own words and the
corpus is tiny (103 chunks). It shows retrieval is wired correctly, not that it would be perfect
on real queries.

**Why is the unsupported-sentence rate 12%?** Because the fixtures were written to include errors:
the checker labels 6 of 49 sentences unsupported, and all 6 are deliberate mistakes. It marks 3
more weakly supported: one deliberate embellishment and two correct sentences whose wording
differs too much from the passage.

**Why not trust the model's own citations?** That is exactly what the checker is for: citations are
verified against what was actually retrieved, and claims against what the cited text actually
says.

**What stops a document from hijacking the model?** Passages are escaped so they cannot break out of
their element, the system prompt marks them as data, and whatever comes back is checked
independently. None of that makes injection impossible; it makes it visible.

**What would you build next?** Run the evaluation against live models with labels from other
people, add an entailment checker and compare it with the claim-level one on `eval/claims.json`,
and move to hybrid retrieval.
