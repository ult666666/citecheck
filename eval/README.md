# Evaluation data

## `questions.json`

36 hand-labeled questions about the fictional handbook in `examples/handbook/`:

- **28 answerable** questions, each with the IDs of the chunks that contain the answer
  (`gold_chunks`). Three need two chunks, one of them from two different documents.
- **8 unanswerable** questions: out-of-domain ones ("Who founded Larkmoor Analytics?"),
  in-domain gaps where the corpus is close but silent (part-time PTO, an international
  per diem, a dental deductible), and one with no searchable words at all.

Each question has a `notes` field saying what it tests. `citecheck eval` checks that
every gold chunk ID exists in a freshly built index, so a typo in the labels fails loudly.

## `fixtures/`: hand-written synthetic responses

**These files are not real model outputs.** Each one is a hand-written `submit_answer`
tool input for one question, replayed by citecheck's fixture client so that the whole
pipeline can run offline, in tests and in CI, with no API key. Every file declares
`"synthetic": true`, and the loader rejects files that do not.

Most fixtures are faithful answers. Some contain deliberate errors so that the checker
and the metrics have something real to catch, and each such file's `note` says so:

| Fixture | Deliberate behavior | What the checker does |
|---|---|---|
| q13 | invents a 60-day rule | unsupported (number not in the passage) |
| q16 | states a $25 fee where the passage says $20 | unsupported (number) |
| q17 | adds an uncited opinion | unsupported (no citation) |
| q19 | cites `security/device.md` (typo) | citation rejected, sentence unsupported |
| q20 | adds "notify your manager" | weakly supported (4 of 6 words) |
| q22 | adds a citation to a chunk ID that does not exist | citation rejected, sentence still supported |
| q08 | cites an irrelevant extra passage | still supported; citation precision drops |
| q24 | answers half and flags insufficient evidence | status "partial" |
| q28 | abstains although the answer was retrieved | false abstention |
| u02 | answers an unanswerable question with the domestic per diem | **missed**: 5 of 6 words match |
| q03 | flips a negation ("not required") | **missed**: every word matches |
| q15 | correct paraphrase in different words | **false flag**: too few shared words |
| q17 | correct sentence that repeats the question's $5,000 | **false flag**: number not in the passage |

The last column describes the lexical checker (`--checker lexical`). The default claim-level
checker also catches q03 (only the sentence is negated) and u02 ("international" where the cited
text says "domestic"), and accepts q17's $5,000 inside its "$1,000 to $10,000 range" comparison.

`expected_labels` holds a human judgment ("supported" or "unsupported") for each sentence,
written by the same person who wrote the fixtures. The evaluation compares the checker's
labels with them. That comparison is small and not independent, so treat it as a
demonstration of the checker's behavior, not as an accuracy estimate.

Questions u03 and u07 have no fixture on purpose: retrieval abstains before any model call.
q03 and q15 also carry a `judge` response, for demonstrating `citecheck ask --judge`.

## `claims.json`: hand-labelled claim cases

140 sentences that score the support checker directly, without retrieval or a model. Each case
names one or two real chunk IDs of the handbook, a `sentence`, a human `label` ("supported" or
"unsupported"), a `category` (what the case tests) and, for three cases, the `question` whose
number the sentence repeats.

How they were labelled: each sentence was written against its cited chunks, to restate them
faithfully, paraphrase them, combine two of their facts, or introduce one kind of error (a
flipped negation, an antonym, a changed modal verb, a wrong number, unit or date, a value moved
to another attribute, another entity, or an added clause). The label was then decided by reading
only the cited chunks: "supported" when they state or directly imply everything in the sentence,
"unsupported" otherwise. The same person wrote the cases, the labels and the checker.

`split` comes from a SHA-256 of the case ID (dev when it is 0 or 1 modulo 5): 63 dev cases, used
to tune the checker, and 77 test cases, held out. The file was committed before any checker
change, the loader rejects a case whose split does not match the rule, and a test pins the
file's hash. `uv run citecheck eval-claims` scores both checkers on the test cases;
`--split dev` and `--split all` select the others, and `--details` lists every disagreement.
