# Evaluation results

citecheck has two offline evaluations. Both print deterministic reports, and
`tests/test_readme.py` fails if a report below drifts from what its command prints.
[DESIGN.md](DESIGN.md#7-evaluation-methodology) defines the metrics, and
[claim-level checking](DESIGN.md#claim-level-checking) describes the protocol for the claim cases.

- [Held-out claim cases](#held-out-claim-cases) score the two checkers on 140 hand-labelled
  sentences, with no retrieval and no model.
- [The end-to-end evaluation](#end-to-end-evaluation) runs retrieval, answering, the checker and
  abstention on 36 questions, with hand-written fixture responses in place of a model.

## Held-out claim cases

`eval/claims.json` holds 140 sentences written against handbook chunks. Each was labelled
supported or unsupported by reading only the chunks it cites. They cover restatements,
paraphrases, compound sentences, inferences and numbers taken from the question, and errors in
negation, antonyms, modality, numbers, units, dates, swapped attributes and entities.

A hash of each case ID put 63 cases in dev and 77 in test. The file was committed before any
checker change, and the checker was tuned on dev only.

```bash
uv run citecheck eval-claims                # the 77 held-out test cases
uv run citecheck eval-claims --split dev    # the 63 cases used for tuning
```

```text
citecheck eval-claims: 77 test cases (48 unsupported, 29 supported) from claims.json
settings: supported>=0.80, weak>=0.50; flagged means any label but supported

                                lexical     claims
  caught unsupported            20 of 48    44 of 48
  false flags on supported      1 of 29     2 of 29
  accuracy                      0.623       0.922

Paired: only claims right on 24 cases, only lexical right on 1; exact McNemar p = 1.5e-06
By category (cases judged correctly, lexical / claims)
  antonym               7 cases     1 / 7
  compound              7 cases     6 / 6
  contradiction         5 cases     4 / 5
  date                  2 cases     1 / 2
  entity                1 cases     1 / 1
  modality              1 cases     0 / 1
  negation              5 cases     1 / 5
  negation-kept         4 cases     4 / 4
  number                8 cases     7 / 8
  paraphrase            9 cases     8 / 8
  restatement          13 cases    13 / 13
  swapped-attribute     9 cases     0 / 5
  unit                  6 cases     2 / 6
```

The claim-level checker catches 44 of the 48 unsupported test sentences. The lexical checker
catches 20. The claim-level checker wrongly flags 2 of the 29 supported sentences, one more than
the lexical checker.

Paired on the same cases, the claim-level checker is right where the lexical checker is wrong 24
times, and the reverse happens once. An exact McNemar test gives p = 1.5e-06.

| Split | Checker | Unsupported caught | Supported flagged | Accuracy |
| --- | --- | ---: | ---: | ---: |
| test (held out) | lexical | 20 of 48 | 1 of 29 | 0.623 |
| test (held out) | claims | 44 of 48 | 2 of 29 | 0.922 |
| dev (tuning) | lexical | 6 of 34 | 8 of 29 | 0.429 |
| dev (tuning) | claims | 28 of 34 | 7 of 29 | 0.794 |

When the checker was frozen after dev tuning, its first score on the test split was 43 of 48
caught. It had the same 2 false flags. Two later fixes came from corpus-wide tests, not from test
results: every handbook sentence copied verbatim must pass, and swapping a word for its opposite
must fail. Those fixes changed one test case, which is now flagged, though for a spurious reason:
the verb "owns" matched the opposite pair "own" and "another".

The same person wrote the cases and the rules, and had read every case, so the held-out figures
are optimistic. [DESIGN.md](DESIGN.md#claim-level-checking) lists every remaining miss.

## End-to-end evaluation

```bash
uv run citecheck eval
```

```text
citecheck eval: 36 questions (28 answerable, 8 unanswerable) from questions.json
settings: k=5, min_score=3, supported>=0.80, weak>=0.50
responses: fixture replay: a hand-written synthetic response, not a real model output.
These numbers exercise the harness and the checker; they do not measure any model.

Retrieval (28 answerable questions)
  recall@5                      1.000    share of gold chunks offered, per question
  hit@5                         1.000    questions with at least one gold chunk offered
Citations
  precision                     0.879    29 of 33 cited chunk IDs are gold
  recall                        0.935    29 of 31 gold chunk IDs were cited
  rejected citations            2        cited chunks that were not offered
Verification (49 answer sentences)
  supported                     40
  weakly supported              3
  unsupported                   6
  unsupported-sentence rate     0.122
Abstention
  accuracy                      0.944    34 of 36 questions
  unanswerable, abstained       7 of 8   2 without a model call
  unanswerable, answered        1        answers that should not exist
  answerable, abstained         1 of 28
Checker vs. human labels in the fixtures
  flagged 7 of 7 sentences labelled unsupported (missed 0)
  flagged 2 of 42 sentences labelled supported (false flags)
Tokens (estimated at ~4 characters per token, not measured): 34 model calls, 34,802 input, 2,133 output, 0 cache-read, 0 cache-write
```

Read this as a test of the harness, not of a model. The responses are hand-written fixtures, and
several are wrong on purpose; [`eval/README.md`](../eval/README.md) lists each deliberate error.
One author wrote the questions, gold labels, fixtures and human labels, and chose the 3.0 score
threshold by looking at this same question set. 36 questions are far too few for statistical
claims.

What the numbers do show:

- Retrieval offers every gold chunk (recall@5 = 1.000). The corpus has 103 chunks and the
  questions use the documents' own words, which is an easy setting for BM25.
- The 4 citations that are not gold are the deliberate errors (an over-cited passage, a typo in a
  chunk ID, a chunk ID that does not exist) plus the citation in the one answer to an
  unanswerable question. The 2 rejected citations are the typo and the nonexistent ID.
- The checker flags all 7 sentences a human labelled unsupported: the invented 60-day rule, a
  wrong $25 fee, an uncited opinion, a sentence whose only citation is mistyped, an added "notify
  your manager", the negation flip shown in the README, and the domestic per diem given for
  international trips ("international" where the cited text says "domestic").
- It falsely flags 2 correct sentences. One is a paraphrase in different words. The other
  correctly places the question's $5,000 inside the range the passage gives, but its other half
  shares too few words with the passage.
- The original lexical checker (`--checker lexical`) flags 5 of the 7. These fixture sentences
  guided the claim-level checker, so they are not a fair test of it; the held-out claim cases
  above are.
