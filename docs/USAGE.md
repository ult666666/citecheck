# Using citecheck

- [Install the CLI](#install-the-cli)
- [More of the offline demo](#more-of-the-offline-demo)
- [Live mode with your own key](#live-mode-with-your-own-key)
- [The web page and JSON API](#the-web-page-and-json-api)

## Install the CLI

To index and search your own `.md` and `.txt` files, install the CLI from the GitHub release
(Python 3.12 or newer):

```bash
pip install https://github.com/ult666666/citecheck/releases/download/v0.2.0/citecheck-0.2.0-py3-none-any.whl
citecheck index path/to/docs
citecheck search "your question"
```

BM25 scores depend on the corpus. On a handful of short files every score can stay below the
default `--min-score` of 3, so results show as `[below threshold]`. Index more files, or pass a
lower `--min-score`.

Outside a clone, `ask` and `serve` need `--live` and your own API key, because offline answers
replay the repository's recorded fixtures. `serve` also needs the optional `api` extra:

```bash
pip install "citecheck[api] @ https://github.com/ult666666/citecheck/releases/download/v0.2.0/citecheck-0.2.0-py3-none-any.whl"
```

## More of the offline demo

From a clone, after `uv sync` and `uv run citecheck index examples/handbook`:

```bash
uv run citecheck search "Who founded Larkmoor Analytics?"            # retrieval only: scores, threshold
uv run citecheck ask "What is Larkmoor's stock ticker symbol?"       # abstains without a model call
uv run citecheck ask --judge "Do I need a doctor's note when I take a sick day?"
uv run citecheck ask --json "How many unused PTO days carry over into the next year?"
uv run citecheck ask --dry-run "How many unused PTO days carry over into the next year?"
uv run citecheck serve                                               # http://127.0.0.1:8000
```

Without `--live`, the model is a fixture: a hand-written synthetic response from `eval/fixtures`,
replayed instead of an API call. The prepared questions are listed in
[`eval/questions.json`](../eval/questions.json). A question that gets past retrieval but has no
prepared response is refused with an error, never answered with an invented one. `--dry-run`
prints the exact Messages API request without calling anything.

## Live mode with your own key

Live mode sends requests to the Anthropic Messages API through the official `anthropic` SDK.
citecheck never reads, stores or prints the key; the SDK finds credentials in
`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or a profile saved by `ant auth login`.

```bash
export ANTHROPIC_API_KEY="your-key"     # or: ant auth login
uv run citecheck index path/to/your/docs
uv run citecheck ask --live "Your question?"
uv run citecheck ask --live --model MODEL_ID --effort low "Your question?"
uv run citecheck ask --live --prices my-prices.toml "Your question?"
uv run citecheck serve --live
uv run citecheck eval --live              # real API calls: one per question that clears retrieval
```

| Option | Effect |
| --- | --- |
| `--model MODEL_ID` | The model to call; `citecheck ask --help` shows the default |
| `--effort LEVEL` | Sets `output_config.effort` (default `medium`); `none` omits it for models that do not support it |
| `--no-fallback` | Turns off server-side refusal fallback |
| `--no-cache` | Turns off the prompt-cache breakpoint after the system prompt |
| `--prices FILE` | Estimates cost from a price table you supply |

Each request uses a strict `submit_answer` tool. The API validates the arguments against the
schema, citecheck validates them again, and a reply without the tool call is retried once. No
`temperature` or `thinking` parameter is sent. When a refusal fallback serves an answer,
citecheck says so and prices each attempt by the model that ran it.
[DESIGN.md](DESIGN.md#5-prompt-and-schema-design) explains each of these choices, including why
forced tool choice is not used and how big the cached prefix is.

No prices are built in. Start from
[`examples/prices.example.toml`](../examples/prices.example.toml), whose numbers are
placeholders, and fill in what you pay.

The default score threshold (3.0) was tuned on the demo corpus, so run `citecheck search` on a
few of your own questions and set `--min-score` to match. In live mode the evaluation cannot
report checker-vs-human agreement, because the human labels belong to the fixture sentences.

## The web page and JSON API

`citecheck serve` starts the JSON API (`POST /api/ask`, `POST /api/search`, `GET /healthz`) and
an HTML page at `/`. The page shows each answer sentence with its label, its reason and its
citations, and lists the passages offered to the model. It accepts the same `--live`,
`--checker` and retrieval options as `ask`.
