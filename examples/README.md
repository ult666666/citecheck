# Examples

## `handbook/`: a fictional employee handbook

The 25 Markdown files in `handbook/` are the demo corpus for citecheck. They describe
the policies of **Larkmoor Analytics, a company that does not exist**. Every document was
written from scratch for this project; none of it is copied from a real handbook, and none
of it is advice about real policies.

Each file starts with YAML front matter (`fictional: true` plus a notice). citecheck's
chunker skips front matter, so the label stays out of the search index and out of the
passages shown to a model.

The documents are short (89 to 206 words each, about 3,000 in total) and deliberately
specific: numbers, deadlines, exceptions and approval rules, so that a wrong answer is easy
to spot. A few details are deliberately missing (part-time PTO, an international per diem,
a dental deductible) so that the evaluation can include in-domain questions the corpus
cannot answer.

```bash
uv run citecheck index examples/handbook
```

## `prices.example.toml`: a placeholder price table

Shows the format of the price table that `--prices` reads. **The numbers in it are
placeholders, not Anthropic's prices.** Copy it and fill in the prices you actually pay.
