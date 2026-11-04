# textbook_toy

`textbook_toy` is a two-record Chinese dataset packaged for offline documentation, examples, and tests.

- Language: Chinese.
- Task: factual-question answering with sampled black-box responses.
- `unit_type`: `sentence`.
- `unit_labels`: `true` means the sentence is intentionally hallucinated; `false` means it is supported by the teaching facts.
- `response_label`: response-level counterpart of `unit_labels`, used by response-level benchmarks.
- `label_source`: `textbook_toy`; the labels are hand-authored teaching facts.
- `claim_labels`: `supported` or `contradicted` against the small textbook fact set.

It is a didactic fixture, not a benchmark. It must not be used to estimate method performance or tune production thresholds.
