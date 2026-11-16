# Contributing

Contributions that improve CPU speed, table structure handling, or confidence-cascade accuracy are welcome.

## Benchmark & Evaluation Rules

1. **Report gains *and* regressions.** Never cherry-pick improved pages or hide degraded tables/text blocks.
2. **Preserve text conservation.** Table repairs must not invent, duplicate, or drop Indra's recognized fragments (unassigned fragments stay in an escaped appendix if not placed in a cell).
3. **Separate development from held-out evaluation.** Do not tune thresholds or heuristics on held-out confirmation pages and then re-report those same pages as unseen validation.
4. **No private documents or credentials.** Only commit synthetic test fixtures or explicitly rights-cleared public test cases. Never commit API keys, model weights, or local home-directory paths.

## Running Offline Tests

All unit tests run offline with zero downloaded weights:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
```
