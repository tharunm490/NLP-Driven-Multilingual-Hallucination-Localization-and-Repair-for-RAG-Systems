# Verification performance benchmark

Generated: `2026-10-27T13:09:22.133071+00:00`

## Environment

- Platform: Windows-11-10.0.26200-SP0
- Python: 3.12.10
- CPU: 12 logical / 6 physical
- Memory: 15.3 GB
- torch: 2.14.1
- transformers: 5.20.1
- onnxruntime: 1.31.0

## Dataset

- NLI cases: 36 (18 short, 18 long)
- Engine scenario `short-request`: 3 claims, 3 evidence chunks
- Engine scenario `long-request`: 13 claims, 6 evidence chunks

Small controlled ISO/IEC 27001 fixture; not a substitute for a public NLI benchmark.

## Engine results

| Config | Run | Wall s | Verify s | Extract ms | Select ms | NLI ms | Checked | Coverage | Pairs | Pair coverage | Incomplete | Accuracy | Checked accuracy | False support | False contradiction | Peak MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unbatched-exhaustive | long-first | 21.9 | 21.9 | 1243.0 | 64.9 | 20633.7 | 13/13 | 100.0% | 78/78 | 100.0% | 0 | 92.3% | 92.3% | 0 | 0 | 3488.6 |
| unbatched-exhaustive | long-repeat | 0.9 | 0.9 | 825.9 | 67.4 | 0.8 | 13/13 | 100.0% | 78/78 | 100.0% | 0 | 92.3% | 92.3% | 0 | 0 | 3488.6 |
| unbatched-exhaustive | short-warm | 0.2 | 0.2 | 227.0 | 10.0 | 0.1 | 3/3 | 100.0% | 9/9 | 100.0% | 0 | 100.0% | 100.0% | 0 | 0 | 3488.6 |
| batched-exhaustive | long-first | 16.9 | 16.9 | 908.8 | 54.3 | 15952.2 | 13/13 | 100.0% | 78/78 | 100.0% | 0 | 92.3% | 92.3% | 0 | 0 | 3258.1 |
| batched-exhaustive | long-repeat | 1.0 | 1.0 | 890.0 | 61.4 | 0.3 | 13/13 | 100.0% | 78/78 | 100.0% | 0 | 92.3% | 92.3% | 0 | 0 | 3258.1 |
| batched-exhaustive | short-warm | 0.2 | 0.2 | 198.7 | 8.8 | 0.1 | 3/3 | 100.0% | 9/9 | 100.0% | 0 | 100.0% | 100.0% | 0 | 0 | 3258.1 |
| legacy-8x2-12pair | long-first | 5.4 | 5.4 | 1953.2 | 24.7 | 3389.7 | 0/13 | 0.0% | 12/78 | 15.4% | 13 | 0.0% | n/a | 0 | 0 | 3436.9 |
| legacy-8x2-12pair | short-warm | 0.6 | 0.6 | 574.7 | 16.3 | 0.1 | 0/3 | 0.0% | 6/9 | 66.7% | 3 | 0.0% | n/a | 0 | 0 | 3436.9 |

## Conclusions

- unbatched-exhaustive long request: 21.9s, 13/13 checked, 78/78 pairs evaluated, 100.0% claim coverage, 92.3% accuracy, 0 false support.
- batched-exhaustive long request: 16.9s, 13/13 checked, 78/78 pairs evaluated, 100.0% claim coverage, 92.3% accuracy, 0 false support.
- legacy-8x2-12pair long request: 5.4s, 0/13 checked, 12/78 pairs evaluated, 0.0% claim coverage, 0.0% accuracy, 0 false support.

## Method and limitations

- Engine runs use the labelled ISO/IEC 27001 fixture with a lexical matcher and supplied evidence. RAG retrieval, embedding generation, and the LLM are excluded so the measurements isolate verification.
- The engine loads NLI and NER before timing request runs, so `load_ms` is reported separately and request timings are warm.
- `long-first` and `long-repeat` use the same engine instance to measure model reuse.
- Times are single-process CPU measurements. Accuracy is on a small controlled fixture and is not a claim of general NLI accuracy.
