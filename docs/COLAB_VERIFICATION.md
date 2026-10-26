# Google Colab verification

The full notebook ran in Google Colab on October 26, 2026 using a T4 GPU and public source commit `ba00c8ee3b03c2f21e5128fda92d1ed4b1334b52`. Its downloaded output was verified on October 27. The [executed notebook](../notebooks/WasteSort_Colab_Executed.ipynb) preserves the downloaded bytes: all six code cells have execution counts 1-6, saved outputs, and no error outputs. The [verification record](../reports/colab_verification.json) records its checksum.

The notebook completed setup, dataset/split preparation, training, validation selection, held-out comparison, uncertainty summaries, regression and archive creation. It reports 1,774 training, 371 validation and 376 test images, with the same six excluded files. MobileNetV2 remained selected (validation macro F1 0.8474362310); displayed test accuracy was 84.5745% and macro F1 81.2199%.

## Separate run, separate results

| Model | Colab test accuracy | Colab macro F1 |
|---|---:|---:|
| SVM | 65.9574% | 63.5168% |
| Random Forest | 63.5638% | 60.2211% |
| CNN | 55.8511% | 53.9926% |
| MobileNetV2 | 84.5745% | 81.2199% |
| Weighted CNN | 56.6489% | 56.8920% |

Values above are rounded from the saved notebook display. CNN and weighted-CNN results differ from the original Windows benchmark; hardware/platform differences can affect neural training. This is not an exact reproduction of every model. The original reports and deployed checkpoints remain unchanged. Regression completed with 89 test predictions, MSE 2900.1936284934814 and R-squared 0.4526027629719195.

Colab created an experiment ZIP, but that ZIP was not retrieved before the runtime disconnected. Therefore its full checkpoint and prediction hashes were not independently compared. A browser output-display error did not prevent execution: the downloaded notebook retains the outputs. This run was assisted through the shared chat, not an independent human replication.

The separate Tuesday notebook has a verified eight-cell local execution. Its separate Colab execution remains unverified: browser reconnection returned `Debugger unattached`. The full Colab run confirms dataset preparation in the full pipeline but does not substitute for claiming that Tuesday notebook itself ran in Colab.
