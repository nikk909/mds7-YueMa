# Green AI Trade-Off — Results

## Experiment Summary

| Experiment | Data Fraction | Epochs | Train Samples | Test Accuracy | CO₂ Emissions (g) |
|------------|---------------|--------|---------------|---------------|-------------------|
| A | 50% | 50 | 30,000 | 88.57% | 0.0003 |
| B | 50% | 100 | 30,000 | 88.33% | 0.0006 |
| C | 100% | 50 | 60,000 | 89.74% | 0.0007 |
| D | 100% | 100 | 60,000 | 89.97% | 0.0013 |

## Winning Configuration: Experiment C

- **Accuracy:** 89.74%
- **CO₂ Emissions:** 0.0007 g
- **Dataset:** 100%  |  **Epochs:** 50

### 1. Data Size: Doubling Effect

Doubling the dataset from 50% → 100% (A→C, B→D):

- **Accuracy did NOT double.** A→C gained only **+1.17 pp** (88.57% → 89.74%); B→D gained **+1.64 pp** (88.33% → 89.97%). Absolute accuracy gains are small, not proportional to data size.
- **Emissions roughly doubled (or more).** A→C: 0.0003 → 0.0007 g (~2.3×); B→D: 0.0006 → 0.0013 g (~2.2×). More data meaningfully raises carbon cost, but only modestly improves accuracy.

### 2. Epochs: Doubling Effect

Doubling epochs from 50 → 100 (A→B, C→D):

- **A→B:** Accuracy slightly *dropped* (88.57% → 88.33%) while CO₂ doubled (0.0003 → 0.0006 g). Extra epochs on half data wasted electricity with no benefit.
- **C→D:** Accuracy rose only **+0.23 pp** (89.74% → 89.97%) while emissions nearly doubled (0.0007 → 0.0013 g). The marginal gain does not justify the carbon cost.

### 3. Production Deployment Choice

I would deploy **Experiment C** (100% data, 50 epochs) because:

1. It reaches near-max accuracy (**89.74%**, within ~0.23 pp of D) at about **half** D’s emissions.
2. Full data helps more than longer training: C beats A/B on accuracy, while 100 epochs (B, D) add cost without meaningful ROI.
3. Under strict cloud budget and carbon monitoring, C offers the best accuracy-per-gram trade-off for production.
