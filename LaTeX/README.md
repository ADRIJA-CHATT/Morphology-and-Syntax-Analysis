# Information Locality: fully self-contained beginner-readable LaTeX + Markdown regeneration bundle

This bundle contains the complete report source and the analysis inputs needed to regenerate the PDF.

## Build

From this directory:

```bash
cd report
python regenerate_tables.py
./build.sh
```

The result is `report/report.pdf`.

## What is included

- `report/report.tex` - the complete self-contained report, including a mathematical primer, worked toy example, optimization primer, and glossary.
- `report/references.bib` - bibliography metadata.
- `report/tables/` - LaTeX table fragments generated from the supplied result files.
- `report/regenerate_tables.py` - rebuilds the table fragments from the included CSV files.
- `report/build.sh` - XeLaTeX build script.
- `figures/` - all 42 report figures.
- `data/` - the result snapshot used by the report.
- `markdown/BEGINNER_GUIDE.md` - a compact introduction to every major mathematical idea.
- `markdown/FORMULA_SHEET.md` - the principal formulas with plain-language definitions.
- `markdown/METHODS.md` - the computational method in beginner-readable form.
- `markdown/DATA_STORY.md` - interpretation of the language-level results.
- `markdown/LANGUAGE_COVERAGE.md` - the 102-language registry and structural-annotation coverage note.
- `markdown/PLOT_CATALOG.md` - descriptions of all 42 figures.
- `markdown/CITATIONS_AND_DATA_SOURCES.md` - data and method references.

## Mathematical scope

The report defines, before use, elementary notation, averages, variance, probability, conditional probability, expectation, logarithms, surprisal, entropy, conditional entropy, mutual information, entropy rate, cross-entropy, language models, n-grams, held-out evaluation, Kneser-Ney smoothing, information locality, monotone correction, exponential decay, least squares, objective functions, nonlinear least squares, half-life, bootstrap, confidence/percentile intervals, power laws, correlation, p-values, regression, standardization, one-hot encoding, imputation, interaction terms, Ridge regression, Bayesian Ridge, Elastic Net, cross-validation, LOCO-CV, MAE, RMSE, and `R^2`.

It also explains Universal Dependencies, treebanks, morphological features, dependency structure, tokens, types, and the entropy-based morphology index.

## Morphology coverage

The morphology index is a text-derived entropy statistic. A UD treebank is not required by its mathematical definition. The current result has 58 populated morphology-index values and 44 missing values because the current processing reaches the text-perturbation calculation from the UD-processing branch and those 44 languages have no matching UD treebank. Their locality estimates are still present.

No missing morphology value is imputed in the report.
