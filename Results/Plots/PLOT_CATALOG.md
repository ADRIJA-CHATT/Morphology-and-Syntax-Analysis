# Plot catalog

### 01_global_locality_curves_characters.png
Each faint line is one language's locality curve; the mean, median and interquartile band summarize the cross-language distribution.

### 02_global_locality_curves_seconds.png
The same locality curves in seconds, showing how speech-rate calibration changes temporal persistence relative to character persistence.

### 03_information_retention_heatmap.png
A language-by-lag heatmap that makes the complete 102-language locality surface visible without relying on 102 overlapping line colours.

### 04_lambda_character_distribution.png
The language-level distribution of the exponential decay parameter in character space.

### 05_lambda_time_distribution.png
The language-level distribution of the exponential decay parameter after speech-time calibration.

### 06_locality_half_life_map.png
Half-life is ln(2)/λ and translates the fitted decay parameter into an interpretable persistence distance.

### 07_lambda_character_extremes.png
Descriptive extremum plot showing the 12 lowest- and 12 highest-rate languages; it is not an overall ranking of languages.

### 08_lambda_time_extremes.png
Descriptive extremum plot showing the 12 lowest- and 12 highest-rate languages; it is not an overall ranking of languages.

### 09_region_lambda_character.png
Boxes summarize within-region distributions and overlaid points show individual languages; sparse groups should be interpreted descriptively.

### 10_region_lambda_time.png
Boxes summarize within-region distributions and overlaid points show individual languages; sparse groups should be interpreted descriptively.

### 11_speech_rate_region.png
Boxes summarize within-region distributions and overlaid points show individual languages; sparse groups should be interpreted descriptively.

### 12_lambda_time_vs_speech_rate.png
The positive association between speech-character rate and λ/second is partly algebraically induced because λ/second = λ/character × speech-character-rate.

### 13_morphology_tradeoff_map.png
D_order and D_structure show the relative contributions of word order and within-word structure; morphology index is the structural share of their sum.

### 14_morphology_index_vs_lambda_character.png
Bivariate diagnostic for morphology_index versus locality_lambda_per_char. The plot is descriptive and should not be interpreted causally; it checks whether a simple language-level covariate visibly tracks locality.

### 15_morphology_index_vs_lambda_time.png
Bivariate diagnostic for morphology_index versus locality_lambda_per_s. The plot is descriptive and should not be interpreted causally; it checks whether a simple language-level covariate visibly tracks locality.

### 16_word_order_redundancy_vs_lambda.png
Bivariate diagnostic for word_order_redundancy versus locality_lambda_per_char. The plot is descriptive and should not be interpreted causally; it checks whether a simple language-level covariate visibly tracks locality.

### 17_word_structure_redundancy_vs_lambda.png
Bivariate diagnostic for word_structure_redundancy versus locality_lambda_per_char. The plot is descriptive and should not be interpreted causally; it checks whether a simple language-level covariate visibly tracks locality.

### 18_memory_cost_vs_lambda.png
Bivariate diagnostic for memory_cost_bits_char versus locality_lambda_per_char. The plot is descriptive and should not be interpreted causally; it checks whether a simple language-level covariate visibly tracks locality.

### 19_ud_morph_feature_density_vs_lambda.png
Bivariate diagnostic for ud_morph_feature_density versus locality_lambda_per_char. The plot is descriptive and should not be interpreted causally; it checks whether a simple language-level covariate visibly tracks locality.

### 20_dependency_distance_vs_lambda.png
Bivariate diagnostic for ud_mean_dependency_distance versus locality_lambda_per_char. The plot is descriptive and should not be interpreted causally; it checks whether a simple language-level covariate visibly tracks locality.

### 21_lexical_ttr_vs_lambda.png
Bivariate diagnostic for lexical_ttr versus locality_lambda_per_char. The plot is descriptive and should not be interpreted causally; it checks whether a simple language-level covariate visibly tracks locality.

### 22_mean_word_length_vs_lambda.png
Bivariate diagnostic for mean_word_length_chars versus locality_lambda_per_char. The plot is descriptive and should not be interpreted causally; it checks whether a simple language-level covariate visibly tracks locality.

### 23_corpus_size_vs_lambda.png
Bivariate diagnostic for n_chars versus locality_lambda_per_char. The plot is descriptive and should not be interpreted causally; it checks whether a simple language-level covariate visibly tracks locality.

### 24_locality_fit_quality.png
Most exponential locality fits are strong; the two weakest are Cantonese Chinese and Mandarin Chinese, which merit sensitivity attention.

### 25_hilberg_alpha_vs_lambda.png
Hilberg α and exponential locality λ summarize different aspects of information structure and show only a weak bivariate association in these results.

### 26_hilberg_fit_distribution.png
83 of 102 Hilberg R² values are negative, contrasting with the high R² of the primary locality exponential fits.

### 27_regression_cv_performance.png
All reported cross-validated R² values are negative, so the current multivariable feature set does not generalize to held-out languages better than the baseline.

### 28_character_observed_vs_predicted.png
Each point is one held-out language and the dashed diagonal is perfect prediction; visible scatter around the diagonal explains the negative cross-validated R².

### 29_time_observed_vs_predicted.png
Each point is one held-out language and the dashed diagonal is perfect prediction; visible scatter around the diagonal explains the negative cross-validated R².

### 30_character_residuals.png
The 20 largest absolute leave-one-language-out residuals identify where the supplied predictor set misses language-specific locality variation.

### 31_time_residuals.png
The 20 largest absolute leave-one-language-out residuals identify where the supplied predictor set misses language-specific locality variation.

### 32_character_regression_coefficients.png
The largest positive and negative fitted coefficients are descriptive model parameters; the negative LOCO-CV R² means they should not be read as validated causal or predictive effects.

### 33_time_regression_coefficients.png
The largest positive and negative fitted coefficients are descriptive model parameters; the negative LOCO-CV R² means they should not be read as validated causal or predictive effects.

### 34_ud_coverage_by_region.png
FLEURS locality is available for all 102 languages, but only 58 have matching UD annotations, so morphology analyses are necessarily a 58-language subset.

### 35_effective_kn_training_chars.png
Effective KN training characters show where the adaptive estimator used less text than the original 100k upper cap, preventing low-resource languages from being rejected solely because they are smaller.

### 36_effective_train_vs_validation.png
The dashed lines mark the originally requested 100k/20k caps; points below them show where actual language-level text availability forced a smaller but valid allocation.

### 37_word_order_redundancy_distribution.png
A direct distribution of one component of the analytic–synthetic trade-off, useful for seeing whether a few languages dominate the cross-language range.

### 38_word_structure_redundancy_distribution.png
A direct distribution of one component of the analytic–synthetic trade-off, useful for seeing whether a few languages dominate the cross-language range.

### 39_morphology_index_by_region.png
The morphology-index distribution differs across broad region groups, but this is an annotated 58-language subset with uneven UD coverage.

### 40_memory_cost_vs_half_life.png
A synthesis map combining cumulative memory cost with information half-life, giving a compact view of the study's efficiency–locality theme.

### 41_morphology_index_vs_lambda_character_102_family.png
All 102 language rows are retained. The 58 languages with a matching UD treebank are plotted at their measured morphology index; the 44 languages without a matching UD treebank are shown in the hatched NA strip at their observed character-domain decay coordinate. No morphology value is imputed.

### 42_morphology_index_vs_lambda_time_102_family.png
All 102 language rows are retained. The 58 languages with a matching UD treebank are plotted at their measured morphology index; the 44 languages without a matching UD treebank are shown in the hatched NA strip at their observed speech-time decay coordinate. No morphology value is imputed.