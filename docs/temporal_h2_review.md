# Temporal H2 benchmark review

## 1. Benchmark definition

- **Target:** Citations from publication year through publication year + 2 (3 calendar years), from `counts_by_year`. Transform: `log1p`.
- **Split:** Time-based. Train: `publication_year <= 2018`; test: `publication_year >= 2019`.
- **Eligibility:** Row included only if `publication_year + horizon_years <= max_available_citation_year`.
- **Features:** publication_year, referenced_works_count, authors_count, institutions_count; categorical: type, language, venue_name, primary_topic, open_access_is_oa.
- **Models:** Baseline (median-predictor) and XGBoost.
- **Limitations:** Calendar-year granularity; empty/missing `counts_by_year` → zero; target is not dataset-level `cited_by_count`.

## 2. Benchmark population

- Processed rows: 1000
- Target-eligible rows: 1000
- Excluded (incomplete horizon): 0
- Train rows: 666; test rows: 334
- Train year range: [2015, 2018]; test year range: [2019, 2020]
- Target zero-rate train: 0.6742; test: 0.6018

**Train target distribution (transformed):** {
    "count": 666,
    "min": 0.0,
    "q05": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.6931471805599453,
    "q95": 2.3978952727983707,
    "max": 4.382026634673881,
    "mean": 0.5101414764158018
  }

**Test target distribution (transformed):** {
    "count": 334,
    "min": 0.0,
    "q05": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 1.0986122886681098,
    "q95": 2.9794681896613806,
    "max": 4.804021044733257,
    "mean": 0.7010030655452112
  }

## 3. Baseline vs XGBoost

| Metric | Baseline | XGBoost |
|--------|----------|---------|
| RMSE   | 1.0744 | 0.7701 |
| MAE    | 0.8049 | 0.4727 |
| R²     | -0.0326 | 0.4695 |
| Spearman | N/A | 0.67145100662018 |

## 4. Subgroup analysis (test set)

See `artifacts/diagnostics/temporal_h2_subgroup_metrics.json` for full slices.

| Publication year | n | MAE (XGBoost) |
|------------------|---|---------------|
| 2019 | 167 | 0.4648 |
| 2020 | 167 | 0.4806 |

| Slice | n | MAE baseline | MAE XGBoost |
|-------|---|--------------|-------------|
| Zero target | 201 | 0.5101414764158019 | 0.2054347920958972 |
| Nonzero target | 133 | 1.250272237058638 | 0.8766276133724271 |

## 5. Zero-inflation

- Test rows with target = 0: 201
- Test rows with target > 0: 133
- MAE baseline on zero-target: 0.5101414764158019; on nonzero: 1.250272237058638
- MAE XGBoost on zero-target: 0.2054347920958972; on nonzero: 0.8766276133724271

## 6. Feature signal

Top feature importances (XGBoost):

- referenced_works_count: 0.0928
- institutions_count: 0.0646
- venue_name_ChemInform: 0.0310
- language_en: 0.0297
- authors_count: 0.0264
- primary_topic_Web visibility and informetrics: 0.0221
- venue_name_Metrika: 0.0214
- primary_topic_Polyamine Metabolism and Applications: 0.0205
- venue_name_Blood: 0.0193
- primary_topic_Organ Transplantation Techniques and Outcomes: 0.0190
- venue_name_Automation in Construction: 0.0181
- venue_name_Proceedings of MOL2NET, International Conference on Multidisciplinary Sciences: 0.0181
- venue_name_Preparative Biochemistry & Biotechnology: 0.0171
- primary_topic_Populism, Right-Wing Movements: 0.0166
- open_access_is_oa_False: 0.0154

Full list: `artifacts/diagnostics/temporal_h2_feature_importance.json`.

## 7. Comparison to other benchmarks

No other benchmark metrics found in artifacts/metrics.

## 8. Interpretation and summary

- **What the temporal H2 result suggests:** Generalization to later publication years with a fixed 3-year citation window; R² and Spearman indicate whether metadata adds predictive signal beyond the baseline.
- **Is the model learning real signal?** Compare XGBoost R²/Spearman to baseline; check feature importance and subgroup MAE to see if gains are broad or driven by a few slices.
- **Benchmark credibility:** If XGBoost materially outperforms baseline and error is stable across years and zero/nonzero slices, the benchmark is usable for further work.
- **Main limitations:** Calendar-year target approximation; metadata-only; zero-inflation; simple baselines.
- **Sensible next steps:** Stronger baselines (e.g. year-stratified median), feature engineering (venue/topic embeddings or bins), calibration for zero-heavy targets, or (later) text features.

## 9. Limitations (explicit)

- Calendar-horizon target is an approximation (calendar years, not exact months).
- Empty/missing `counts_by_year` is treated as zero yearly-count series.
- Target distribution is not the same as dataset-level `cited_by_count`.
- Venue/topic semantics may reflect source data rather than canonical taxonomy.
