# Unified benchmark comparison

**Generated:** 2026-03-10T15:48:29.614493+00:00

## Rows (by benchmark mode and model)

| Benchmark mode | Target | Model | RMSE | MAE | R² | Zero rate | MAE (zero) | MAE (non-zero) |
|----------------|--------|-------|------|-----|-----|-----------|------------|-----------------|
| representative_proxy | proxy | baseline | 1.3601419245086808 | 1.136464694583115 | -0.00046101388598773774 | 0.495 | 1.0354657088412753 | 1.2354637004092748 |
| representative_h2 | h2 | baseline | 0.9238977545391578 | 0.73142989009128 | -1.412743539241923e-05 | 0.635 | 0.5731947307865529 | 1.006715715183066 |
| temporal_proxy | proxy | baseline | 1.3914345102518908 | 1.1605279355184404 | -0.00017784802985598702 | 0.521 | 1.0351079532066882 | 1.2969221662824713 |
| temporal_h2 | h2 | baseline | 1.0743735503877943 | 0.8048642044562128 | -0.0325876761960624 | 0.6018 | 0.5101414764158019 | 1.250272237058638 |
| temporal_h2 | h2 | hurdle | 0.8652560787841556 | 0.4777701292043963 | 0.33026092260902895 | 0.6018 | 0.1472949705051603 | 0.9772100306972267 |
| temporal_h2 | h2 | ridge | 0.7902550343424783 | 0.5477766973196253 | 0.4413357720153509 | 0.6018 | 0.34208518832078993 | 0.8586337898667374 |
| representative_proxy | proxy | xgboost | 0.8883942800514693 | 0.6128458925540756 | 0.57318151269131 | 0.495 | 0.35556535618473784 | 0.8650317648368918 |
| representative_h2 | h2 | xgboost | 0.6193967723203442 | 0.39296186271917405 | 0.5505346798414003 | 0.635 | 0.23769115935277751 | 0.6630903466579734 |
| temporal_proxy | proxy | xgboost | 0.8963681566152485 | 0.6203598858752802 | 0.5849269557085531 | 0.521 | 0.3976072632204527 | 0.8626033630124054 |
| temporal_h2 | h2 | xgboost | 0.7700822826976326 | 0.47270618499942557 | 0.4694936523132517 | 0.6018 | 0.2054347920958972 | 0.8766276133724271 |
| temporal_h2 | h2 | year_conditioned | 1.2685644159725018 | 0.7010030655452112 | -0.4395984533223012 | 0.6018 | 0.0 | 1.7604137134744402 |

## Missing benchmark runs

- **representative_proxy** / **ridge**: No metrics artifact found for this combination.
- **representative_proxy** / **year_conditioned**: No metrics artifact found for this combination.
- **representative_proxy** / **hurdle**: No metrics artifact found for this combination.
- **temporal_proxy** / **ridge**: No metrics artifact found for this combination.
- **temporal_proxy** / **year_conditioned**: No metrics artifact found for this combination.
- **temporal_proxy** / **hurdle**: No metrics artifact found for this combination.
- **representative_h2** / **ridge**: No metrics artifact found for this combination.
- **representative_h2** / **year_conditioned**: No metrics artifact found for this combination.
- **representative_h2** / **hurdle**: No metrics artifact found for this combination.

---
Eligibility filtering and target semantics (e.g. horizon) apply per run; see run metadata in each metrics JSON.