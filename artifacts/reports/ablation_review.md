# Metadata ablation comparison

**Generated:** 2026-03-10T15:48:29.614527+00:00
**Full model:** xgb_temporal_h2 (available: True)

## Ablations

| Ablation | Features removed | RMSE | MAE | R² | Δ RMSE | Δ MAE | Δ R² | Interpretation |
|----------|------------------|------|-----|-----|--------|-------|------|-----------------|
| categorical_only | publication_year, referenced_works_count | 1.0268931901458365 | 0.7070819151531671 | 0.056662999590753715 | 0.2568109074482039 | 0.23437573015374152 | -0.41283065272249797 | Removing categorical_only hurts R² substantially;  |
| no_primary_topic | primary_topic | 0.7686097454901358 | 0.4777297131782545 | 0.47152056155244804 | -0.001472537207496738 | 0.005023528178828951 | 0.0020269092391963506 | Ablation no_primary_topic: see metric deltas vs fu |
| no_publication_year | publication_year | 0.7717659193439886 | 0.4756121746656253 | 0.4671714165440256 | 0.0016836366463560681 | 0.0029059896661997042 | -0.0023222357692260776 | Ablation no_publication_year: see metric deltas vs |
| no_venue_name | venue_name | 0.7737663414552991 | 0.48034179198787097 | 0.46440564666573325 | 0.0036840587576665307 | 0.007635606988445398 | -0.005088005647518434 | Ablation no_venue_name: see metric deltas vs full  |
| numeric_only | type, language, venue_name, primary_topi | 0.7886199848306823 | 0.5037067836279182 | 0.44364514979655634 | 0.018537702133049727 | 0.031000598628492637 | -0.02584850251669535 | Ablation numeric_only: see metric deltas vs full m |