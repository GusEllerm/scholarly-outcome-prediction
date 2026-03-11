# Experimental / non-benchmark experiment configs

Configs in this directory are **not** part of the active benchmark suite. They are retained for experimentation or legacy runs only.

- **Tweedie**: Count-aware GLM; removed from the default benchmark because its distributional assumptions are not aligned with the shared target transformation semantics used by ridge/tree models. Configs: `tweedie_representative.yaml`, `tweedie_temporal.yaml`, `tweedie_representative_h2.yaml`, `tweedie_temporal_h2.yaml`. The model remains registered in the codebase; benchmark-analysis excludes it from comparison output.
