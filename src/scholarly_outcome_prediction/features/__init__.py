"""Feature building and preprocessing.

Modality-aware structure:
- build_metadata_features / build_feature_matrix: metadata-only features (current).
- Text and hybrid feature builders are reserved for future use (see build_features.py).
"""

from scholarly_outcome_prediction.features.build_features import (
    build_feature_matrix,
    build_metadata_features,
    get_feature_column_names,
)
from scholarly_outcome_prediction.features.preprocess import build_preprocessor

__all__ = [
    "build_feature_matrix",
    "build_metadata_features",
    "build_preprocessor",
    "get_feature_column_names",
]
