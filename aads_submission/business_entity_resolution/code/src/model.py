import os
import xgboost as xgb
import pandas as pd
import numpy as np

from src.features import FEATURE_COLS


class EntityMatchingModel:
    def __init__(self, use_transformer=False):
        self.model = xgb.XGBClassifier(
            # Many shallow trees (max_depth=6) rather than few deep ones, plus
            # min_child_weight/gamma/reg_alpha/reg_lambda all pulling toward
            # simpler splits and subsample/colsample_bytree<1: the positive
            # (true-match) pair count is comparatively small next to how many
            # candidate pairs exist, so these all guard against overfitting
            # to that limited positive signal rather than tuned for raw
            # accuracy on a large balanced dataset.
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1.0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            # Bias toward precision: penalize false positives more
            scale_pos_weight=0.7,
            tree_method='hist',  # histogram-binned splits -- needed for speed at 10M+ row scale
            eval_metric='logloss',
            early_stopping_rounds=30,
        )
        self.use_transformer = use_transformer
        self.transformer_model = None
        self.is_fitted = False

        if self.use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self.transformer_model = SentenceTransformer('intfloat/multilingual-e5-base')
            except Exception:
                self.transformer_model = None

    def prepare_X(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Select model feature columns, gracefully handling missing columns.
        
        When a pre-trained model is loaded, detects its expected feature set
        and only selects those columns to avoid feature_names mismatch errors.
        """
        # If model is already fitted, check its expected features
        if self.is_fitted and hasattr(self.model, 'get_booster'):
            try:
                booster = self.model.get_booster()
                model_features = booster.feature_names
                if model_features:
                    available = [c for c in model_features if c in df_features.columns]
                    if len(available) == len(model_features):
                        return df_features[available]
                    # If not all model features are available, fall back
            except Exception:
                pass

        available = [c for c in FEATURE_COLS if c in df_features.columns]
        if not available:
            # Legacy fallback for models trained with old feature set
            legacy_cols = [
                'name_lev_ratio', 'name_jw_dist', 'name_token_sort', 'name_token_set',
                'addr_jaccard', 'addr_ngram_sim', 'zip_exact',
                'name_length_ratio', 'name_token_overlap'
            ]
            available = [c for c in legacy_cols if c in df_features.columns]
        if self.use_transformer and 'semantic_sim' in df_features.columns:
            available.append('semantic_sim')
        return df_features[available]

    def fit(self, df_features: pd.DataFrame, y: pd.Series,
            df_val: pd.DataFrame = None, y_val: pd.Series = None):
        X = self.prepare_X(df_features)
        if df_val is not None and y_val is not None:
            X_val = self.prepare_X(df_val)
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=50)
        else:
            # Without validation, disable early stopping
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        if df_features.empty:
            return np.array([], dtype=np.float32)

        X = self.prepare_X(df_features)
        if self.is_fitted:
            if isinstance(self.model, xgb.Booster):
                return self.model.predict(xgb.DMatrix(X)).astype(np.float32)
            else:
                return self.model.predict_proba(X)[:, 1].astype(np.float32)
        else:
            return self._heuristic_score(X)

    def _heuristic_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        High-precision composite heuristic for scoring entity pairs.
        Designed to produce well-calibrated scores for threshold-based matching.
        """
        cols = X.columns.tolist()

        # Name score (strongest signal)
        if 'stripped_jw' in cols:
            jw = X['stripped_jw'].values / 100.0
        elif 'name_jw_dist' in cols:
            jw = X['name_jw_dist'].values / 100.0
        else:
            jw = np.zeros(len(X), dtype=np.float32)

        if 'stripped_tsr' in cols:
            tsr = X['stripped_tsr'].values / 100.0
        elif 'name_token_sort' in cols:
            tsr = X['name_token_sort'].values / 100.0
        else:
            tsr = np.zeros(len(X), dtype=np.float32)

        ts = X['name_token_set'].values / 100.0 if 'name_token_set' in cols else np.zeros(len(X))
        pr = X['name_partial_ratio'].values / 100.0 if 'name_partial_ratio' in cols else np.zeros(len(X))

        # Address score (supporting signal)
        aj = X['addr_jaccard'].values / 100.0 if 'addr_jaccard' in cols else np.zeros(len(X))
        ze = X['zip_exact'].values if 'zip_exact' in cols else np.zeros(len(X))
        no = X['nums_overlap'].values if 'nums_overlap' in cols else np.zeros(len(X))

        # Exact match boost
        se = X['stripped_exact'].values if 'stripped_exact' in cols else np.zeros(len(X))
        ftm = X['first_token_match'].values if 'first_token_match' in cols else np.zeros(len(X))

        # Composite score emphasizing name precision
        name_score = 0.30 * jw + 0.25 * tsr + 0.20 * ts + 0.10 * pr + 0.15 * np.maximum(jw, tsr)
        addr_score = 0.50 * aj + 0.30 * no + 0.20 * ze
        bonus = 0.10 * se + 0.05 * ftm

        score = 0.65 * name_score + 0.25 * addr_score + 0.10 * bonus

        return score.astype(np.float32)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> bool:
        if os.path.exists(path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(path)
            self.is_fitted = True
            return True
        return False
