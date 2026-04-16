"""Task adapter for the MCDR (3-AFC rats) task."""
from __future__ import annotations

from functools import lru_cache
import types
from typing import List, Tuple, Dict, Any

import jax.numpy as jnp
import numpy as np
import polars as pl

from glmhmmt.runtime import get_data_dir
from glmhmmt.tasks import TaskAdapter, _register, resolve_plots_module
from glmhmmt.tasks.fitted_regressors import (
    FittedWeightRegressorSpec,
    weighted_sum_regressor,
)

_BIAS_HOT_COL_PREFIX = "bias_"
_CHOICE_LAG_COL_PREFIX = "choice_lag_"
_CHOICE_LAG_SIDES = ("L", "C", "R")
_CHOICE_SIDE_TO_CLASS = {"L": 0, "C": 1, "R": 2}
_NUM_CHOICE_LAGS = 15
_STIM_PARAM_MODEL_ID = "One-hot"
_RAW_PARAM_MODEL_ID = "one hot sessions lags"
_STIM_HOT_COLS = tuple(
    f"stim{stim_idx}{side}"
    for stim_idx in range(1, 5)
    for side in _CHOICE_LAG_SIDES
)

EMISSION_COLS: list[str] = [
    "bias",
    "bias_param",
    "biasL", "biasC", "biasR", "onsetL", "onsetC", "onsetR", "delay",
    "SL", "SC", "SR",
    "SLxdelay", "SCxdelay", "SRxdelay",
    "SLxD", "SCxD", "SRxD",
    "D", "DL", "DC", "DR",
    "A_L", "A_C", "A_R",
    "choice_lag_param",
    "stim_param",
    "speed1", "speed2", "speed3",
    "stim1L", "stim1C", "stim1R",
    "stim2L", "stim2C", "stim2R",
    "stim3L", "stim3C", "stim3R",
    "stim4L", "stim4C", "stim4R",
]

TRANSITION_COLS: list[str] = ["A_plus", "A_minus", "A_L", "A_C", "A_R"]

_BIAS_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="bias_param",
    fit_task="MCDR",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_BIAS_HOT_COL_PREFIX,),
    class_idx=0,
)
_CHOICE_LAG_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="choice_lag_param",
    fit_task="MCDR",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_CHOICE_LAG_COL_PREFIX,),
    class_idx=0,
)
_STIM_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="stim_param",
    fit_task="MCDR",
    fit_model_kind="glm",
    fit_model_id=_STIM_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_features=_STIM_HOT_COLS,
    class_idx=0,
)


def _safe_weighted_sum_regressor(
    part,
    spec: FittedWeightRegressorSpec,
) -> np.ndarray | None:
    try:
        return weighted_sum_regressor(part, spec, dtype=np.float32)
    except (FileNotFoundError, ValueError):
        return None


def _bias_hot_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_BIAS_HOT_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _choice_lag_sort_key(name: str) -> tuple[int, int, str]:
    suffix = name.removeprefix(_CHOICE_LAG_COL_PREFIX)
    if len(suffix) >= 2 and suffix[:-1].isdigit() and suffix[-1] in _CHOICE_LAG_SIDES:
        return (int(suffix[:-1]), _CHOICE_LAG_SIDES.index(suffix[-1]), name)
    return (10**9, 10**9, name)


def _stim_hot_sort_key(name: str) -> tuple[int, int, str]:
    if name.startswith("stim") and len(name) >= 6 and name[4].isdigit() and name[-1] in _CHOICE_LAG_SIDES:
        return (int(name[4]), _CHOICE_LAG_SIDES.index(name[-1]), name)
    return (10**9, 10**9, name)


def _bias_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_BIAS_HOT_COL_PREFIX)
            and col.removeprefix(_BIAS_HOT_COL_PREFIX).isdigit()
        ],
        key=_bias_hot_sort_key,
    )


def _choice_lag_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_CHOICE_LAG_COL_PREFIX)
            and len(col.removeprefix(_CHOICE_LAG_COL_PREFIX)) >= 2
            and col.removeprefix(_CHOICE_LAG_COL_PREFIX)[:-1].isdigit()
            and col.removeprefix(_CHOICE_LAG_COL_PREFIX)[-1] in _CHOICE_LAG_SIDES
        ],
        key=_choice_lag_sort_key,
    )


def _stim_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [col for col in columns if col in _STIM_HOT_COLS],
        key=_stim_hot_sort_key,
    )


def _choice_lag_names() -> list[str]:
    return [
        f"{_CHOICE_LAG_COL_PREFIX}{lag_idx:02d}{side}"
        for lag_idx in range(1, _NUM_CHOICE_LAGS + 1)
        for side in _CHOICE_LAG_SIDES
    ]


def _max_sessions_from_df(df: pl.DataFrame) -> int:
    if "subject" not in df.columns or "session" not in df.columns:
        return _max_subject_sessions()
    return int(
        df.group_by("subject")
        .agg(pl.col("session").n_unique().alias("n_sessions"))
        .select(pl.col("n_sessions").max())
        .item()
        or 0
    )


def _infer_bias_hot_cols_from_df(df: pl.DataFrame) -> list[str]:
    existing = _bias_hot_cols(list(df.columns))
    if existing:
        return existing
    max_sessions = _max_sessions_from_df(df)
    return [f"{_BIAS_HOT_COL_PREFIX}{idx}" for idx in range(max_sessions)]


@lru_cache(maxsize=1)
def _max_subject_sessions() -> int:
    dataset_path = get_data_dir() / "df_filtered.parquet"
    df = pl.read_parquet(dataset_path)
    df = df.filter(pl.col("subject") != "A84")
    return int(
        df.group_by("subject")
        .agg(pl.col("session").n_unique().alias("n_sessions"))
        .select(pl.col("n_sessions").max())
        .item()
        or 0
    )


@_register(["mcdr"])
class MCDRAdapter(TaskAdapter):
    """Adapter for the 3-AFC MCDR rat data."""

    task_key: str    = "MCDR"
    task_label: str  = "MCDR"
    num_classes: int = 3
    data_file: str   = "df_filtered.parquet"
    sort_col: str    = "trial_idx"
    session_col: str = "session"

    # ── state-scoring options ────────────────────────────────────────────────
    # Each entry maps a label to a list of (feature_name, class_idx) pairs;
    # class_idx 0 = Left weight, 1 = Right weight.
    # The score for each state k is the mean of W[k, cls, feat_idx] over the
    # listed pairs.  States are then ranked highest-score → "Engaged".
    _SCORING_OPTIONS: dict = {
        "S_coh":     [("SL", 0), ("SR", 1)],
        "S1_coh":   [("stim1L", 0), ("stim1R", 1)],
        "S2_coh":   [("stim2L", 0), ("stim2R", 1)],
        "S3_coh":   [("stim3L", 0), ("stim3R", 1)],
        "S4_coh":   [("stim4L", 0), ("stim4R", 1)],
        "onset_coh": [("onsetL", 0), ("onsetR", 1)],
        "bias_coh":  [("biasL", 0), ("biasR", 1)],
    }
    scoring_key: str = "S_coh"

    # ── data preparation ────────────────────────────────────────────────────

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("subject") != "A84")

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        """Return the MCDR trial dataframe with all derived regressors."""
        df_sub = df_sub.sort(self.sort_col)
        session_order = df_sub["session"].unique(maintain_order=True).to_list()
        session_to_idx = {session_name: idx for idx, session_name in enumerate(session_order)}
        max_sessions = _max_subject_sessions()
        session_idx_expr = pl.col("session").replace_strict(session_to_idx).cast(pl.Int32)
        df_sub = df_sub.with_columns(
            [
                session_idx_expr.alias("_session_idx"),
                ((pl.col("stimd_n") - pl.col("stimd_n").mean()) / pl.col("stimd_n").std()).alias("stimd_n_z"),
            ]
        )
        lag_exprs: list[pl.Expr] = []
        for lag_idx in range(1, _NUM_CHOICE_LAGS + 1):
            lagged_response = pl.col("response").shift(lag_idx).over(self.session_col)
            for side, class_idx in _CHOICE_SIDE_TO_CLASS.items():
                lag_exprs.append(
                    lagged_response.eq(class_idx).fill_null(False).cast(pl.Float32).alias(
                        f"{_CHOICE_LAG_COL_PREFIX}{lag_idx:02d}{side}"
                    )
                )
        session_bias_exprs = [
            pl.col("_session_idx").eq(idx).cast(pl.Float32).alias(f"{_BIAS_HOT_COL_PREFIX}{idx}")
            for idx in range(max_sessions)
        ]
        df_sub = df_sub.with_columns(
            [
                pl.col("response").cast(pl.Int32),
                (pl.col("x_c") == "L").cast(pl.Float32).alias("biasL"),
                (pl.col("x_c") == "C").cast(pl.Float32).alias("biasC"),
                (pl.col("x_c") == "R").cast(pl.Float32).alias("biasR"),
                pl.lit(1.0).cast(pl.Float32).alias("bias"),
                pl.col("delay_d").cast(pl.Float32).alias("delay"),
                ((pl.col("x_c") == "L") * pl.col("onset")).cast(pl.Float32).alias("onsetL"),
                ((pl.col("x_c") == "C") * pl.col("onset")).cast(pl.Float32).alias("onsetC"),
                ((pl.col("x_c") == "R") * pl.col("onset")).cast(pl.Float32).alias("onsetR"),
                ((pl.col("x_c") == "L") * pl.col("stimd_n_z")).cast(pl.Float32).alias("SL"),
                ((pl.col("x_c") == "C") * pl.col("stimd_n_z")).cast(pl.Float32).alias("SC"),
                ((pl.col("x_c") == "R") * pl.col("stimd_n_z")).cast(pl.Float32).alias("SR"),
                ((pl.col("x_c") == "L") * pl.col("delay_d")).cast(pl.Float32).alias("DL"),
                ((pl.col("x_c") == "C") * pl.col("delay_d")).cast(pl.Float32).alias("DC"),
                ((pl.col("x_c") == "R") * pl.col("delay_d")).cast(pl.Float32).alias("DR"),
                pl.col("ttype_n").cast(pl.Float32).alias("D"),
                ((pl.col("x_c") == "L") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SLxD"),
                ((pl.col("x_c") == "C") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SCxD"),
                ((pl.col("x_c") == "R") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SRxD"),
                ((pl.col("x_c") == "L") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SLxdelay"),
                ((pl.col("x_c") == "C") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SCxdelay"),
                ((pl.col("x_c") == "R") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SRxdelay"),
                (
                    (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0)) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "L")
                ).cast(pl.Float32).alias("stim1L"),
                (
                    (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0)) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "C")
                ).cast(pl.Float32).alias("stim1C"),
                (
                    (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0)) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "R")
                ).cast(pl.Float32).alias("stim1R"),
                (
                    (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "L")
                ).cast(pl.Float32).alias("stim2L"),
                (
                    (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "C")
                ).cast(pl.Float32).alias("stim2C"),
                (
                    (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "R")
                ).cast(pl.Float32).alias("stim2R"),
                (
                    (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "L")
                ).cast(pl.Float32).alias("stim3L"),
                (
                    (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "C")
                ).cast(pl.Float32).alias("stim3C"),
                (
                    (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "R")
                ).cast(pl.Float32).alias("stim3R"),
                ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "L")).cast(pl.Float32).alias("stim4L"),
                ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "C")).cast(pl.Float32).alias("stim4C"),
                ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "R")).cast(pl.Float32).alias("stim4R"),
                pl.col("performance").shift(1).fill_null(0).cast(pl.Float32).over(self.session_col).alias("previous_outcome"),
                pl.col("response").shift(1).fill_null(0.0).eq(0).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_L"),
                pl.col("response").shift(1).fill_null(0.0).eq(1).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_C"),
                pl.col("response").shift(1).fill_null(0.0).eq(2).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_R"),
                (1 / (pl.col("timepoint_3") - pl.col("timepoint_4"))).cast(pl.Float32).alias("speed3"),
                (1 / (pl.col("timepoint_3") - pl.col("timepoint_2"))).cast(pl.Float32).alias("speed2"),
                (1 / (pl.col("timepoint_2") - pl.col("timepoint_1"))).cast(pl.Float32).alias("speed1"),
                *session_bias_exprs,
                *lag_exprs,
            ]
        )
        df_sub = df_sub.with_columns(
            [
                pl.col("previous_outcome").ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_plus"),
                (1.0 - pl.col("previous_outcome")).ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_minus"),
            ]
        )
        df_sub = df_sub.with_columns(
            [
                ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).cast(pl.Float32).alias(c)
                for c in ["speed1", "speed2", "speed3"]
            ]
        ).drop("_session_idx")
        bias_param = _safe_weighted_sum_regressor(df_sub, _BIAS_PARAM_SPEC)
        choice_lag_param = _safe_weighted_sum_regressor(df_sub, _CHOICE_LAG_PARAM_SPEC)
        stim_param = _safe_weighted_sum_regressor(df_sub, _STIM_PARAM_SPEC)
        return df_sub.with_columns(
            [
                (
                    pl.Series("bias_param", bias_param)
                    if bias_param is not None
                    else pl.lit(0.0).cast(pl.Float32).alias("bias_param")
                ),
                (
                    pl.Series("choice_lag_param", choice_lag_param)
                    if choice_lag_param is not None
                    else pl.lit(0.0).cast(pl.Float32).alias("choice_lag_param")
                ),
                (
                    pl.Series("stim_param", stim_param)
                    if stim_param is not None
                    else pl.lit(0.0).cast(pl.Float32).alias("stim_param")
                ),
            ]
        )

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` from the MCDR feature dataframe."""
        feature_df = self.build_feature_df(df_sub, tau=tau)
        return self.build_design_matrices(
            feature_df,
            emission_cols=emission_cols,
            transition_cols=transition_cols,
        )

    def build_design_matrices(
        self,
        feature_df,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` from the MCDR feature dataframe."""
        ecols = emission_cols if emission_cols is not None else list(EMISSION_COLS)
        ucols = transition_cols if transition_cols is not None else list(TRANSITION_COLS)
        allowed_ecols = set(self.available_emission_cols(feature_df))
        bad_e = [c for c in ecols if c not in allowed_ecols]
        bad_u = [c for c in ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}")

        y = jnp.asarray(feature_df["response"].to_numpy().astype(np.int32))
        X = jnp.asarray(feature_df.select(ecols).to_numpy().astype(np.float32)) if ecols else jnp.empty((len(y), 0), dtype=jnp.float32)
        U = jnp.asarray(feature_df.select(ucols).to_numpy().astype(np.float32)) if ucols else jnp.empty((len(y), 0), dtype=jnp.float32)
        names = {"X_cols": list(ecols), "U_cols": list(ucols)}
        return y, X, U, names

    # ── column defaults ─────────────────────────────────────────────────────

    def default_emission_cols(self, df=None) -> List[str]:
        del df
        return list(EMISSION_COLS)

    def default_transition_cols(self) -> List[str]:
        return list(TRANSITION_COLS)

    def available_emission_cols(self, df=None) -> List[str]:
        available_cols = list(EMISSION_COLS)
        if df is not None:
            available_cols.extend(self.bias_hot_cols(df))
            available_cols.extend(self.choice_lag_cols(df))
            available_cols.extend(self.stim_hot_cols(df))
        return list(dict.fromkeys(available_cols))

    def resolve_design_names(
        self,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
        df=None,
    ) -> Dict[str, List[str]]:
        ecols = list(emission_cols) if emission_cols is not None else list(EMISSION_COLS)
        ucols = list(transition_cols) if transition_cols is not None else list(TRANSITION_COLS)
        allowed_ecols = set(self.available_emission_cols(df))
        bad_e = [c for c in ecols if c not in allowed_ecols]
        bad_u = [c for c in ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}")
        return {"X_cols": ecols, "U_cols": ucols}

    def bias_hot_cols(self, df: pl.DataFrame) -> List[str]:
        """Return session one-hot bias columns."""
        return _infer_bias_hot_cols_from_df(df)

    def choice_lag_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        """Return explicit previous-choice lag columns."""
        if df is not None:
            existing = _choice_lag_cols(list(df.columns))
            if existing:
                return existing
        return _choice_lag_names()

    def stim_hot_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        """Return stimulus-window one-hot columns."""
        if df is None:
            return list(_STIM_HOT_COLS)
        existing = _stim_hot_cols(list(df.columns))
        return existing if existing else list(_STIM_HOT_COLS)

    @property
    def choice_labels(self) -> list[str]:
        return ["Left", "Center", "Right"]

    @property
    def probability_columns(self) -> list[str]:
        return ["pL", "pC", "pR"]

    def get_correct_class(self, df: pl.DataFrame) -> np.ndarray:
        if "stimulus" in df.columns:
            vals = df["stimulus"].to_numpy().astype(int)
            unique = set(np.unique(vals).tolist())
            if unique.issubset({0, 1, 2}):
                return vals.astype(int)
            if unique.issubset({1, 2, 3}):
                return (vals - 1).astype(int)

        if "x_c" in df.columns:
            vals = df["x_c"].to_numpy()
            mapping = {"L": 0, "C": 1, "R": 2}
            out = np.array([mapping.get(str(v), -1) for v in vals], dtype=int)
            if np.any(out < 0):
                bad = sorted({str(v) for v, idx in zip(vals, out) if idx < 0})
                raise ValueError(f"Unexpected MCDR x_c values: {bad}")
            return out

        raise ValueError(
            "Could not derive MCDR correct class. Expected stimulus coded as "
            "0/1/2 or 1/2/3, or x_c coded as L/C/R."
        )
    # ── column mapping ───────────────────────────────────────────────────────

    @property
    def behavioral_cols(self) -> dict:
        """MCDR column mapping (canonical → actual)."""
        return {
            "trial_idx":   "trial_idx",
            "trial":       "trial",
            "session":     "session",
            "stimulus":    "stimulus",
            "response":    "response",
            "performance": "performance",
        }

    # ── plots ────────────────────────────────────────────────────────────────

    def get_plots(self) -> types.ModuleType:
        return resolve_plots_module(
            adapter_module_name=__name__,
            task_key=self.task_key,
        )
    # ── state labelling ─────────────────────────────────────────────────────

    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        """MCDR engagement scoring driven by ``self.scoring_key``.

        The state with the highest mean coherent weight (as defined by the
        selected scoring regressor) is labelled "Engaged"; the rest are
        "Disengaged" (or "Disengaged 1", "Disengaged 2", … for K>2).
        """
        import numpy as np

        pairs = self._SCORING_OPTIONS.get(
            getattr(self, "scoring_key", "S_coh"),
            self._SCORING_OPTIONS["S_coh"],
        )

        def _scoh(W, feat_names):
            name2fi = {n: i for i, n in enumerate(feat_names)}
            scores = np.zeros(W.shape[0])
            n = 0
            for feat, cls in pairs:
                if feat in name2fi:
                    scores += W[:, cls, name2fi[feat]]
                    n += 1
            return scores / max(1, n)

        base_feat = list(names.get("X_cols", []))
        state_labels: dict = {}
        state_order: dict  = {}
        for subj in subjects:
            W = arrays_store[subj].get("emission_weights") if subj in arrays_store else None
            if W is None:
                state_labels[subj] = {k: f"State {k+1}" for k in range(K)}
                state_order[subj]  = list(range(K))
                continue
            W_np    = np.asarray(W)
            feat    = list(arrays_store[subj].get("X_cols", base_feat))
            scores  = _scoh(W_np, feat)
            ranking = list(np.argsort(scores)[::-1])
            engaged_k = int(ranking[0])
            others    = [int(k) for k in ranking[1:]]
            labels: dict = {engaged_k: "Engaged"}

            if K == 2:
                labels[others[0]] = "Disengaged"
                order = [engaged_k] + others

            elif K == 4:
                name2fi = {n: i for i, n in enumerate(feat)}
                sl_fi   = name2fi.get("SL")
                sr_fi   = name2fi.get("SR")

                # Disengaged L: state most driven by SL (left-choice weight)
                if sl_fi is not None:
                    dis_l = others[int(np.argmax(W_np[others, 0, sl_fi]))]
                else:
                    dis_l = others[0]
                remaining = [k for k in others if k != dis_l]

                # Disengaged R: state most driven by SR (right-choice weight)
                if sr_fi is not None:
                    dis_r = remaining[int(np.argmax(W_np[remaining, 1, sr_fi]))]
                else:
                    dis_r = remaining[0]
                dis_c = [k for k in remaining if k != dis_r][0]

                labels[dis_l] = "Disengaged L"
                labels[dis_r] = "Disengaged R"
                labels[dis_c] = "Disengaged C"
                order = [engaged_k, dis_l, dis_r, dis_c]

            else:
                dis = 1
                for k in others:
                    labels[k] = f"Disengaged {dis}"
                    dis += 1
                order = [engaged_k] + others

            state_labels[subj] = labels
            state_order[subj]  = order
        return state_labels, state_order
