"""Task adapter for the 2AFC drug/saline cohort."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from glmhmmt.tasks import _register
from glmhmmt.tasks.fitted_regressors import (
    mean_feature_weights_from_fit,
    resolved_source_features,
    weighted_sum_regressor,
)
from glmhmmt.runtime import get_data_dir

from .two_afc import (
    TwoAFCAdapter,
    _CHOICE_LAG_PARAM_CORRECT_COL,
    _CHOICE_LAG_PARAM_COL,
    _CHOICE_LAG_PARAM_2_COL,
    _KEEP_EXPERIMENTS,
    _STIM_PARAM_COL,
    _at_choice_param_spec,
    _bias_param_spec,
    _choice_lag_param_2_spec,
    _choice_lag_param_correct_spec,
    _choice_lag_param_spec,
    _stim_param_spec,
)


# Simple switch for the existing *_param regressors in this adapter.
# "2AFC_DRUG": use the drug-cohort fitted weights.
# "2AFC": use the pooled mean fitted weights from the base 2AFC fits.
PARAM_WEIGHT_SOURCE = "2AFC_DRUG"
# PARAM_WEIGHT_SOURCE = "2AFC"

if PARAM_WEIGHT_SOURCE not in {"2AFC_DRUG", "2AFC"}:
    raise ValueError("PARAM_WEIGHT_SOURCE must be '2AFC_DRUG' or '2AFC'.")

_USE_POOLED_BASE_2AFC_PARAM_WEIGHTS = PARAM_WEIGHT_SOURCE == "2AFC"
_PARAM_FIT_TASK = "2AFC" if _USE_POOLED_BASE_2AFC_PARAM_WEIGHTS else "2AFC_DRUG"
_STIM_PARAM_SALINE_COL = "stim_param_saline"
_STIM_PARAM_DRUG_COL = "stim_param_drug"
_CONDITION_STIM_PARAM_COLS = (_STIM_PARAM_SALINE_COL, _STIM_PARAM_DRUG_COL)

_DRUG_INTERACTION_SOURCES: tuple[str, ...] = (
    "cumulative_reward",
    "filtered_choice",
    "filtered_reward",
    "filtered_stim_side",
    "trial_index",
)
_DRUG_EMISSION_INTERACTION_SOURCES: tuple[str, ...] = (
    "stim_param",
    "choice_lag_param",
)
DRUG_INTERACTION_COLS: list[str] = [
    f"drug_x_{source_col}" for source_col in _DRUG_INTERACTION_SOURCES
]
DRUG_EMISSION_INTERACTION_COLS: list[str] = [
    f"drug_x_{source_col}" for source_col in _DRUG_EMISSION_INTERACTION_SOURCES
]
EMISSION_COLS: list[str] = [
    *TwoAFCAdapter.emission_cols,
    *_CONDITION_STIM_PARAM_COLS,
    "Drug",
    *DRUG_EMISSION_INTERACTION_COLS,
]
TRANSITION_COLS: list[str] = [*TwoAFCAdapter.transition_cols, "Drug", *DRUG_INTERACTION_COLS]

_STIM_PARAM_SPEC = _stim_param_spec(fit_task=_PARAM_FIT_TASK)
_BIAS_PARAM_SPEC = _bias_param_spec(fit_task=_PARAM_FIT_TASK)
_AT_CHOICE_PARAM_SPEC = _at_choice_param_spec(fit_task=_PARAM_FIT_TASK)
_CHOICE_LAG_PARAM_SPEC = _choice_lag_param_spec(fit_task=_PARAM_FIT_TASK)
_CHOICE_LAG_PARAM_2_SPEC = _choice_lag_param_2_spec(fit_task=_PARAM_FIT_TASK)
_CHOICE_LAG_PARAM_CORRECT_SPEC = _choice_lag_param_correct_spec(fit_task=_PARAM_FIT_TASK)
_STIM_PARAM_SALINE_SPEC = _stim_param_spec(
    target_name=_STIM_PARAM_SALINE_COL,
    fit_task="2AFC_DRUG",
    fit_model_id="one hot saline",
)
_STIM_PARAM_DRUG_SPEC = _stim_param_spec(
    target_name=_STIM_PARAM_DRUG_COL,
    fit_task="2AFC_DRUG",
    fit_model_id="one hot drug",
)


def _weighted_sum_regressor_zero_fill(
    part: pd.DataFrame,
    spec,
    *,
    pooled_mean: bool = False,
) -> np.ndarray:
    """Project onto fitted weights, treating absent fitted columns as zero."""
    source_features = resolved_source_features(spec)
    missing_cols = [col for col in source_features if col not in part.columns]
    if missing_cols:
        part = part.copy()
        for col in missing_cols:
            part[col] = np.float32(0.0)

    if not pooled_mean:
        return weighted_sum_regressor(part, spec, dtype=np.float32)

    weights = mean_feature_weights_from_fit(spec)
    out = np.zeros(len(part), dtype=np.float32)
    for feat in source_features:
        out += np.asarray(part[feat], dtype=np.float32) * np.float32(weights[feat])
    return out


def _normalize_condition_stim_params(
    feature_df: pl.DataFrame | pd.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    """Normalize condition-specific stimulus params by their column maxima."""
    present_cols = [
        col for col in _CONDITION_STIM_PARAM_COLS if col in feature_df.columns
    ]
    if not present_cols:
        return feature_df

    if isinstance(feature_df, pl.DataFrame):
        exprs = []
        for col in present_cols:
            max_value = feature_df.select(
                pl.col(col).cast(pl.Float32, strict=False).max()
            ).item()
            if max_value is None or not np.isfinite(float(max_value)) or float(max_value) == 0.0:
                continue
            exprs.append(
                (pl.col(col).cast(pl.Float32, strict=False) / float(max_value)).alias(col)
            )
        return feature_df.with_columns(exprs) if exprs else feature_df

    df_pd = feature_df.copy()
    for col in present_cols:
        values = pd.to_numeric(df_pd[col], errors="coerce")
        max_value = float(values.max())
        if not np.isfinite(max_value) or max_value == 0.0:
            continue
        df_pd[col] = (values / max_value).astype("float32")
    return df_pd


def _add_drug_interactions(feature_df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame | pd.DataFrame:
    if "Drug" not in feature_df.columns:
        return feature_df

    if isinstance(feature_df, pl.DataFrame):
        drug_expr = pl.col("Drug").cast(pl.Float32, strict=False).fill_null(0.0)
        interaction_exprs = [
            (
                drug_expr
                * pl.col(source_col).cast(pl.Float32, strict=False).fill_null(0.0)
            ).alias(f"drug_x_{source_col}")
            for source_col in (*_DRUG_INTERACTION_SOURCES, *_DRUG_EMISSION_INTERACTION_SOURCES)
            if source_col in feature_df.columns
        ]
        exprs = [drug_expr.alias("Drug"), *interaction_exprs]
        return feature_df.with_columns(exprs) if exprs else feature_df

    df_pd = feature_df.copy()
    drug = pd.to_numeric(df_pd["Drug"], errors="coerce").fillna(0.0)
    df_pd["Drug"] = drug.astype("float32")
    for source_col in (*_DRUG_INTERACTION_SOURCES, *_DRUG_EMISSION_INTERACTION_SOURCES):
        if source_col in df_pd.columns:
            df_pd[f"drug_x_{source_col}"] = (
                drug * pd.to_numeric(df_pd[source_col], errors="coerce").fillna(0.0)
            ).astype("float32")
    return df_pd


@_register(["two_afc_drug", "2afc_drug", "2AFC_DRUG"])
class TwoAFCDrugAdapter(TwoAFCAdapter):
    """Adapter for the 2AFC drug/saline cohort."""

    task_key: str = "2AFC_DRUG"
    task_label: str = "2AFC Drug"
    data_file: str = "df_alexis_drug_combined.parquet"
    emission_cols: list[str] = EMISSION_COLS
    transition_cols: list[str] = TRANSITION_COLS
    stim_param_spec = _STIM_PARAM_SPEC
    bias_param_spec = _BIAS_PARAM_SPEC
    at_choice_param_spec = _AT_CHOICE_PARAM_SPEC
    choice_lag_param_spec = _CHOICE_LAG_PARAM_SPEC
    choice_lag_param_2_spec = _CHOICE_LAG_PARAM_2_SPEC
    choice_lag_param_correct_spec = _CHOICE_LAG_PARAM_CORRECT_SPEC
    stim_param_saline_spec = _STIM_PARAM_SALINE_SPEC
    stim_param_drug_spec = _STIM_PARAM_DRUG_SPEC

    def read_dataset(self) -> pl.DataFrame:
        """Return all Alexis 2AFC batches with a unified ``Drug`` column.

        Older 2AFC batches do not have drug/saline annotations, so their
        ``Drug`` values stay null and are treated as rest. Batch 6 keeps the
        original 0/1 coding and gets a readable ``condition`` label for plotting.
        """
        data_dir = get_data_dir()
        base = pl.read_parquet(data_dir / "alexis_combined.parquet")
        drug = pl.read_parquet(data_dir / self.data_file)

        if "Drug" not in base.columns:
            base = base.with_columns(pl.lit(None, dtype=pl.Float64).alias("Drug"))
        else:
            base = base.with_columns(pl.col("Drug").cast(pl.Float64, strict=False))
        drug = drug.with_columns(pl.col("Drug").cast(pl.Float64, strict=False))

        df = pl.concat([base, drug], how="diagonal_relaxed")
        return df.with_columns(
            pl.when(pl.col("Drug").is_null())
            .then(pl.lit("rest"))
            .when(pl.col("Drug") == 1)
            .then(pl.lit("drug"))
            .when(pl.col("Drug") == 0)
            .then(pl.lit("saline"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("condition")
        )

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        if "Experiment" not in df.columns:
            return df
        return df.filter(pl.col("Experiment").is_in(_KEEP_EXPERIMENTS))

    def condition_filter_options(self) -> list[str]:
        return ["all", "nan", "rest", "drug", "saline"]

    def drug_condition_col(self, df: pl.DataFrame | pd.DataFrame) -> str | None:
        for col in ("Drug", "drug"):
            if col in df.columns:
                return col
        return None

    def filter_condition_df(
        self,
        df: pl.DataFrame | pd.DataFrame,
        condition_filter: str = "all",
    ) -> pl.DataFrame | pd.DataFrame:
        selected = str(condition_filter or "all").strip().lower()
        if selected in {"all", ""}:
            return df
        if selected not in {"nan", "null", "none", "no_drug", "saline", "rest", "drug"}:
            raise ValueError(
                f"Unknown 2AFC drug condition filter {condition_filter!r}. "
                "Expected one of: all, nan, rest, saline, drug."
            )

        drug_col = self.drug_condition_col(df)
        if drug_col is None:
            raise ValueError("2AFC_DRUG requires a 'Drug' or 'drug' column for condition filtering.")

        if selected in {"nan", "null", "none", "no_drug", "rest"}:
            if isinstance(df, pl.DataFrame):
                return df.filter(pl.col(drug_col).is_null())
            return df.loc[pd.to_numeric(df[drug_col], errors="coerce").isna()].copy()

        target = 1 if selected == "drug" else 0
        if isinstance(df, pl.DataFrame):
            filter_col = "__drug_condition_filter"
            return (
                df.with_columns(
                    pl.col(drug_col)
                    .cast(pl.Int64, strict=False)
                    .alias(filter_col)
                )
                .filter(pl.col(filter_col) == target)
                .drop(filter_col)
            )

        df_pd = df.copy()
        values = pd.to_numeric(df_pd[drug_col], errors="coerce")
        return df_pd.loc[values == target].copy()

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        feature_df = super().build_feature_df(df_sub, tau=tau)
        feature_df = _normalize_condition_stim_params(feature_df)
        return _add_drug_interactions(feature_df)

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols=None,
        transition_cols=None,
    ):
        feature_df = self.build_feature_df(df_sub, tau=tau)
        return self.build_design_matrices(
            feature_df,
            emission_cols=emission_cols,
            transition_cols=transition_cols,
        )

    def default_emission_cols(self, df: pl.DataFrame | None = None) -> list[str]:
        default_cols = super().default_emission_cols(df)
        return [col for col in default_cols if col not in _CONDITION_STIM_PARAM_COLS]

    def available_emission_cols(self, df: pl.DataFrame | None = None) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    *super().available_emission_cols(df),
                    *_CONDITION_STIM_PARAM_COLS,
                ]
            )
        )

    def build_emission_groups(self, available_cols: list[str]) -> list[dict]:
        groups = [
            group
            for group in super().build_emission_groups(available_cols)
            if not any(
                member in _CONDITION_STIM_PARAM_COLS
                for member in group.get("members", {}).values()
            )
        ]
        available = set(available_cols)
        extra_groups = [
            {
                "key": _STIM_PARAM_SALINE_COL,
                "label": "stim param saline",
                "members": {"N": _STIM_PARAM_SALINE_COL},
            },
            {
                "key": _STIM_PARAM_DRUG_COL,
                "label": "stim param drug",
                "members": {"N": _STIM_PARAM_DRUG_COL},
            },
        ]
        groups.extend(
            group
            for group in extra_groups
            if next(iter(group["members"].values())) in available
        )
        return groups

    def build_design_matrices(
        self,
        feature_df,
        emission_cols=None,
        transition_cols=None,
    ):
        requested = (
            list(emission_cols)
            if emission_cols is not None
            else self.default_emission_cols(feature_df)
        )
        include_stim_strength = "stim_strength" in requested or any(
            str(col).startswith("sf_") for col in requested
        )
        include_stim_param = (
            _STIM_PARAM_COL in requested
            or "drug_x_stim_param" in requested
        )
        include_stim_param_saline = _STIM_PARAM_SALINE_COL in requested
        include_stim_param_drug = _STIM_PARAM_DRUG_COL in requested
        include_bias_param = "bias_param" in requested
        include_at_choice_param = "at_choice_param" in requested
        include_choice_lag_param = (
            _CHOICE_LAG_PARAM_COL in requested
            or "drug_x_choice_lag_param" in requested
        )
        include_choice_lag_param_2 = _CHOICE_LAG_PARAM_2_COL in requested
        include_choice_lag_param_correct = _CHOICE_LAG_PARAM_CORRECT_COL in requested

        missing_optional = (
            (include_stim_strength and not any(str(col).startswith("sf_") for col in feature_df.columns))
            or (include_stim_param and _STIM_PARAM_COL not in feature_df.columns)
            or (include_stim_param_saline and _STIM_PARAM_SALINE_COL not in feature_df.columns)
            or (include_stim_param_drug and _STIM_PARAM_DRUG_COL not in feature_df.columns)
            or (include_bias_param and "bias_param" not in feature_df.columns)
            or (include_at_choice_param and "at_choice_param" not in feature_df.columns)
            or (include_choice_lag_param and _CHOICE_LAG_PARAM_COL not in feature_df.columns)
            or (include_choice_lag_param_2 and _CHOICE_LAG_PARAM_2_COL not in feature_df.columns)
            or (
                include_choice_lag_param_correct
                and _CHOICE_LAG_PARAM_CORRECT_COL not in feature_df.columns
            )
        )
        if missing_optional:
            raw_cols = [
                col
                for col in [
                    "subject",
                    "Trial",
                    "Side",
                    "Drug",
                    "Choice",
                    "Hit",
                    "Punish",
                    "Session",
                    "ILD",
                    "Filename",
                    "Experiment",
                    "Task",
                    "P",
                    "AW",
                    "WarmUp",
                    "Date",
                    "condition",
                ]
                if col in feature_df.columns
            ]
            if raw_cols:
                feature_df = feature_df.select(raw_cols)
            feature_df = super()._build_feature_df(
                feature_df,
                include_stim_strength=include_stim_strength,
            )
            if (
                include_stim_param
                or include_stim_param_saline
                or include_stim_param_drug
                or include_bias_param
                or include_at_choice_param
                or include_choice_lag_param
                or include_choice_lag_param_2
                or include_choice_lag_param_correct
            ):
                feature_pd = feature_df.to_pandas()
                for spec in [
                    self.stim_param_spec if include_stim_param else None,
                    self.stim_param_saline_spec if include_stim_param_saline else None,
                    self.stim_param_drug_spec if include_stim_param_drug else None,
                    self.bias_param_spec if include_bias_param else None,
                    self.at_choice_param_spec if include_at_choice_param else None,
                    self.choice_lag_param_spec if include_choice_lag_param else None,
                    self.choice_lag_param_2_spec if include_choice_lag_param_2 else None,
                    self.choice_lag_param_correct_spec if include_choice_lag_param_correct else None,
                ]:
                    if spec is None or spec.target_name in feature_pd.columns:
                        continue
                    feature_pd[spec.target_name] = _weighted_sum_regressor_zero_fill(
                        feature_pd,
                        spec,
                        pooled_mean=(
                            _USE_POOLED_BASE_2AFC_PARAM_WEIGHTS
                            and spec.fit_task == "2AFC"
                        ),
                    )
                feature_df = pl.from_pandas(feature_pd)
        feature_df = _normalize_condition_stim_params(feature_df)
        feature_df = _add_drug_interactions(feature_df)
        return super().build_design_matrices(
            feature_df,
            emission_cols=emission_cols,
            transition_cols=transition_cols,
        )

    def choice_half_life(self, subject: str | None) -> float | None:
        del subject
        return None


__all__ = [
    "EMISSION_COLS",
    "TRANSITION_COLS",
    "DRUG_INTERACTION_COLS",
    "DRUG_EMISSION_INTERACTION_COLS",
    "TwoAFCDrugAdapter",
]
