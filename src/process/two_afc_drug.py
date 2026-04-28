"""Task adapter for the 2AFC drug/saline cohort."""
from __future__ import annotations

import pandas as pd
import polars as pl

from glmhmmt.tasks import _register
from glmhmmt.tasks.fitted_regressors import FittedWeightRegressorSpec

from .two_afc import (
    EMISSION_COLS as BASE_EMISSION_COLS,
    TRANSITION_COLS as BASE_TRANSITION_COLS,
    TwoAFCAdapter,
    _KEEP_EXPERIMENTS,
)


EMISSION_COLS: list[str] = [
    col
    for col in BASE_EMISSION_COLS
    if col not in {"bias_param", "at_choice_param"}
]
TRANSITION_COLS: list[str] = [*BASE_TRANSITION_COLS, "Drug"]
_STIM_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="stim_param",
    fit_task="2AFC",
    fit_model_kind="glm",
    fit_model_id="one hot lapses",
    arrays_suffix="glm_arrays.npz",
    exclude_features=("bias", "stim_0"),
    excluded_subjects=("325", "325.0"),
    sign=1.0,
)


@_register(["two_afc_drug", "2afc_drug", "2AFC_DRUG"])
class TwoAFCDrugAdapter(TwoAFCAdapter):
    """Adapter for the 2AFC drug/saline cohort."""

    task_key: str = "2AFC_DRUG"
    task_label: str = "2AFC Drug"
    data_file: str = "df_alexis_drug_combined.parquet"
    emission_cols: list[str] = EMISSION_COLS
    transition_cols: list[str] = TRANSITION_COLS
    stim_param_spec: FittedWeightRegressorSpec = _STIM_PARAM_SPEC

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        if "Experiment" not in df.columns:
            return df
        return df.filter(pl.col("Experiment").is_in(_KEEP_EXPERIMENTS))

    def condition_filter_options(self) -> list[str]:
        return ["all", "saline", "drug"]

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
        if selected not in {"saline", "drug"}:
            raise ValueError(
                f"Unknown 2AFC drug condition filter {condition_filter!r}. "
                "Expected one of: all, saline, drug."
            )

        drug_col = self.drug_condition_col(df)
        if drug_col is None:
            raise ValueError("2AFC_DRUG requires a 'Drug' or 'drug' column for condition filtering.")

        target = 1 if selected == "drug" else 0
        if isinstance(df, pl.DataFrame):
            filter_col = "__drug_condition_filter"
            return (
                df.with_columns(
                    pl.col(drug_col)
                    .fill_null(0)
                    .cast(pl.Int64, strict=False)
                    .alias(filter_col)
                )
                .filter(pl.col(filter_col) == target)
                .drop(filter_col)
            )

        df_pd = df.copy()
        values = pd.to_numeric(df_pd[drug_col], errors="coerce").fillna(0).astype(int)
        return df_pd.loc[values == target].copy()

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        return self._build_feature_df(
            df_sub,
            tau=tau,
            include_stim_strength=False,
            include_stim_param=True,
            include_bias_param=False,
            include_at_choice_param=False,
        )

    def choice_half_life(self, subject: str | None) -> float | None:
        del subject
        return None

    def bias_hot_cols(self, df: pl.DataFrame) -> list[str]:
        return []

    def choice_lag_cols(self, df: pl.DataFrame | None = None) -> list[str]:
        return []


__all__ = [
    "EMISSION_COLS",
    "TRANSITION_COLS",
    "TwoAFCDrugAdapter",
]
