"""Task adapter for the Tiffany 2ADC drug/saline cohort."""
from __future__ import annotations

import pandas as pd
import polars as pl

from glmhmmt.tasks import _register

from .two_adc import (
    EMISSION_COLS as BASE_EMISSION_COLS,
    TRANSITION_COLS as BASE_TRANSITION_COLS,
    TwoAFCDelayAdapter,
)


_DRUG_INTERACTION_SOURCES: tuple[str, ...] = (
    "cumulative_reward",
    "filtered_choice",
    "filtered_reward",
    "filtered_stim_side",
    "trial_index",
)
_DRUG_EMISSION_INTERACTION_SOURCES: tuple[str, ...] = (
    "stim_x_delay_param",
    "choice_lag_param",
)
DRUG_INTERACTION_COLS: list[str] = [
    f"drug_x_{source_col}" for source_col in _DRUG_INTERACTION_SOURCES
]
DRUG_EMISSION_INTERACTION_COLS: list[str] = [
    f"drug_x_{source_col}" for source_col in _DRUG_EMISSION_INTERACTION_SOURCES
]
EMISSION_COLS: list[str] = [*BASE_EMISSION_COLS, *DRUG_EMISSION_INTERACTION_COLS]
TRANSITION_COLS: list[str] = [*BASE_TRANSITION_COLS, "drug_code", *DRUG_INTERACTION_COLS]


def _add_drug_interactions(feature_df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame | pd.DataFrame:
    if "drug_code" not in feature_df.columns:
        return feature_df

    if isinstance(feature_df, pl.DataFrame):
        drug_expr = pl.col("drug_code").cast(pl.Float32, strict=False).fill_null(0.0)
        interaction_exprs = [
            (
                drug_expr
                * pl.col(source_col).cast(pl.Float32, strict=False).fill_null(0.0)
            ).alias(f"drug_x_{source_col}")
            for source_col in (*_DRUG_INTERACTION_SOURCES, *_DRUG_EMISSION_INTERACTION_SOURCES)
            if source_col in feature_df.columns
        ]
        exprs = [drug_expr.alias("drug_code"), *interaction_exprs]
        return feature_df.with_columns(exprs) if exprs else feature_df

    df_pd = feature_df.copy()
    drug = pd.to_numeric(df_pd["drug_code"], errors="coerce").fillna(0.0)
    df_pd["drug_code"] = drug.astype("float32")
    for source_col in (*_DRUG_INTERACTION_SOURCES, *_DRUG_EMISSION_INTERACTION_SOURCES):
        if source_col in df_pd.columns:
            df_pd[f"drug_x_{source_col}"] = (
                drug * pd.to_numeric(df_pd[source_col], errors="coerce").fillna(0.0)
            ).astype("float32")
    return df_pd


@_register(["two_afc_delay_drug", "2afc_delay_drug", "2AFC_delay_DRUG", "2adc_drug", "2ADC_DRUG"])
class TwoADCDrugAdapter(TwoAFCDelayAdapter):
    """Adapter for Tiffany 2ADC saline and NR2B sessions."""

    task_key: str = "2ADC_DRUG"
    task_label: str = "2ADC Drug"
    emission_cols: list[str] = EMISSION_COLS
    transition_cols: list[str] = TRANSITION_COLS

    def read_dataset(self) -> pl.DataFrame:
        df = super().read_dataset()
        if "drug" not in df.columns:
            return df.with_columns(
                [
                    pl.lit(None, dtype=pl.Utf8).alias("condition"),
                    pl.lit(None, dtype=pl.Float32).alias("drug_code"),
                ]
            )

        drug_clean = pl.col("drug").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
        return df.with_columns(
            [
                pl.when(drug_clean == "rest")
                .then(pl.lit("rest"))
                .when(drug_clean == "saline")
                .then(pl.lit("saline"))
                .when(drug_clean.is_null())
                .then(pl.lit(None, dtype=pl.Utf8))
                .otherwise(pl.lit("drug"))
                .alias("condition"),
                pl.when(drug_clean.is_in(["nr2b", "drug"]))
                .then(pl.lit(1.0))
                .when(drug_clean == "saline")
                .then(pl.lit(0.0))
                .otherwise(pl.lit(None, dtype=pl.Float32))
                .alias("drug_code"),
            ]
        )

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def condition_filter_options(self) -> list[str]:
        return ["all", "nan", "rest", "drug", "saline"]

    def filter_condition_df(
        self,
        df: pl.DataFrame | pd.DataFrame,
        condition_filter: str = "all",
    ) -> pl.DataFrame | pd.DataFrame:
        selected = str(condition_filter or "all").strip().lower()
        if selected in {"all", ""}:
            return df
        if selected in {"null", "none", "no_drug"}:
            selected = "nan"
        if selected not in {"nan", "rest", "saline", "drug"}:
            raise ValueError(
                f"Unknown 2ADC drug condition filter {condition_filter!r}. "
                "Expected one of: all, nan, rest, saline, drug."
            )

        if isinstance(df, pl.DataFrame):
            if "condition" in df.columns:
                condition_expr = pl.col("condition").cast(pl.Utf8).str.to_lowercase()
            elif "drug" in df.columns:
                drug_expr = pl.col("drug").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
                condition_expr = (
                    pl.when(drug_expr == "rest")
                    .then(pl.lit("rest"))
                    .when(drug_expr == "saline")
                    .then(pl.lit("saline"))
                    .when(drug_expr.is_null())
                    .then(pl.lit(None, dtype=pl.Utf8))
                    .otherwise(pl.lit("drug"))
                )
            else:
                raise ValueError("2ADC_DRUG requires a 'condition' or 'drug' column for condition filtering.")
            filter_col = "__condition_filter"
            filtered = df.with_columns(condition_expr.alias(filter_col))
            if selected == "nan":
                return filtered.filter(pl.col(filter_col).is_null()).drop(filter_col)
            return filtered.filter(pl.col(filter_col) == selected).drop(filter_col)

        df_pd = df.copy()
        if "condition" in df_pd.columns:
            condition = df_pd["condition"].astype("string").str.lower()
        elif "drug" in df_pd.columns:
            drug = df_pd["drug"].astype("string").str.strip().str.lower()
            condition = pd.Series(pd.NA, index=df_pd.index, dtype="string")
            condition.loc[drug == "rest"] = "rest"
            condition.loc[drug == "saline"] = "saline"
            condition.loc[drug.notna() & ~drug.isin(["rest", "saline"])] = "drug"
        else:
            raise ValueError("2ADC_DRUG requires a 'condition' or 'drug' column for condition filtering.")

        if selected == "nan":
            return df_pd.loc[condition.isna()].copy()
        return df_pd.loc[condition == selected].copy()

    def choice_half_life(self, subject: str | None) -> float | None:
        del subject
        return None

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        return _add_drug_interactions(super().build_feature_df(df_sub, tau=tau))

    def build_design_matrices(
        self,
        feature_df,
        emission_cols=None,
        transition_cols=None,
    ):
        return super().build_design_matrices(
            _add_drug_interactions(feature_df),
            emission_cols=emission_cols,
            transition_cols=transition_cols,
        )


__all__ = [
    "EMISSION_COLS",
    "TRANSITION_COLS",
    "DRUG_INTERACTION_COLS",
    "DRUG_EMISSION_INTERACTION_COLS",
    "TwoADCDrugAdapter",
]
