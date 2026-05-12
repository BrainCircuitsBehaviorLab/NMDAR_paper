"""Task adapter for the Tiffany 2ADC drug/saline cohort."""
from __future__ import annotations

import pandas as pd
import polars as pl

from glmhmmt.tasks import _register

from .two_adc import TRANSITION_COLS as BASE_TRANSITION_COLS, TwoAFCDelayAdapter


TRANSITION_COLS: list[str] = [*BASE_TRANSITION_COLS, "drug_code"]


@_register(["two_afc_delay_drug", "2afc_delay_drug", "2AFC_delay_DRUG", "2adc_drug", "2ADC_DRUG"])
class TwoADCDrugAdapter(TwoAFCDelayAdapter):
    """Adapter for Tiffany 2ADC saline and NR2B sessions."""

    task_key: str = "2ADC_DRUG"
    task_label: str = "2ADC Drug"
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


__all__ = ["TRANSITION_COLS", "TwoADCDrugAdapter"]
