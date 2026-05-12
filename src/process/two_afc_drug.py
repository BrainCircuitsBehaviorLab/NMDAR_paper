"""Task adapter for the 2AFC drug/saline cohort."""
from __future__ import annotations

import pandas as pd
import polars as pl

from glmhmmt.tasks import _register
from glmhmmt.tasks.fitted_regressors import (
    resolved_source_features,
    weighted_sum_regressor,
)
from glmhmmt.runtime import get_data_dir

from .two_afc import (
    EMISSION_COLS as BASE_EMISSION_COLS,
    TRANSITION_COLS as BASE_TRANSITION_COLS,
    TwoAFCAdapter,
    _AT_CHOICE_PARAM_SPEC as BASE_AT_CHOICE_PARAM_SPEC,
    _BIAS_PARAM_SPEC as BASE_BIAS_PARAM_SPEC,
    _KEEP_EXPERIMENTS,
    _SF_COL_PREFIX,
    _STIM_PARAM_COL,
    _STIM_PARAM_SPEC as BASE_STIM_PARAM_SPEC,
)


EMISSION_COLS: list[str] = list(BASE_EMISSION_COLS)
TRANSITION_COLS: list[str] = [*BASE_TRANSITION_COLS, "Drug"]


@_register(["two_afc_drug", "2afc_drug", "2AFC_DRUG"])
class TwoAFCDrugAdapter(TwoAFCAdapter):
    """Adapter for the 2AFC drug/saline cohort."""

    task_key: str = "2AFC_DRUG"
    task_label: str = "2AFC Drug"
    data_file: str = "df_alexis_drug_combined.parquet"
    emission_cols: list[str] = EMISSION_COLS
    transition_cols: list[str] = TRANSITION_COLS
    stim_param_spec = BASE_STIM_PARAM_SPEC
    bias_param_spec = BASE_BIAS_PARAM_SPEC
    at_choice_param_spec = BASE_AT_CHOICE_PARAM_SPEC

    def read_dataset(self) -> pl.DataFrame:
        """Return all Alexis 2AFC batches with a unified ``Drug`` column.

        Older 2AFC batches do not have drug/rest annotations, so their
        ``Drug`` values stay null. Batch 6 keeps the original 0/1 coding and
        gets a readable ``condition`` label for plotting.
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
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("Drug") == 1)
            .then(pl.lit("drug"))
            .when(pl.col("Drug") == 0)
            .then(pl.lit("rest"))
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
                "Expected one of: all, nan, rest, drug."
            )

        drug_col = self.drug_condition_col(df)
        if drug_col is None:
            raise ValueError("2AFC_DRUG requires a 'Drug' or 'drug' column for condition filtering.")

        if selected in {"nan", "null", "none", "no_drug"}:
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
        return self._build_feature_df(
            df_sub,
            tau=tau,
            include_stim_strength=False,
            include_stim_param=False,
            include_bias_param=False,
            include_at_choice_param=False,
        )

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
            str(col).startswith(_SF_COL_PREFIX) for col in requested
        )
        include_stim_param = _STIM_PARAM_COL in requested
        include_bias_param = "bias_param" in requested
        include_at_choice_param = "at_choice_param" in requested

        missing_optional = (
            (include_stim_strength and not any(str(col).startswith(_SF_COL_PREFIX) for col in feature_df.columns))
            or (include_stim_param and _STIM_PARAM_COL not in feature_df.columns)
            or (include_bias_param and "bias_param" not in feature_df.columns)
            or (include_at_choice_param and "at_choice_param" not in feature_df.columns)
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
            feature_df = self._build_feature_df(
                feature_df,
                include_stim_strength=include_stim_strength,
                include_stim_param=include_stim_param,
                include_bias_param=False,
                include_at_choice_param=False,
            )
            if include_bias_param or include_at_choice_param:
                feature_pd = feature_df.to_pandas()
                for spec in [
                    self.bias_param_spec if include_bias_param else None,
                    self.at_choice_param_spec if include_at_choice_param else None,
                ]:
                    if spec is None or spec.target_name in feature_pd.columns:
                        continue
                    for source_col in resolved_source_features(spec):
                        if source_col not in feature_pd.columns:
                            feature_pd[source_col] = 0.0
                    feature_pd[spec.target_name] = weighted_sum_regressor(
                        feature_pd,
                        spec,
                    )
                feature_df = pl.from_pandas(feature_pd)
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
    "TwoAFCDrugAdapter",
]
