import marimo

__generated_with = "0.22.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parsing filtered_df.csv
    """)
    return


@app.cell
def _():
    import pandas as pd
    import polars as pl
    import numpy as np
    from glmhmmt.runtime import get_runtime_paths

    paths = get_runtime_paths()
    return np, paths, pd


@app.cell
def _(paths, pd):
    paths.show_paths()  # Verificar que las rutas se han configurado correctamente
    all_df = pd.read_csv(paths.DATA_PATH/"MCDR_all.csv", sep = ";")
    # df.to_parquet(paths.DATA_PATH/"MCDR_all.parquet")
    all_df
    return (all_df,)


@app.cell
def _(all_df, np, pd):
    # Define manipulation conditions
    df = all_df.copy()
    condition_injection = (df['injection'] != 'Rest') & (df['injection']  != 'Saline')
    condition_opto = df['opto_on'] > 0
    condition_probabilities = (df['batch'] == '9B') & (df['task'].isin(['StageTraining_9B_V3', 'StageTraining_9B_V4', 'StageTraining_9B_V5']))
    condition_probabilities2 = ((df['batch'] == '11B') &
        (~df['bias_prob'].isin(['[0, 0, 0]', '[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]']) |
         pd.isna(df['bias_prob']))
    )

    # Set manipulation based on combined conditions
    df['manipulation'] = np.where(condition_injection | condition_opto | condition_probabilities | condition_probabilities2, 1, 0)
    df = df[df["manipulation"] == 0]
    df['valids']=1

    # NO STAGE TRAINING CODES NO VALIDS
    df.loc[~df['task'].str.contains('StageTraining'), 'valids'] = 0

    # MISSING STATES NO VALID TRIALS
    df.loc[df['STATE_Response_window_START'].isnull(), 'valids'] = 0
    df.loc[df['STATE_Response_window_END'].isnull(),  'valids']= 0
    df.loc[df['STATE_Exit_START'].isnull(),  'valids']= 0

    # # EASY TRIALS NO VALID TRIALS
    df.loc[df['trial']<10, 'valids']=0
    df.loc[((df['trial']<12) & ((df['batch']=='3B') | (df['batch']=='34B') | (df['batch'] == '4B') | (df['batch'] == '5B'))), 'valids'] = 0

    # # REMOVE BPOD ERROR TRIALS
    df.loc[df['BPODCRASH'].notnull(), 'valids'] = 0

    # # REMOVE BADLY CATEGORIZED RESPONSES (OUTLIERS)
    df.loc[((df['r_n'].isnull()) & (df['trial_result']!='miss')) , 'valids'] = 0

    # # REMOVE EMPTY RESPONSE TIME
    df.loc[((df['response_time_first'].isnull()) & (df['trial_result']!='miss')), 'valids'] = 0
    return (df,)


@app.cell
def _(df, pd):
    n_nan_original = df["bias_prob"].isna().sum()
    converted = pd.to_numeric(df["bias_prob"], errors="coerce")

    n_nan_after = converted.isna().sum()
    n_new_nan = n_nan_after - n_nan_original

    print(f"NaN originales: {n_nan_original}")
    print(f"NaN después: {n_nan_after}")
    print(f"Nuevos NaN introducidos: {n_new_nan}")
    mask_new_nan = converted.isna() & df["bias_prob"].notna()

    df.loc[mask_new_nan, "bias_prob"].value_counts()
    return


@app.cell
def _(df):
    df["bias_prob"].value_counts()
    return


@app.cell
def _(df, paths, pd):
    to_plot= df.loc[((df['valids']==1) & (df['manipulation']==0) &(df['training_stage']=='DataCollection') & (df['stim_dur_dl'] == 0) | (df['stim_dur_dl'].isna()))] # Remove weird & 
    to_plot['trial_length'] = pd.to_numeric(to_plot['trial_length'], errors='coerce')
    to_plot['stim_d'] = to_plot['trial_length'] - to_plot['delay_abs']
    to_plot = to_plot[to_plot["misses"] == 0]
    to_plot['delay_d'] = to_plot['delay_abs']
    response_code = {'L': 0, 'C': 1, 'R': 2}
    to_plot['response'] = to_plot['r_c'].map(response_code)
    to_plot['stimulus'] = to_plot['x_c'].map(response_code)
    to_plot.sort_values(['subject', 'session', 'trial', 'date'], inplace=True)
    to_plot['trial_idx'] = to_plot.groupby(['subject']).cumcount()
    to_plot = to_plot[['subject', 'batch', 'trial', 'session', 'date', 'x_c', 'r_c', 'ttype_n', 'stimd_n', 'performance', 'bias_prob']]
    to_plot = to_plot[to_plot["batch"].isin(["3B", "11B"])]
    to_plot.to_parquet(paths.DATA_PATH / "MCDR_all.parquet")
    to_plot.to_csv(paths.DATA_PATH / "MCDR_bad.csv")
    return (to_plot,)


@app.cell
def _(to_plot):
    to_plot.groupby(["batch", "subject"]).size()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
