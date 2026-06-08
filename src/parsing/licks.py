import pandas as pd
import numpy as np
import ast
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem
from scipy.stats import ttest_1samp
from scipy.stats import median_abs_deviation
from scipy.stats import zscore
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import seaborn as sns


# Define functions to get lick variables: RTs, N licks, ILI
def curate_licks(licks, df_behavior, time_window=1):
    """
    Curate licks for a given behavioral session (remove licks before Response Window opens or after ITI ends).
    Note that timestamps (in seconds) of licks and events are relative to the trial onset (0 s).
    :param licks: Series with a list of licks per trial
    :param df_behavior: DataFrame with the behavioral data of a session
    :param time_window: Time window to consider for licks (in seconds). This will result in N licks = lick rate (Hz)
    :return: Curated Series with l
    :return: Series with curated licks
    """

    premature_lick_trials = []  # List of indices of trials where licks happened before response window
    tolerance = 0.0  # In seconds, tolerance for licks outside the response window (delay of motor Arduino code)

    for trial in range(len(licks)):
        resp_win_start = df_behavior.RespWinStart.iloc[trial]
        resp_win_end = df_behavior.RespWinEnd.iloc[trial]

        # Check if any licks before response window start to detect trials in which the motor was stuck
        # (do not include in the condition licks after response window end, as these depend on the time the lickport was
        # available (which varies depending on trial outcome)
        if any((lick < resp_win_start - tolerance) for lick in licks.iloc[trial]):
            premature_lick_trials.append(trial)

        licks[trial] = [lick for lick in licks.iloc[trial]
                        if resp_win_start - tolerance <= lick <= resp_win_end + time_window + tolerance]

    return licks, premature_lick_trials


def get_peri_event_licks(df_behavior, event='StimStart', time_window=1):
    """
    Get peri-event licks for a given behavioral session
    :param df_behavior: DataFrame with behavior data
    :return: pd.Series with a list of peri-event licks per trial
    """

    # Convert string of lists back lists
    try:
        df_behavior["Port1In"] = df_behavior["Port1In"].apply(ast.literal_eval)
        df_behavior["Port2In"] = df_behavior["Port2In"].apply(ast.literal_eval)
    except ValueError:
        print('Port1In or Port2In are already lists')

    licks_left = df_behavior.Port1In.copy()
    licks_right = df_behavior.Port2In.copy()

    # Curate licks
    licks_left, premature_lick_trials_left = curate_licks(licks_left, df_behavior, time_window)
    licks_right, premature_lick_trials_right = curate_licks(licks_right, df_behavior, time_window)

    for trial in range(len(df_behavior)):

        # Align licks to the event time
        event_time = df_behavior[event].iloc[trial]
        licks_left[trial] = [x - event_time for x in licks_left.iloc[trial]]  # Left
        licks_right[trial] = [x - event_time for x in licks_right.iloc[trial]]  # Right

    licks = [licks_left] + [licks_right]
    premature_lick_trials = [premature_lick_trials_left] + [premature_lick_trials_right]

    return licks, premature_lick_trials


def inter_lick_interval(licks, method='first'):
    """
    Compute the mean inter-lick interval (ILI) of the licks of a behavioral session.
    :return: Inter-lick interval (ILI) of the licks per trial
    """

    ili = []

    for trial in range(len(licks)):
        licks_trial = (licks.iloc[trial])
        licks_trial.sort()  # Sort licks in ascending order
        n_licks = len(licks_trial)
        if n_licks < 2:  # Minimum of 2 licks required to compute ILI
            ili.append(np.nan)
        else:
            intervals = np.diff(licks_trial)
            if method == 'first':
                ili.append(intervals[0])  # First ili (interval between 1st-2nd lick)
            elif method == 'mean':
                ili.append(np.mean(intervals))  # Mean ili
            else:
                raise ValueError("method must be 'first' or 'mean'")

    return ili


def get_rt(df_behavior):
    """
    Compute the reaction time (RT) of a behavioral session from EVENT data.
    :param df_behavior: DataFrame with the behavioral data
    :return: Reaction time (RT) of the licks per trial
    """

    rt = []

    for trial in range(len(df_behavior)):

        if df_behavior.Miss.iloc[trial] == 1:
            rt.append(np.nan)
        else:
            rt.append(df_behavior.RespWinLen.iloc[trial])

    return rt


def get_rt2(licks, df_behavior):
    """
    Compute the reaction time (RT) of a behavioral session from LICK data.
    :param df_behavior: DataFrame with the behavioral data
    :return: Reaction time (RT) of the licks per trial
    """

    rt2 = []

    # Combine left and right licks into a single Series
    licks = pd.Series([left + right for left, right in zip(licks[0], licks[1])])

    for trial in range(len(licks)):

        trial_licks = licks.iloc[trial]
        trial_licks.sort()  # Sort licks in ascending order

        if not trial_licks or df_behavior.Miss.iloc[trial] == 1:  # If no licks in the trial
            rt2.append(np.nan)
        else:
            # Align response window start to stimulus onset
            stim_start = df_behavior.StimStart.iloc[trial]
            resp_win_start = df_behavior.RespWinStart.iloc[trial]
            resp_win_start_aligned = resp_win_start - stim_start

            # Find the first lick within the response window
            first_lick = min(trial_licks)
            first_lick_aligned = first_lick - resp_win_start_aligned

            if first_lick_aligned < 0:  # If the first lick is before the response window
                rt2.append(np.nan)
            else:
                rt2.append(first_lick_aligned)  # Use the first lick as RT

    return rt2


def add_lick_data(df_behavior):
    """
    Main function to compute licks and reaction times for a given behavioral session.
    :param df_behavior: DataFrame with the behavioral data of a session
    :return: DataFrame with additional columns for licks and reaction times
    """

    # Apply lick functions
    licks, premature_lick_trials = get_peri_event_licks(df_behavior, event='StimStart')

    # Combine left and right premature lick trials
    premature_lick_trials = sorted(set(premature_lick_trials[0] + premature_lick_trials[1]))

    # Print % premature lick trials
    try:
        print(f'{len(premature_lick_trials) / len(df_behavior) * 100:.2f}% of premature lick trials (before response window)')
    except ZeroDivisionError:
        print('0% of premature lick trials (before response window)')

    # # Drop premature lick trials from DataFrame
    # df_behavior = df_behavior.drop(index=premature_lick_trials).reset_index(drop=True)

    licks_left = licks[0]
    licks_right = licks[1]
    n_licks_left = [len(lick) for lick in licks_left]
    n_licks_right = [len(lick) for lick in licks_right]
    # n_licks = np.where(df_behavior.Side == 0, n_licks_left, n_licks_right)  # N licks in correct side
    # n_licks = np.where(df_behavior.Choice == 0, n_licks_left, n_licks_right)  # N licks in chosen side
    n_licks = np.array(n_licks_left) + np.array(n_licks_right)  # Total N licks in both sides
    ili_left = inter_lick_interval(licks_left)
    ili_right = inter_lick_interval(licks_right)
    # ili = np.where(df_behavior.Side == 0, ili_left, ili_right)  # ILI in correct side
    # ili = np.where(df_behavior.Choice == 0, ili_left, ili_right)  # ILI in chosen side
    ili = [np.nanmean([l, r]) if not (np.isnan(l) and np.isnan(r)) else np.nan for l, r in zip(ili_left, ili_right)]
    rt = get_rt(df_behavior)

    # Add lick vars to DataFrame
    df_behavior['LicksLeft'] = licks[0]
    df_behavior['LicksRight'] = licks[1]
    df_behavior['nLicksLeft'] = n_licks_left
    df_behavior['nLicksRight'] = n_licks_right
    df_behavior['nLicks'] = n_licks
    df_behavior['leftILI'] = ili_left
    df_behavior['rightILI'] = ili_right
    df_behavior['ILI'] = ili
    df_behavior['RT'] = rt

    # Add RT2 to DataFrame
    licks, premature_lick_trials = get_peri_event_licks(df_behavior, event='StimStart', time_window=0)
    rt2 = get_rt2(licks, df_behavior)
    df_behavior['RT2'] = rt2

    return df_behavior


def model_licks(df_behavior, var='RT', drug=False, me=False):

    # Create column for after error trials (invert AfterHit)
    df_behavior['AfterError'] = 1 - df_behavior['AfterHit']

    # Before error: next trial is an error
    df_behavior['BeforeError'] = df_behavior['AfterError'].shift(-1)
    df_behavior['BeforeError'] = df_behavior['BeforeError'].fillna(0).astype(int)

    if var == 'RT' or var == 'ILI':
        formula_cols = ['Choice', 'Hit', 'BeforeError', 'AfterError', 'absILD', 'NormTrial', 'NormTrial2', 'p1']
        formula = f'{var} ~ Choice + Hit + BeforeError + AfterError + absILD + absILD:Hit + NormTrial + NormTrial2 + p1'
        family = sm.families.Gaussian()
        print(f'Fitting GLM with Gaussian family for {var}...')
    elif var == 'nLicks':
        # Remove Hit and AfterHit for nLicks as will fit only correct trials
        formula_cols = ['Choice', 'absILD', 'NormTrial', 'NormTrial2', 'p1']
        formula = f'{var} ~ Choice + absILD + NormTrial + NormTrial2 + p1'
        family = sm.families.Poisson()
        me = False  # No mixed effects for Poisson
        print(f'Fitting GLM with Poisson family for {var}...')
    else:
        raise ValueError('Variable must be RT, ILI, or nLicks')

    formula_cols.append(var)
    if drug:
        formula_cols.append('Drug')
        parts = formula.split(' + ')
        parts.append('Drug')
        parts.append('Drug:p1')
        formula = ' + '.join(parts)
        print('Including Drug and its interaction with p(eng) as a predictor...')

    subjects = df_behavior.Subject.unique()
    all_params = []
    all_pvals = []

    for subj in subjects:

        df_subj = df_behavior[df_behavior.Subject == subj].copy()
        print(f'Fitting model for subject {subj} ({len(df_subj)})...')

        # For nLicks, fit only correct trials
        if var == 'nLicks':
            df_subj = df_subj[df_subj.Hit == 1]

        # Keep only sessions with at least 50 trials
        min_trials_session = 50
        trials_session = df_subj.groupby('Session').size()  # Count trials per session after subsetting
        to_drop = trials_session[trials_session < min_trials_session]  # Identify sessions to drop
        print(f'Dropping {len(to_drop)} sessions with <50 trials:')
        for sess_id, n_trials in to_drop.items():
            print(f'Session {sess_id}: {n_trials} trials')
        df_subj = df_subj.groupby('Session').filter(lambda x: len(x) >= min_trials_session)

        # Transform Choice: 0→-1, 1→1
        df_subj['Choice'] = df_subj['Choice'] * 2 - 1

        # Normalize absILD
        df_subj['absILD'] = df_subj['absILD'] / df_subj['absILD'].max()

        # Normalize within session and zscore within subject
        df_subj['NormTrial'] = df_subj.groupby('Session')['Trial'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()))
        df_subj['NormTrial2'] = df_subj['NormTrial'] ** 2  # Squared term
        df_subj['NormTrial'] = df_subj['NormTrial'].astype(float)
        df_subj['NormTrial2'] = df_subj['NormTrial2'].astype(float)

        # Drop rows with NaNs in any of the formula columns (also remove the misses)
        df_subj = df_subj.dropna(subset=formula_cols)
        df_subj.reset_index(drop=True, inplace=True)

        # Yet to include:
        # Session index (included with tne mixed effects model)
        # p(engaged) from HMM
        # Drug for drug analyses

        # Mixed effects model with random intercepts for sessions
        if me:
            model = smf.mixedlm(formula=formula, data=df_subj, groups=df_subj.Session)
            result = model.fit()
            params = result.fe_params

        # Fit GLM (Gaussian family for continuous RTs)
        else:
            model = smf.glm(formula=formula, data=df_subj, family=family)
            # model = smf.ols(formula=formula, data=df_subj)  # Much faster for Gaussian than GLM
            result = model.fit()
            params = result.params  # Different attribute name for mixedlm

        print(result.summary())

        # Store fixed-effect estimates and p-values
        params.name = subj
        all_params.append(params)
        pvals = result.pvalues
        pvals.name = subj
        all_pvals.append(pvals)

    # Convert to DataFrames
    df_params = pd.DataFrame(all_params)
    df_p = pd.DataFrame(all_pvals)

    return df_params, df_p


def tukey_fence(arr, k=1.5):
    """
    Apply Tukey's fences to identify non-outlier data points in a 1D array (designed for RTs).
    :param arr: 1D array of data points (e.g., RTs)
    :param k: Multiplier for the interquartile range (IQR) to define the fences (default: 1.5)
    :return: Boolean mask indicating non-outlier data points
    """

    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    IQR = Q3 - Q1
    lower_fence = Q1 - (k * IQR)
    upper_fence = Q3 +(k * IQR)
    mask = (arr >= lower_fence) & (arr <= upper_fence)

    return mask


########################################################################################################################

# Plotting functions


def plot_licks_dist(df_behavior, var='RT', bin_size=0.001, smooth=True, z=False, density=False, sem=True, **kwargs):
    """
    Plot the licks distribution for a variable of interest.
    :param df_behavior: DataFrame with the behavioral data
    :param var: Variable to plot (e.g., 'RT', 'nLicks', 'ILI')
    :param density: If True, plot density instead of frequency (default: False)
    :return: None
    """

    # if z:
    #     df_behavior[var] = df_behavior.groupby('Subject')[var].transform(zscore)
    #     bin_edges = (-3, 3)  # Normal distribution edges
    #     n_bins = 1000
    #     bin_size = (bin_edges[1] - bin_edges[0]) / n_bins  # Update
    #     bins = np.linspace(bin_edges[0], bin_edges[1], n_bins + 1)
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #     window = 0.06  # bin_size * sigma_old
    #     sigma = window / bin_size  # = 10
    # else:
    #     bin_edges = (0, df_behavior.RespWin.unique()[0])
    #     n_bins = int((bin_edges[1] - bin_edges[0]) / bin_size)
    #     bins = np.linspace(bin_edges[0], bin_edges[1], n_bins + 1)
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #     window = 0.01  # bin_size * 10
    #     sigma = window / bin_size
    #     xlim = (0, 0.5)

    # plt.figure(constrained_layout=True, **kwargs)
    color = kwargs.pop('color', 'k')  # Default black
    label = kwargs.pop('label', None)  # Default None
    subjects = df_behavior.Subject.unique()

    # Continuous variables
    if var == 'RT' or var == 'ILI':

        if var == 'RT':
            xlim = (0, 0.5)
            xticks = ([0, 0.25, 0.5], ['0', '0.25', '0.5'])
            # pass
        elif var == 'ILI':
            mean_ILI = df_behavior.ILI.mean()
            std_ILI = df_behavior.ILI.std()
            xlim = (mean_ILI - std_ILI, mean_ILI + std_ILI)
            xticks = ([0, 0.05, 0.1, 0.15, 0.2], ['0', '0.05', '0.1', '0.15', '0.2'])

        bin_edges = (0, df_behavior.RespWin.unique()[0])
        n_bins = int((bin_edges[1] - bin_edges[0]) / bin_size)
        bins = np.linspace(bin_edges[0], bin_edges[1], n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # if density:
        #     ylabel = 'Density'
        #     sns.kdeplot(df_behavior[var], color=color, **kwargs)
        # else:
        ylabel = 'Frequency (norm.)'
        window = 0.01  # bin_size * 10
        sigma = window / bin_size
        # Average across them (mean and sem)
        if len(subjects) > 1:
            hists = []
            for subj in subjects:
                data = df_behavior.loc[df_behavior.Subject == subj, var].copy()
                hist, _ = np.histogram(data, bins=bins, density=False)
                hist = hist / hist.sum()  # Normalize per subject
                hists.append(hist)
            hists = np.array(hists)
            mean_hist = np.mean(hists, axis=0)
            if smooth:
                mean_hist = gaussian_filter1d(mean_hist, sigma=sigma)
            mean_hist = mean_hist / mean_hist.sum()
            if sem:
                sem_hist = hists.std(axis=0) / np.sqrt(hists.shape[0])
                if smooth:
                    sem_hist = gaussian_filter1d(sem_hist, sigma=sigma)
                plt.fill_between(bin_centers, mean_hist - sem_hist, mean_hist + sem_hist, color=color, alpha=0.25,
                                 edgecolor='none')
            plt.plot(bin_centers, mean_hist, color=color, label=label)
        else:
            hist, _ = np.histogram(df_behavior[var], bins=bins, density=False)
            # hist = np.convolve(hist, np.ones(window)/window, mode='same')  # Moving average
            if smooth:
                hist = gaussian_filter1d(hist, sigma=sigma)  # Gaussian filter
            hist = hist / hist.sum()  # Normalize to sum 1
            plt.plot(bin_centers, hist, color=color, label=label)
            # plt.step(bin_centers, hist, color=color, label=label)
            # plt.hist(df_behavior[var], bins=bins, density=False, color=color, edgecolor=color, **kwargs)

        plt.xlim(xlim)
        plt.xticks(*xticks)
        # plt.ylim(0, None)
        plt.xlabel('Time (s)')

    # Discrete variable
    elif var == 'nLicks':
        min_val = df_behavior[var].min()
        max_val = df_behavior[var].max()
        bins = np.arange(min_val - 0.5, max_val + 1.5, 1)  # Centers bins on integers
        plt.hist(df_behavior[var], bins=bins, histtype='step', color=color, density=True)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatic integer ticks
        plt.xlim(0, 16)
        # plt.xlim(min_val - 0.5, max_val + 0.5)
        plt.xlabel('N licks')
        plt.xticks((4, 8, 12))
        loc = 'best'
        if density:
            ylabel = 'Density'
        else:
            ylabel = 'Frequency (norm.)'

    if df_behavior.Subject.unique().size > 1:
        title = (f'{var}\n'
                 f'N={len(df_behavior.Subject.unique())}, {len(df_behavior)/1000:.1f}k trials')
    else:
        title = (f'{var}\n'
                 f'{df_behavior.Subject.unique()[0]}, {len(df_behavior)/1000:.1f}k trials')

    # plt.title(title)
    plt.ylabel(ylabel)
    sns.despine()

    # if 'label' in kwargs:
    #     plt.legend()


def plot_licks_split(df_behavior, var='RT', split='outcome', kind='hist', **kwargs):
    """
    Plot the licks distribution of a variable of interest split by condition.
    :param df_behavior: DataFrame with the behavioral data
    :param var: Variable to plot (e.g., 'RT', 'nLicks', 'ILI')
    :param split: Split by 'outcome', 'choice', 'stim', 'rep_choice', 'rep_trial', 'prev_out', or 'session_half'
    :param kind: Kind of plot to use ('hist' or 'kde')
    :return:
    """

    # Split
    if split == 'outcome':
        split_var_name = 'Hit'
        colors = ['tab:red', 'tab:green']
        labels = ['Error', 'Correct']
    elif split == 'choice':
        split_var_name = 'Choice'
        colors = ['tab:blue', 'tab:orange']
        labels = ['Left', 'Right']
    elif split == 'stim':
        split_var_name = 'Side'
        colors = ['tab:blue', 'tab:orange']
        labels = ['Left', 'Right']
    elif split == 'rep_choice':
        split_var_name = 'RepChoice'
        colors = ['tab:purple', 'tab:brown']
        labels = ['Alt.', 'Rep.']
    elif split == 'rep_trial':
        split_var_name = 'RepTrial'
        colors = ['tab:purple', 'tab:brown']
        labels = ['Alt.', 'Rep.']
    elif split == 'prev_out':
        split_var_name = 'AfterHit'
        colors = ['tab:red', 'tab:green']
        labels = ['After error', 'After correct']
    elif split == 'half':
        split_var_name = 'SessionHalf'
        colors = ['tab:gray', 'k']
        labels = ['1st half', '2nd half']
    elif split == 'drug':
        split_var_name = 'Drug'
        colors = ['tab:gray', 'tab:pink']
        labels = ['Saline', 'Drug']
    elif split == 'state':
        split_var_name = 'State'
        colors = ['tab:gray', 'tab:green']
        labels = ['Dis.', 'Eng.']

    # plt.figure(constrained_layout=True, **kwargs)

    ylim = []
    for i in range(2):

        split_var = df_behavior[df_behavior[split_var_name] == i][var]

        # Continuous variables
        if var == 'RT' or var == 'ILI':

            if var == 'RT':
                xlim = (0, 0.5)
                xticks = (0, 0.1, 0.2, 0.3, 0.4, 0.5)
                loc = 'upper right'
            elif var == 'ILI':
                mean_ILI = df_behavior.ILI.mean()
                std_ILI = df_behavior.ILI.std()
                xlim = (mean_ILI - std_ILI, mean_ILI + std_ILI)
                xticks = (0, 0.05, 0.1, 0.15, 0.2)
                loc = 'lower center'

            if kind == 'hist':
                # plt.hist(split_var, bins=1000, density=False, label=labels[i], color=colors[i],
                #          edgecolor='none', alpha=0.5)
                plot_licks_dist(df_behavior[df_behavior[split_var_name] == i], var=var, label=labels[i], color=colors[i], **kwargs)
                ylabel = 'Frequency (norm.)'
            elif kind == 'kde':
                sns.kdeplot(split_var, color=colors[i], label=labels[i])
                ylabel = 'Density'
            plt.xlim(xlim)
            # plt.xticks(*xticks)
            plt.xticks(xticks, xticks)
            plt.xlabel('Time (s)')

            # Fix y-axis
            ylim.append(plt.gca().get_ylim()[1])

        # Discrete variable
        elif var == 'nLicks':
            min_val = split_var.min()
            max_val = split_var.max()
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)  # Centers bins on integers
            density = True if kind == 'kde' else False
            plt.hist(split_var, bins=bins, density=True, histtype='step', color=colors[i], label=labels[i])
            # ax = plt.gca()
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatic integer ticks
            plt.xlim(0, 16)
            plt.xticks((4, 8, 12))
            # plt.xlim(min_val - 0.5, max_val + 0.5)
            plt.xlabel('N licks')
            loc = 'best'
            if kind == 'kde':
                ylabel = 'Density'
            else:
                ylabel = 'Frequency (norm.)'

        if df_behavior.Subject.unique().size > 1:
            title = (f'{var}\n'
                     f'N={len(df_behavior.Subject.unique())}, {len(df_behavior)/1000:.1f}k trials')
        else:
            title = (f'{var}\n'
                     f'{df_behavior.Subject.unique()[0]}, {len(df_behavior)/1000:.1f}k trials')

    # plt.ylim(0, max(ylim))  # Set y-axis limit to 10% above the highest peak
    plt.legend(loc=loc, frameon=False)
    # plt.title(title)
    plt.ylabel(ylabel)
    sns.despine()


def plot_ild_dist(df_behavior, var='RT', insets=False, **kwargs):
    """
    Plot the licks distribution of a variable of interest split by absolute ILD levels.
    :param df_behavior: DataFrame with the behavioral data of a session
    :param var: Reaction Time (RT) or number of licks (nLicks)
    :return:
    """

    # plt.figure(constrained_layout=True)

    # Collapse the signed ILD levels to absolute values for cleaner visualization
    abs_ilds = sorted(df_behavior.absILD.unique().astype(int), reverse=False)
    # palette = list(sns.color_palette('tab10', len(abs_ilds)))[::-1]

    norm = plt.Normalize(vmin=min(abs_ilds), vmax=16)  # Scale |ILD| → 0–1
    cmap = cm.get_cmap('Reds')  #Continuous colormap
    palette = [cmap(norm(ild)) for ild in abs_ilds]

    peaks = {}
    # Plot the distribution for each absolute ILD level
    for i, ild in enumerate(abs_ilds):
        df_ild = df_behavior[df_behavior.absILD == ild]
        color = palette[i]

        # Continuous variables
        if var == 'RT' or var == 'ILI':

            if var == 'RT':
                xlim = (0, 0.5)
                # xlim = (0, 0.2)
                loc = 'upper center'
            elif var == 'ILI':
                mean_ILI = df_behavior.ILI.mean()
                std_ILI = df_behavior.ILI.std()
                xlim = (mean_ILI - std_ILI, mean_ILI + std_ILI)
                loc = 'upper left'

            # Plot and capture the Line2D object
            # sns.kdeplot(df_ild[var], color=color, label=ild)
            plot_licks_dist(df_behavior[df_behavior.absILD == ild], var=var, sem=False, color=color, label=ild, **kwargs)
            # plt.plot(bin_centers, hist, color=color, label=ild)
            plt.xlim(xlim)

            # Extract x and y data from the plotted line
            line = plt.gca().lines[-1]
            x, y = line.get_data()

            # Find the peak (x at maximum y)
            peak_rt = x[np.argmax(y)]
            peaks[ild] = peak_rt
            print(f'ILD {ild}: peak {var} = {peak_rt:.3f} s')

        # Discrete variable
        elif var == 'nLicks':
            min_val = df_ild[var].min()
            max_val = df_ild[var].max()
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)  # Centers bins on integers
            plt.hist(df_ild[var], bins=bins, density=True, histtype='step', color=color, label=ild)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatic integer ticks
            plt.xlim(0, 16)
            # plt.xlim(min_val - 0.5, max_val + 0.5)
            plt.xlabel(var)

            # nlicks = df_ild[var]
            # nlicks_mean = nlicks.mean()
            # nlicks_sem = sem(nlicks)
            # plt.errorbar(i, nlicks_mean, nlicks_sem, fmt='o', color=color, label=ild)
            loc = 'upper right'

    # # Print peaks
    # for ild, peak in peaks.items():
    #     print(f'ILD {ild}: peak {var} = {peak:.3f} s')

    mean_rt = np.mean(list((peaks.values())))
    print(f'Mean {var} = {mean_rt:.3f} s')

    # plt.axvline(0.15, color='k', linestyle='--')
    plt.title(var + ' distribution')
    plt.xlabel(var)
    plt.ylabel('Density')
    plt.legend(loc=loc, frameon=False, title='|ILD|')
    sns.despine()

    if var == 'RT' and insets:

        # Zoomed inset on the peak of the distribution
        ax = plt.gca()  # get current axes
        ax_inset = inset_axes(ax, width='30%', height='30%', loc='upper right')
        xlim = (mean_rt - 0.025, mean_rt + 0.025)

        for i, line in enumerate(ax.lines):
            x = line.get_xdata()
            y = line.get_ydata()
            mask = (x >= xlim[0]) & (x <= xlim[1])
            color = palette[i]
            ax_inset.plot(x[mask], y[mask], label=line.get_label(), color=color)

        ax_inset.set_xlim(xlim)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        sns.despine(ax=ax_inset)

        # # Zoomed inset on the second peak of the distribution
        # ax_inset = inset_axes(ax, width='30%', height='30%', loc='center right')
        # xlim = (0.15, 0.2)
        #
        # for i, line in enumerate(ax.lines):
        #     x = line.get_xdata()
        #     y = line.get_ydata()
        #     mask = (x >= xlim[0]) & (x <= xlim[1])
        #     color = palette[i]
        #     ax_inset.plot(x[mask], y[mask], label=line.get_label(), color=color)
        #
        # ax_inset.set_xlim(xlim)
        # ax_inset.set_xticks([])
        # ax_inset.set_yticks([])
        # sns.despine(ax=ax_inset)


def plot_ild_dist_mean(df_behavior, var='RT', jitter=0, **kwargs):
    """
    Plot the mean ± SEM of a variable for each absolute ILD level as a categorical bar plot.
    """

    abs_ilds = sorted(df_behavior.absILD.unique().astype(int))
    subjects = df_behavior.Subject.unique()
    palette = list(sns.color_palette('tab10', len(abs_ilds)))

    color = kwargs.pop('color', 'k')  # Default black
    label = kwargs.pop('label', None)  # Default None

    centers = []
    errors = []

    # Compute means and SEMs
    for ild in abs_ilds:

        per_subject_vals = []
        for subj in subjects:
            df_subj_ild = df_behavior[(df_behavior.Subject == subj)&(df_behavior.absILD == ild)]

            if var == 'RT':
                val = df_subj_ild[var].median()  # per-subject median
            else:
                val = df_subj_ild[var].mean()    # per-subject mean
            per_subject_vals.append(val)
        per_subject_vals = np.array(per_subject_vals)

        center = np.mean(per_subject_vals, axis=0)
        error = sem(per_subject_vals, axis=0)

        # if var == 'RT':
        #     center = np.median(per_subject_vals)
        #     error = median_abs_deviation(per_subject_vals, scale='normal') / np.sqrt(len(per_subject_vals))
        #     ylabel = 'Median ± MAD/√N (s)'
        # else:
        #     center = per_subject_vals.mean()
        #     error = sem(per_subject_vals)
        #     ylabel = 'Mean ± SEM'

        centers.append(center)
        errors.append(error)

    # y_min = min(centers) - 0.25 * (max(centers) - min(centers))
    x = np.arange(len(abs_ilds))  # positions for the bars
    # plt.bar(x, centers, yerr=errors, color=palette)
    # # plt.bar(x, centers - y_min, bottom=y_min, yerr=errors, color=palette)

    plt.errorbar(x + jitter, centers, errors , marker='o', color=color, label=label)

    # plt.ylim(y_min, None)
    plt.xticks(x, abs_ilds)  # ILD values as category labels
    plt.xlabel('|ILD|')
    plt.ylabel('Mean ± SEM')
    plt.title(f'{var}')
    sns.despine()


def plot_ild_dist_mean_split(df_behavior, var='RT', split='outcome', **kwargs):
    """
    Plot the mean ± SEM of a variable for each absolute ILD level, split by condition.
    """

    # Split
    if split == 'outcome':
        split_var_name = 'Hit'
        colors = ['tab:red', 'tab:green']
        labels = ['Error', 'Correct']
    elif split == 'choice':
        split_var_name = 'Choice'
        colors = ['tab:blue', 'tab:orange']
        labels = ['Left', 'Right']
    elif split == 'stim':
        split_var_name = 'Side'
        colors = ['tab:blue', 'tab:orange']
        labels = ['Left', 'Right']
    elif split == 'rep_choice':
        split_var_name = 'RepChoice'
        colors = ['tab:purple', 'tab:brown']
        labels = ['Alt.', 'Rep.']
    elif split == 'rep_trial':
        split_var_name = 'RepTrial'
        colors = ['tab:purple', 'tab:brown']
        labels = ['Alt.', 'Rep.']
    elif split == 'prev_out':
        split_var_name = 'AfterHit'
        colors = ['tab:red', 'tab:green']
        labels = ['After error', 'After correct']
    elif split == 'half':
        split_var_name = 'SessionHalf'
        colors = ['tab:gray', 'k']
        labels = ['1st half', '2nd half']
    elif split == 'drug':
        split_var_name = 'Drug'
        colors = ['tab:gray', 'tab:pink']
        labels = ['Saline', 'Drug']

    for _ in range(2):
        if _ == 0:
            jitter = 0
        else:
            jitter = 00

        df_split = df_behavior[df_behavior[split_var_name] == _]
        plot_ild_dist_mean(df_split, var=var, jitter=jitter, color=colors[_], label=labels[_])

    plt.legend()


def plot_licks_per_subject(df_behavior, plot_func, ncols=5, **kwargs):
    """
    Plot a subplot per subject of a given plotting function. Adjust automatically the figure grid to fit all subjects.
    :param df_behavior: DataFrame containing the behavioral data.
    :param plot_func: Plotting function to be applied.
    :param kwargs: Keyword arguments to be passed to `plot_func`.
    :return: None
    """

    subjects = df_behavior['Subject'].unique()
    n_subj = len(subjects)

    # Layout: N columns, enough rows
    nrows = -(-n_subj // ncols)  # Ceiling division

    figsize = kwargs.pop('figsize', fig_size(n_cols=0, ratio=1))
    # figsize = fig_size(n_cols=0, ratio=1)

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=False, squeeze=False, figsize=figsize, constrained_layout=True)

    for ax, subject in zip(axes.flatten(), subjects):
        df_subject = df_behavior[df_behavior['Subject'] == subject]
        plt.sca(ax)  # Make ax current
        ax.set_box_aspect(1)  # Make axes square
        plot_func(df_subject, **kwargs)
        # title = (f'#{subject}, {len(df_behavior) / 1000:.1f}k trials')
        title = (f'#{subject}')
        ax.set_title(title)

        # Remove legends if any
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Collect all y-limits and apply the same ylim to all subplots
    max_y = max(ax.get_ylim()[1] for ax in axes.flatten())
    for ax in axes.flatten():
        ax.set_ylim(0, max_y)

    # Remove y-ticks and labels for all axes except first column
    for i, ax in enumerate(axes.flatten()):
        if i % ncols != 0:  # not first column
            ax.set_yticklabels([])
            ax.set_ylabel('')

    # Remove unused axes if any
    for ax in axes.flatten()[len(subjects):]:
        ax.remove()

    # Remove subplots' axes labels
    for ax in axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Recover kwargs for labeling
    var = kwargs.get('var', None)
    density = kwargs.get('density', None)

    if var == 'RT' or var == 'ILI':
        title = 'Reaction time' if var == 'RT' else 'Interlick interval'
        supxlabel = 'Time (s)'
    else:
        title = 'Number of licks'
        supxlabel = 'N licks'

    # if density:
    #     supylabel = 'Density'
    # else:
    supylabel = 'Frequency (norm.)'

    fig.suptitle(title)
    fig.supxlabel(supxlabel, fontsize=plt.rcParams['axes.labelsize'])
    fig.supylabel(supylabel, fontsize=plt.rcParams['axes.labelsize'])


def plot_chrono_curve(df_behavior, absolute=True, **kwargs):
    """
    Plot the chronometric curve of a behavioral session (all trials).
    :param df_behavior: DataFrame with the behavioral data of a session
    :param absolute: If True, plot the absolute value of ILD (default: True)
    :return:
    """

    # df_behavior.loc[df_behavior['ILD'] == -70, 'ILD'] = -20
    # df_behavior.loc[df_behavior['ILD'] == 70, 'ILD'] = 20

    subjects = df_behavior['Subject'].unique()

    if absolute:
        df_behavior['absILD'] = df_behavior['ILD'].abs()
        ild_col = 'absILD'
        # mean_rts = df_behavior.groupby('absILD')['RT'].mean().reset_index()
        # ilds = sorted(df_behavior['absILD'].unique())
        # x = mean_rts['absILD']
        xlabel = '|ILD|'
    else:
        ild_col = 'ILD'
        # mean_rts = df_behavior.groupby('ILD')['RT'].mean().reset_index()
        # ilds = sorted(df_behavior['ILD'].unique())
        # x = mean_rts['ILD']
        xlabel = 'ILD'

    if len(subjects) > 1:
        # Mean across mice
        mouse_means = df_behavior.groupby(['Subject', ild_col])['RT'].median().reset_index()   # Mean per mouse per ILD
        mean_rts = mouse_means.groupby(ild_col)['RT'].mean().reset_index()
        sem_rts = mouse_means.groupby(ild_col)['RT'].sem().reset_index()
    else:
        # Single mouse: just mean per ILD
        mean_rts = df_behavior.groupby(ild_col)['RT'].median().reset_index()
        sem_rts = df_behavior.groupby(ild_col)['RT'].sem().reset_index()

    ilds = sorted(df_behavior[ild_col].unique())
    x = mean_rts[ild_col]
    y = mean_rts['RT']
    yerr = sem_rts['RT']

    if df_behavior.Subject.unique().size > 1:
        title = f'Chronometric Curve\n N={len(df_behavior.Subject.unique())} mice, {len(df_behavior)} trials'
    else:
        title = f'Chronometric Curve\n ID: {df_behavior.Subject.unique()[0]}, N={len(df_behavior)} trials'

    print(mean_rts)
    plt.figure(constrained_layout=True, **kwargs)
    # plt.plot(x, y, color='k', marker='o', linestyle='-')
    plt.errorbar(x, y, yerr=yerr, fmt='o-', color='k', linestyle='-')
    plt.xticks(ilds)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Mean RT (s)')
    plt.grid()
    sns.despine()


def plot_chrono_curve_split(df_behavior, split='outcome', absolute=True):
    """
    Plot the chronometric curve of a behavioral session split by outcome.
    :param df_behavior: DataFrame with the behavioral data of a session
    :param split: Split by 'outcome' or 'hit_error'
    :param absolute: If True, plot the absolute value of ILD (default: True)
    :return:
    """

    # Split
    if split == 'outcome':
        var = 'Hit'
        colors = ['tab:red', 'tab:green']
        labels = ['Error', 'Hit']
    elif split == 'choice':
        var = 'Choice'
        colors = ['tab:blue', 'tab:orange']
        labels = ['Left', 'Right']
    elif split == 'stim':
        var = 'Side'
        colors = ['tab:blue', 'tab:orange']
        labels = ['Left', 'Right']
    elif split == 'repeat':
        var = 'RepTrial'
        colors = ['tab:purple', 'tab:brown']
        labels = ['Alt.', 'Rep.']
    elif split == 'prev_out':
        var = 'AfterHit'
        colors = ['tab:red', 'tab:green']
        labels = ['Error', 'Hit']

    # Collapse trials by ILD or not
    if absolute:
        df_behavior['absILD'] = df_behavior['ILD'].abs()
        ilds = sorted(df_behavior['absILD'].unique())
        ild_col = 'absILD'
        xlabel = '|ILD|'
    else:
        ilds = sorted(df_behavior['ILD'].unique())
        ild_col = 'ILD'
        xlabel = 'ILD'

    plt.figure(constrained_layout=True)
    for i in range(2):

        subset = df_behavior[df_behavior[var] == i]
        mean_rts = subset.groupby(ild_col)['RT'].mean().reset_index()
        x = mean_rts[ild_col]
        y = mean_rts['RT']

        plt.plot(x, y, color=colors[i], marker='o', linestyle='-', label=labels[i])

    # Title
    if df_behavior.Subject.unique().size > 1:
        title = f'Chronometric Curve\n N={len(df_behavior.Subject.unique())} mice, {len(df_behavior)} trials'
    else:
        title = f'Chronometric Curve\n ID: {df_behavior.Subject.unique()[0]}, N={len(df_behavior)} trials'

    plt.xticks(ilds)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Mean RT (s)')
    plt.legend(frameon=False)
    plt.grid()
    sns.despine()


def plot_model_licks(df_params, df_p, **kwargs):

    # Compute mean, SEM and p values across mice
    params_mean = df_params.mean()
    params_sem = df_params.sem()

    for name in params_mean.index:
        mean_val = params_mean[name]
        sem_val = params_sem[name]
        print(f'{name}: {mean_val:.5f} ± {sem_val:.5f}')

    # Drop intercept column
    df_params = df_params.drop('Intercept', axis=1, errors='ignore')  # ME doesn't have intercept
    df_p = df_p.drop('Intercept', axis=1, errors='ignore')  # ME doesn't have intercept

    # Apply Bonferroni correction
    n_tests = len(df_params.columns)
    t_test_results = {}
    for col in df_params.columns:
        t_stat, p_val = ttest_1samp(df_params[col], 0, nan_policy='omit')
        p_bonf = min(p_val * n_tests, 1)  # Bonferroni correction (cap at 1)
        t_test_results[col] = {'t': t_stat, 'p': p_val, 'p_bonf': p_bonf}
        print(f'Test {col}: t = {t_stat:.2f}, p = {p_bonf:.4f}')

    print(t_test_results)

    # Drop Trial and Trial^2 columns if present
    df_params = df_params.drop(['NormTrial', 'NormTrial2'], axis=1, errors='ignore')

    # Highlight effects where mean p-value < 0.05
    # colors = ['red' if df_p[col].mean() < 0.05 else 'gray' for col in df_p.columns]
    colors = ['tab:pink' if t_test_results[col]['p_bonf'] < 0.05 else 'tab:gray' for col in df_params.columns]

    # Define the labels to rename
    label_map = {
        'Hit': 'Correct',
        'BeforeError': 'Before\nerror',
        'AfterError': 'After\nerror',
        'absILD': "|ILD|'",
        'absILD:Hit': "|ILD|':\nCorrect",
        'NormTrial': "Trial'",
        'NormTrial2': "Trial²",
        'p1': r'$p$(eng.)',
        'Drug': 'Drug',
        'Drug:p1': 'Drug:\n$p$(eng.)'
    }

    plt.figure(constrained_layout=True, **kwargs)
    plt.axhline(0, color='k', linestyle='--')

    # bp = plt.boxplot([df_params[col] for col in df_params.columns],
    #                  showfliers=False,
    #                  labels=[label_map.get(col, col) for col in df_params.columns],
    #                  patch_artist=True,
    #                  medianprops=dict(color='black'), showcaps=False)
    sns.boxplot(data=df_params.rename(columns=label_map),
                palette=colors,
                showfliers=False,
                showcaps=False,
                fill=False,
                width=0.5)  # Matplotlib's default, prevent adaptive width
    # plt.xticks(rotation=45, ha='center')
    plt.ylabel('Weights')

    pvals = [t_test_results[col]['p_bonf'] for col in df_params.columns]
    y = [df_params[col].max() for col in df_params.columns]
    add_stars(pvals, y)

    # # Color the boxes
    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)

    plt.title('Kernel')
    sns.despine()

    return t_test_results


########################################################################################################################

# DEVELOPPING:

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import seaborn as sns
# from glue_sessions.glue_sessions import *
# from cherry import *
# from licks import *
# from plotting_style import *
# from my_fun.my_fun import save_notebook_files
# # Plotting style
# default_figsize = np.array(plt.rcParams['figure.figsize'])
# style_path = os.path.expanduser('~/PycharmProjects/alexis_style.mplstyle')
# plt.style.use(style_path)
#
#
# experiments = [
#     '2AFC_2',
#     '2AFC_3',
#     '2AFC_4',
#     # '2AFC_5',
#     # '2AFC_6',
# ]
# df_behavior = glue_groups(experiments)
# df_behavior = df_behavior[df_behavior.Miss == 0].reset_index(drop=True)
# df_behavior = df_behavior[df_behavior.P > 0].reset_index(drop=True)
#
# # Save old columns for comparison later
# old_columns = list(df_behavior.columns)
#
# # Add lick data to the DataFrame
# df_behavior = add_lick_data(df_behavior)
#
# # Add session half index (0 for the 1st and 1 for the 2nd)
# df_behavior['SessionHalf'] = (df_behavior.Trial >= df_behavior.groupby('Session').Trial.transform('max') / 2).astype(int)
# # Add absolute ILD
# loc = df_behavior.columns.get_loc('ILD') + 1  # To the right of ILD column
# df_behavior.insert(loc, 'absILD', df_behavior['ILD'].abs())  # Add previous stimulus side column
#
# # Print new columns added to the DataFrame
# new_columns = list(df_behavior.columns)
# new_columns = [col for col in new_columns if col not in old_columns]
# print(new_columns)
#
# # Sometimes pandas doesn't recognize all nans as actual nans. Force it and remove them
# df_behavior['RT'] = pd.to_numeric(df_behavior['RT'], errors='coerce')
# df_behavior = df_behavior.dropna(subset=['RT']).reset_index(drop=True)
# assert len(df_behavior) == len(df_behavior['RT'].dropna().values.reshape(-1, 1))
#
# if '2AFC_4' in experiments:
#     df_behavior.loc[df_behavior.Experiment == '2AFC_4', 'RT'] -= 0.15
# if '2AFC_5' in experiments:
#     df_behavior.loc[df_behavior.Experiment == '2AFC_5', 'RT'] -= 0.15
# if '2AFC_6' in experiments:
#     df_behavior.loc[df_behavior.Experiment == '2AFC_6', 'RT'] -= 0.15
#
# df_behavior.Subject = df_behavior.Subject.astype(int).astype(str).str.zfill(3)  # 0 padd subjects for consistent XXX ID and ensure string format
# animals = main(experiments)  # Cherry pick
# all_animals = [a for expt in ['2AFC_2', '2AFC_3', '2AFC_4'] for a in animals[expt]]  # Unpack cherries
# print(f'{len(all_animals)} mice: {all_animals}')
# df_behavior = df_behavior[df_behavior['Subject'].isin(all_animals)]  # Filter cherries
