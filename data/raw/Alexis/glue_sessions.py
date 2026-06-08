import time
from pathlib import Path
import os
import numpy as np
import pandas as pd
from parse.parse import parse
from parse.parse_v2 import parse_v2
import csv
from my_fun.my_fun import *
# To do:
# Add training day index column to df


# Define functions
def glue_sessions(animal=None, protocol='stage_training_v6', experiment='2AFC_6', to_csv=False):
    """
    Glue all the sessions of a given animal.
    :param animal: ID number of the animal
    :param protocol: task code version
    :param experiment: batch of the animal
    :param to_csv: if True save data as .csv file
    :return: pandas DataFrame with the data, .csv file with the ID of the corrupted sessions
    """

    if experiment is None:

        folder_in = Path.home() / 'pv_nmdar_eranet' / 'experiments'  # Where the data for all animals is
        experiments = os.listdir(folder_in)  # List experiments
        experiments.sort()  # Sort them by name

        experiments_to_remove = ['.idea', 'Daily check', 'WaterCalibration']
        for _ in range(len(experiments_to_remove)):
            try:
                experiments.remove(experiments_to_remove[_])
            except ValueError:
                pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    folder_in = Path.home() / 'pv_nmdar_eranet' / 'experiments' / experiment / 'setups'  # Where the data for all animals is

    if animal is None:

        animals = os.listdir(folder_in)  # List animals
        animals.sort()  # Sort them by name

        # Usually I don't want Test subject(s)
        animals_to_remove = ['Test', 'Test0', 'Test1', 'Test2', 'Test3', 'Test4', 'Test5', 'Test6', 'Test7', 'Test8',
                             '.idea']  # Pycharm's file

        for i in range(len(animals_to_remove)):
            try:
                animals.remove(animals_to_remove[i])
            except ValueError:
                pass

        print('Animals: ' + str(animals)[1:-1])  # Remove square brackets
        animal = input('Enter animal')  # Ask user to input animal to glue sessions from

    # # Check if csv from that animal already exist, and if so, import it
    glued_sessions = []  # Initialize empty list so if it's the first time glue all sessions

    # Select the output folder and create it if it doesn't exist
    folder_out = Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment
    if not os.path.exists(folder_out):
        # os.mkdir(folder_out)
        folder_out.mkdir(parents=True, exist_ok=True)

    glued_animals = os.listdir(folder_out)
    glued_animals.sort()
    glued_animals = [x for x in glued_animals if x.endswith('.csv')]  # Get rid of non csv files

    if animal + '.csv' in glued_animals:
        df = pd.read_csv(Path(Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment / animal).with_suffix('.csv'),
                         low_memory=False)
        glued_sessions = df.Session.unique().tolist()
    else:
        df = pd.DataFrame()  # Create empty DataFrame if there's no csv yet for that animal

    folder_in = Path(folder_in / animal / 'sessions')  # Update folder_in with selected animal
    sessions = os.listdir(folder_in)  # List sessions
    sessions.sort()  # Sort them by date

    if protocol is None:

        protocols = []  # Initiate list
        for i, session in enumerate(sessions):
            # print(i, session)
            protocols.append(sessions[i][4:-16])  # Remove animal ID (beginning) and date and time (end)

        print('There are ' + str(len(sessions)) + ' sessions of this animal, ' + str(len(np.unique(protocols))) +
              ' protocols found:')
        for i in range(len(np.unique(protocols))):
            print(i, ' ', np.unique(protocols)[i], ': ', protocols.count(np.unique(protocols)[i]), sep='')

        protocols = list(np.unique(protocols))
        protocol = input('Enter protocol (choose number)')
        protocol = str(protocols[int(protocol)])

    print('Gluing sessions of animal ' + animal + '...\n')

    corrupted_sessions = []

    for i in range(len(sessions)):

        # Loop only over sessions with the selected protocol that aren't glued yet
        if protocol in sessions[i] and sessions[i] not in glued_sessions:
            # path = folder_in + sessions[i] + '/' + sessions[i] + '.csv'  # Get csv file path to input parse.py
            path = Path(folder_in / sessions[i] / sessions[i]).with_suffix('.csv')  # Get csv file path to input parse.py
            print('Parsing session ' + "'" + sessions[i] + "'" + '...', sep='')

            try:
                if protocol == 'stage_training':
                    df_session = parse(path)  # Parse session

                elif protocol in ['stage_training_v2', 'stage_training_v3', 'stage_training_v4', 'stage_training_v5',
                                  'stage_training_v6']:
                    df_session = parse_v2(path)  # Parse session v2
                df = pd.concat([df, df_session])  # Add parsed session to the bottom of the DataFrame
            except (IndexError, ValueError, FileNotFoundError, ZeroDivisionError):  # When passing 2 exceptions it must be in this syntax
                print(
                    f"The session '{sessions[i]}' is corrupted. Adding to corrupted sessions log and continuing with "
                    f"next session...")
                corrupted_sessions.append(sessions[i])

        else:
            pass

    if to_csv:
        # df.to_csv(folder_out + animal + '.csv', index=False)  # index=False to avoid the 'Unmmaed: 0' column
        df.to_csv(Path(folder_out / animal).with_suffix('.csv'), index=False)  # index=False to avoid the 'Unmmaed: 0' column

    print('The corrupted sessions are:', *corrupted_sessions, '\n', sep='\n')

    if corrupted_sessions:  # If corrupted sessions isn't empty, save them in a .csv file
        # Save corrupted sessions in a separate csv file
        with open(Path(Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment / '_corrupted_sessions.csv'),
                  'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(corrupted_sessions)

    return df, corrupted_sessions


def update_glued_sessions(experiment='2AFC_6'):
    """
    Update the glued_sessions .csv files for all animals with the non yet included sessions.
    :param experiment: batch of the animals
    :return:
    """

    if experiment is None:

        folder = Path.home() / 'pv_nmdar_eranet' / 'experiments'  # Where the data for all animals is
        experiments = os.listdir(folder)  # List experiments
        experiments.sort()  # Sort them by name

        experiments_to_remove = ['.idea', 'Daily check', 'WaterCalibration']
        for _ in range(len(experiments_to_remove)):
            try:
                experiments.remove(experiments_to_remove[_])
            except ValueError:
                pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    if experiment == 'Ephys':
        protocol = 'stage_training_v5'  # Ephys experiment (2AFC_5) uses stage_training_v5 protocol
    else:
        protocol = 'stage_training_v' + experiment[-1]  # Get the last digit of the experiment

    folder = Path.home() / 'pv_nmdar_eranet' / 'experiments' / experiment / 'setups'  # Where the data for all animals is
    animals = os.listdir(folder)  # List animals
    animals.sort()  # Sort them by name

    # Usually I don't want Test subject(s)
    animals_to_remove = ['Test', 'Test0', 'Test1', 'Test2', 'Test3', 'Test4', 'Test5', 'Test6', 'Test7', 'Test8',
                         '.idea']  # Pycharm's file
    for i in range(len(animals_to_remove)):
        try:
            animals.remove(animals_to_remove[i])
        except ValueError:
            pass

    for i in range(len(animals)):
        print(f'Updating sessions of animal {animals[i]}...')
        glue_sessions(animal=animals[i], protocol=protocol, experiment=experiment, to_csv=True)


def glue_animals(experiment='2AFC_6', path_session='glue_sessions', filter_drug=True, update=False, to_csv=False):
    """
    Glue all the sessions from all the animals of a given batch.
    :param experiment: batch of animals
    :param update: If True update first the glued sessions
    :param to_csv: if True save data as .csv file
    :return: pandas DataFrame with the data
    """

    # Get the path to the data
    experiment, folder = get_experiment(experiment, path_session=path_session)

    # Update first the glued sessions
    if update:
        update_glued_sessions(experiment=experiment)  # Update glued sessions first

    animals = os.listdir(folder)  # List animals
    animals.sort()  # Sort them by name
    animals = [x for x in animals if not 'corrupted_sessions' in x]  # Get rid of the corrupted sessions csv files

    df = pd.DataFrame()  # Create empty dataframe

    for i in range(len(animals)):
        df_animal = pd.read_csv(Path(folder / animals[i]), low_memory=False)
        if filter_drug:
            df_animal = filter_drug_sessions(df_animal)
            # print(f'Filtering paired saline-drug sessions for experiment {experiment}')
        df = pd.concat([df, df_animal])  # Add parsed session to the bottom of the DataFrame
    df.reset_index(drop=True, inplace=True)

    if to_csv:
        filename = experiment + '_glued_sessions'
        df.to_csv(Path(folder / filename).with_suffix('.csv'), index=False)  # index=False to avoid the 'Unnamed: 0' column

    return df


def glue_groups(experiments=['2AFC_2', '2AFC_3', '2AFC_4', '2AFC_5', '2AFC_6'], path_session='glue_sessions'):
    """
    Glue all sessions from all animals from all groups.
    :param groups: Batches of animals to glue together
    :return: pandas Dataframe with the data
    """

    df = pd.DataFrame()

    for exp in experiments:
        df_groups = glue_animals(experiment=exp, path_session=path_session, filter_drug=False, update=False, to_csv=False)
        df = pd.concat([df, df_groups])  # Add parsed session to the bottom of the DataFrame
    df.reset_index(drop=True, inplace=True)

    return df


def add_drug_column(experiment):
    """
    Add a 'Drug' column filled with NaNs to the glued sessions .csv files so they match group 6 (pharma)
    :param experiment: group of animals
    :return:
    """

    experiment, folder_in = get_experiment(experiment)
    animals = os.listdir(folder_in)  # List animals
    animals.sort()  # Sort them by name
    animals = [x for x in animals if not 'corrupted_sessions' in x]  # Get rid of the corrupted sessions csv files

    for _ in range(len(animals)):
        df = pd.read_csv(Path(folder_in / animals[_]), low_memory=False)
        if 'Drug' not in df.columns:
            print(f'Adding drug column to animal {animals[_]}...')
            df['Drug'] = np.nan
            df.to_csv(Path(folder_in / animals[_]), index=False)  # index=False to avoid the 'Unnamed: 0' column
        else:
            print(f'Drug column already exists in animal {animals[_]}. Skipping to next animal...')
