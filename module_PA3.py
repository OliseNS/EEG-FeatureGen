#%% MODULE BEGINS
module_name = 'module_PA3'

'''
Version: 4.2

Description:
    Feature Generation for EEG Data with Additional Visualization.

Authors:
    NeNai: Olisemeka Nmarkwe and Sujana Mehta. (W0762669 and W0757459 respectively)

Date Created     :  11/16/2024
Date Last Updated:  11/20/2024

Doc:
    This module generates features from EEG data while addressing redundancy and enhancing modularity.

Notes:
    Ensure the data directory structure and channel names are correctly configured before running.
'''

#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os

from copy import deepcopy as dpcpy
import pickle as pckl
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import butter, filtfilt, iirnotch
from matplotlib import pyplot as plt
import seaborn as sns

#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Base directory containing 'sb1' and 'sb2' folders

pathSoIRoot = 'INPUT\\stream'
base_dir = f'{pathSoIRoot}\\' 

#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DEFAULT_WINDOW_SIZE = 100  # Number of samples per window
DEFAULT_OVERLAP = 0.5      # 50% overlap between windows

#%% FUNCTION DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_files_insession(subject, session, channel_name):
    pathSoi = rf"{base_dir}{subject}\{session}"
    filelist = os.listdir(pathSoi)
    session_data = {}

    for soi_file in filelist:
        with open(f'{pathSoi}\\{soi_file}', 'rb') as fp:
            soi = pckl.load(fp)
        
        channel_info = soi['info']['eeg_info']['channels']
        channel_indices = [i for i, ch in enumerate(channel_info) if ch['label'][0] == channel_name]
        
        if channel_indices:
            index = channel_indices[0]
            session_data[soi_file] = soi['series'][index]
        
        sfreq = soi['info']['eeg_info']['effective_srate']

    return session_data, sfreq

def apply_notch_filter(data, fs, freqs):
    for freq in freqs:
        b, a = iirnotch(w0=freq, Q=30, fs=fs)
        data = filtfilt(b, a, data)
    return data

def apply_impedance_filter(data, fs, center, tolerance):
    low, high = center - tolerance, center + tolerance
    b, a = butter(N=2, Wn=[low, high], btype='bandstop', fs=fs)
    return filtfilt(b, a, data)

def apply_band_pass_filter(data, fs, lowcut, highcut):
    b, a = butter(N=2, Wn=[lowcut, highcut], btype='bandpass', fs=fs)
    return filtfilt(b, a, data)

def apply_rereferencing(data):
    reference = np.mean(data, axis=0)
    return data - reference

def get_stat_for_window(window_signal, se_val, sb_val, stream_id, index):
    mean = np.mean(window_signal)
    std_dev = np.std(window_signal)
    kur = kurtosis(window_signal)
    skewness = skew(window_signal)
    return {
        'sb': sb_val,
        'se': se_val,
        'stream_id': stream_id,
        'window_id': index,
        'mean': mean,
        'std': std_dev,
        'kur': kur,
        'skew': skewness           
    }

def get_features(df):
    grouped_df = df.groupby(['sb', 'se', 'stream_id'])[['mean', 'std', 'kur', 'skew']].agg([np.mean, np.std])
    result_df = grouped_df.assign(
        f1=lambda x: x['mean']['mean'],
        f2=lambda x: x['mean']['std'],
        f3=lambda x: x['std']['mean'],
        f4=lambda x: x['std']['std'],
        f5=lambda x: x['kur']['mean'],
        f6=lambda x: x['kur']['std'],
        f7=lambda x: x['skew']['mean'],
        f8=lambda x: x['skew']['std']
    )


    return result_df.reset_index()

def create_target_class(sb_value):
    return 'Sb1' if sb_value.lower() == 'sb1' else 'Sb2'


def new_df(df):
    new_df = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'target']].copy()
    new_df['sample_id'] = range(1, len(new_df) + 1)
    return new_df[['sample_id', 'target', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']].sort_values(by='sample_id')

def create_datasets(subject1, session1, subject2, session2, channel):
    # Helper function to collect features for a given subject and session
    def collect_features(subject, session, channel):
        session_data, sfreq = load_files_insession(subject, session, channel)
        statistics = []

        for file_name, data in session_data.items():
            # Apply preprocessing steps
            data = apply_notch_filter(data, sfreq, [60, 120, 180, 240])
            data = apply_impedance_filter(data, sfreq, 125, 1)
            data = apply_band_pass_filter(data, sfreq, 0.5, 32)
            data = apply_rereferencing(data)

            # Apply windowing and feature extraction
            for i in range(0, len(data) - DEFAULT_WINDOW_SIZE + 1, int(DEFAULT_WINDOW_SIZE * (1 - DEFAULT_OVERLAP))):
                window = data[i:i + DEFAULT_WINDOW_SIZE]
                statistics.append(get_stat_for_window(window, session, subject, file_name, i))

        return pd.DataFrame(statistics)

    # Collect features for training/validation and testing
    train_validate_df = pd.concat([
        collect_features(subject1, session1, channel),
        collect_features(subject2, session1, channel)
    ])
    test_df = pd.concat([
        collect_features(subject1, session2, channel),
        collect_features(subject2, session2, channel)
    ])

    # Aggregate features and add target labels
    train_validate_df = get_features(train_validate_df)
    train_validate_df['target'] = train_validate_df['sb'].apply(create_target_class)

    test_df = get_features(test_df)
    test_df['target'] = test_df['sb'].apply(create_target_class)

    # Structure the final dataframes
    train_validate_df = new_df(train_validate_df)
    test_df = new_df(test_df)

    return train_validate_df, test_df

def save_and_plot(train_validate, test, channel_name):
    # Save datasets to CSV files
    train_validate.to_csv(f'TrainValidateData_{channel_name}.csv', index=False)
    test.to_csv(f'TestData_{channel_name}.csv', index=False)

    # Combine datasets for visualization
    features_df = pd.concat([train_validate, test])

    # Plot for Training Data
    plot_data(train_validate, channel_name, "Training Data")

    # Plot for Testing Data
    plot_data(test, channel_name, "Testing Data")

def plot_data(df, channel_name, dataset_type):
    continuous_features = df.select_dtypes(include=[np.number]).columns.drop(['sample_id']).tolist()

    print(f"Continuous Features for Plotting: {continuous_features}")

    # Histogram for categorical attribute
    plt.figure(figsize=(6, 4))
    sns.histplot(df['target'], kde=False, palette="Set2", discrete=True)
    plt.title(f"{dataset_type} - Categorical Distribution for Channel: {channel_name}", fontsize=14)
    plt.xlabel('Target Class (Sb1 / Sb2)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xticks([0, 1], ['Sb1', 'Sb2'])
    plt.tight_layout()
    plt.savefig(f"{dataset_type}_Categorical_Distribution_{channel_name}.png")
    plt.close()

    # Sorted Bar Charts for Continuous Attributes
    num_cols_sorted = 2
    num_rows_sorted = (len(continuous_features) - 1) // num_cols_sorted + 1

    plt.figure(figsize=(7 * num_cols_sorted, 4 * num_rows_sorted))
    for i, column in enumerate(continuous_features, start=1):
        plt.subplot(num_rows_sorted, num_cols_sorted, i)
        sorted_values = df[column].sort_values().values
        plt.bar(range(len(sorted_values)), sorted_values, color='skyblue')
        plt.title(f"Sorted Distribution: {column}", fontsize=10, pad=5)
        plt.xlabel("Sorted Index", fontsize=8)
        plt.ylabel("Value", fontsize=8)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
    plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.5)
    plt.suptitle(f"{dataset_type} - Sorted Distributions for Channel: {channel_name}", fontsize=16, y=1.02)
    plt.savefig(f"{dataset_type}_Sorted_Distributions_{channel_name}.png", bbox_inches="tight")
    plt.close()

    # Box-Whisker and Violin Plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    sns.boxplot(data=df[continuous_features], orient='h', palette='Set2', ax=axes[0])
    axes[0].set_title(f"{dataset_type} - Box-Whisker Plots", fontsize=12, pad=10)
    axes[0].tick_params(axis='y', labelsize=8)
    axes[0].tick_params(axis='x', labelsize=8)

    sns.violinplot(data=df[continuous_features], orient='h', palette='Set3', ax=axes[1])
    axes[1].set_title(f"{dataset_type} - Violin Plots", fontsize=12, pad=10)
    axes[1].tick_params(axis='y', labelsize=8)
    axes[1].tick_params(axis='x', labelsize=8)

    plt.suptitle(f"{dataset_type} - Box-Whisker and Violin Plots for Channel: {channel_name}", fontsize=16, y=1.05)
    plt.tight_layout(pad=2.5, w_pad=3.0, h_pad=3.0)
    plt.savefig(f"{dataset_type}_BoxWhisker_Violin_{channel_name}.png", bbox_inches="tight")
    plt.close()



#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    for channel in ['M1', 'M2', 'CPz']:
        train_validate, test = create_datasets('sb1', 'se1', 'sb2', 'se2', channel)
        save_and_plot(train_validate, test, channel)

    print("Feature generation completed.")

# %%
