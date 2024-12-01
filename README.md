# EEG Feature Generation Module 

## Overview

This Python module provides functionality for generating features from EEG data, including the preprocessing, feature extraction, and visualization of results. It is designed to handle EEG data for two subjects (referred to as `sb1` and `sb2`) and provides a modular approach to data processing.

### Version: 2.0  
**Authors:** Olisemeka Nmarkwe
**Date Created:** 11/16/2024  
**Date Last Updated:** 11/30/2024

## Features
- **EEG Data Preprocessing**: Includes notch filtering, impedance filtering, band-pass filtering, and re-referencing of EEG data.
- **Feature Extraction**: Statistical features such as mean, standard deviation, kurtosis, and skewness are extracted from sliding windows of EEG data.
- **Target Class Generation**: Assigns target labels based on subject (`sb1` or `sb2`).
- **Visualization**: This function generates histograms, boxplots, violin plots, and sorted distribution plots for the training and testing datasets.

## Requirements
- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`

The input folder contains the EEG data for two subjects (`sb1` and `sb2`). Each subject folder should have the necessary session data for the feature extraction process.

## Module Workflow

1. **Loading EEG Data**: The function `load_files_insession()` loads the EEG data for a given subject and session. The data is processed by channel name and stored for feature extraction.

2. **Data Preprocessing**: The following preprocessing functions are applied:
    - **Notch Filtering**: Removes power line noise at specific frequencies (60, 120, 180, 240 Hz).
    - **Impedance Filtering**: Filters out noise around the center frequency of 125 Hz.
    - **Bandpass Filtering**: Retains frequencies between 0.5 and 32 Hz.
    - **Re-referencing**: Applies a common reference to the EEG data by subtracting the mean of all channels.

3. **Feature Extraction**: For each window in the EEG data, the following features are calculated:
    - Mean
    - Standard deviation
    - Kurtosis
    - Skewness

4. **Statistical Aggregation**: The features are aggregated for each subject, session, and stream, followed by the creation of a target class for classification (`Sb1` or `Sb2`).

5. **Visualization**: The module generates several plots:
    - **Histograms** for categorical distributions (`Sb1` vs `Sb2`).
    - **Sorted Bar Charts** for continuous features.
    - **Box-Whisker Plots** and **Violin Plots** for the continuous features.

6. **Save Data**: The processed feature datasets are saved as CSV files (`TrainValidateData_{channel_name}.csv`, `TestData_{channel_name}.csv`) for further analysis.



