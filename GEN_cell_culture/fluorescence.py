import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
from GEN_Utils import FileHandling

logger.info('Import OK')

input_folder = 'examples/python/gauss_models/normalised/'
output_folder = 'examples/python/phase_fluorescence/'

fluorescence_col = 'TPE'
plate_samples = ['TPE only', '1', '1.5', '2', '3', '4']*4
plate_cords = [f'{x}{y}' for x in string.ascii_uppercase[0:4]
               for y in range(1, 7)]
sample_map = dict(zip(plate_cords, plate_samples))

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Generate filelist
file_list = [filename for filename in os.listdir(input_folder)]

# Collect important info into summary df, grouped according to phase

sample_data = []
for filename in file_list:
    sample_name = os.path.splitext(filename)[0]

    raw_data = pd.read_csv(f'{input_folder}{filename}')
    raw_data.rename(columns={fluorescence_col: "fluorescence"}, inplace=True)

    fluo_data = raw_data.copy()[['phase', 'fluorescence']]
    fluo_data = fluo_data.groupby('phase').median().T
    fluo_data['sample'] = sample_name

    sample_data.append(fluo_data)

summary_df = pd.concat(sample_data).reset_index(drop=True)

summary_df['plate'] = summary_df['sample'].str[0]
summary_df['well'] = summary_df['sample'].str[1:]
summary_df['sample'] = summary_df['well'].map(sample_map)

summary_df.sort_values(['sample'], inplace=True)

FileHandling.df_to_excel(data_frames=[summary_df], sheetnames=[
                         'fluorescence_per_phase'], output_path=f'{output_folder}per_phase_median_TPE.xlsx')

# Generate equivalent dataset, ignoring phase
sample_data = {}
for filename in file_list:
    sample_name = os.path.splitext(filename)[0]

    raw_data = pd.read_csv(f'{input_folder}{filename}')
    raw_data.rename(columns={fluorescence_col: "fluorescence"}, inplace=True)

    fluo_data = raw_data.copy()['fluorescence']
    sample_data[sample_name] = fluo_data.median()

summary_df = pd.DataFrame.from_dict(sample_data, orient='index').reset_index()
summary_df.rename(columns={'index': 'sample',
                           0: 'med_fluorescence'}, inplace=True)

summary_df['plate'] = summary_df['sample'].str[0]
summary_df['well'] = summary_df['sample'].str[1:]
summary_df['sample'] = summary_df['well'].map(sample_map)

summary_df.sort_values(['plate', 'sample'], inplace=True)

FileHandling.df_to_excel(data_frames=[summary_df], sheetnames=[
                         'median_TPE'], output_path=f'{output_folder}median_TPE.xlsx')
