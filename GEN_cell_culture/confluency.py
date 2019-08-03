import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, re

from ProteomicsUtils.LoggerConfig import logger_config
from ProteomicsUtils import FileHandling, StatUtils, CalcUtils, PlotUtils

logger = logger_config(__name__)
logger.info("Import OK")

input_day1 = '_Measurements.csv'
input_day2 = '_Measurements.csv'

output_path = 'Python_results/confluency/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

# Read in results from Day 1
day1_raw = pd.read_csv(input_day1)
day1_raw

## append more useable sample labels
day1_cleaning = day1_raw.copy()
# determine well number
day1_cleaning['Well #'] = day1_cleaning['Label'].str.split(' - ', expand=True)[0]

#day1_cleaning['Plating regime'] = ['Low' if any(label in low_list for ext in extensionsToCheck) in low_list else 'High' if label in high_list else np.nan for label in day1_cleaning['Label']]
day1_cleaning['Plate #'] = day1_cleaning['Well #'].str.split('_', expand=True)[0]
day1_cleaning['Treatment #'] = day1_cleaning['Well #'].str.split('_', expand=True)[2]
day1_cleaning['Plate_coords'] = day1_cleaning['Plate #'].map(str) +'_' + day1_cleaning['Treatment #'].map(str)
day1_cleaning

# Add density
plate_1_density = ['TPE only', '1', '1.5', '2', '3', '4']
plate_2_density = ['PI only', 'No stain', 'Control', 'SFM']
plate_cords = [f'{x}_{y}' for x in range (1, 3) for y in range (1, 7)]
density_map = dict(zip(plate_cords, (plate_1_density+plate_2_density)))
day1_cleaning['Density'] = day1_cleaning['Plate_coords'].map(density_map)


# Generate individual comparative results for Day1 -> Day2
# Read in results from Day 1
day2_raw = pd.read_csv(input_day2)
day2_raw

## append more useable sample labels
day2_cleaning = day2_raw.copy()
# determine well number
day2_cleaning['Well #'] = day2_cleaning['Label'].str.split(' - ', expand=True)[0]

#day1_cleaning['Plating regime'] = ['Low' if any(label in low_list for ext in extensionsToCheck) in low_list else 'High' if label in high_list else np.nan for label in day1_cleaning['Label']]
day2_cleaning['Plate #'] = day2_cleaning['Well #'].str.split('_', expand=True)[0]
day2_cleaning['Treatment #'] = day2_cleaning['Well #'].str.split('_', expand=True)[2]
day2_cleaning['Plate_coords'] = day2_cleaning['Plate #'].map(str) +'_' + day2_cleaning['Treatment #'].map(str)
day2_cleaning

# Add drug treatment name
day2_cleaning['Density'] = day2_cleaning['Plate_coords'].map(density_map)

day2_cleaning

# Generate plot for each plate

plotting_plates = []
for group, data in day2_cleaning.groupby('Plate #'):

    day2_plot = day2_cleaning[['Well #', 'Density', '%Area']]
    day2_plot.rename(columns={'%Area': 'After'}, inplace=True)
    day1_plot = day1_cleaning[day1_cleaning['Plate #'] == group][['Well #', 'Density', '%Area']]
    day1_plot.rename(columns={'%Area': 'Before'}, inplace=True)

    plotting = pd.merge(day1_plot, day2_plot, on=['Well #', 'Density'])


    plotting = plotting.melt(id_vars=['Density'], value_vars = ['Before', 'After'])
    categories = ['Before', 'After']
    plotting['Hue position'] = plotting['variable'].astype("category", ordered=True, categories=categories).cat.codes
    plotting_plates.append(plotting)

density_list = [plate_1_density, plate_2_density]

fig, axes = plt.subplots(2, 1, figsize=(8, 8))

for x, plate in enumerate(plotting_plates):
    # Generate figures
    sns.barplot(x='Density', y='value', data=plate, hue='variable', dodge=True,errwidth=1.25,alpha=0.25,ci=None, ax=axes[x])
    sns.swarmplot(x='Density', y='value', data=plate, hue='variable', dodge=True, ax=axes[x])

    # To generate custom error bars
    sample_list = list(plate['Density'])[0:len(density_list[x])]
    xcentres=np.arange(0, len(sample_list))
    delt=0.2
    xneg=[x-delt for x in xcentres]
    xpos=[x+delt for x in xcentres]
    xvals=xneg+xpos
    xvals.sort()
    yvals=plate.groupby(["Density","Hue position"]).mean().T[sample_list].T['value']
    yerr=plate.groupby(["Density","Hue position"]).std().T[sample_list].T['value']

    (_, caps, _) = axes[x].errorbar(x=xvals,y=yvals,yerr=yerr,capsize=4,elinewidth=1.25,ecolor="black", linewidth=0)
    for cap in caps:
        cap.set_markeredgewidth(2)
    axes[x].set_ylabel("Confluency (%)")

    # To only label once in legend
    handles, labels = axes[x].get_legend_handles_labels()
    axes[x].legend(handles[0:2], ['Day 1', 'Day 2'], bbox_to_anchor=(1.26, 1.05))

    # rotate tick labels

    for label in axes[x].get_xticklabels():
        label.set_rotation(45)

plt.ylabel("Confluency (%)")
plt.tight_layout()
plt.autoscale()
    #plt.show()

FileHandling.fig_to_pdf([fig], output_path+f'Confluency_')
FileHandling.fig_to_svg([f'Confluency_'], [fig], output_path)
