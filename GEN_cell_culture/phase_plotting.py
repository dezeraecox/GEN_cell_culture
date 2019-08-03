import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import string
from loguru import logger

from datetime import datetime

from GEN_Utils import FileHandling

logger.info("Import OK")

input_path = 'Python_results/gauss_models/'
output_path = 'Python_results/phase_plotting/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

matplotlib.rcParams.update(_VSCode_defaultMatplotlib_Params)
matplotlib.rcParams.update({'figure.facecolor': (1,1,1,1)})

# read in summary df
summary = []
for filename in [filename for filename in os.listdir(input_path) if '.xlsx' in filename]:
    summary.append(pd.read_excel(f'{input_path}{filename}'))
summary = pd.concat(summary)

# Assign sample-specific descriptors
summary['well'] = summary['sample'].str.strip('1')

plate_sample = ['TPE only', '1', '1.5', '2', '3', '4']
plate_cords = [f'{x}{y}' for x in string.ascii_uppercase[0:4]
               for y in range(1, 7)]
sample_map = dict(zip(plate_cords, plate_sample*4))
summary['density'] = summary['well'].map(sample_map)

phase_name = ['G', 'S', 'M']
phase_num = [1, 2, 3]
phase_map = dict(zip(phase_name, phase_num))


# Generate melted df for plotting scattbar
for_plot = pd.melt(summary, id_vars=['density'], value_vars=[
                   'G', 'S', 'M'], var_name='phase', value_name='proportion', col_level=None)
plotting_plates = []
data = for_plot.copy()
data['Hue position'] = data['phase'].map(phase_map)
plotting_plates.append(data)

samples = ['1', '1.5', '2', '3', '4']
fig, axes = plt.subplots(1, 1, figsize=(8, 3))

for x, plate in enumerate(plotting_plates):
    # Generate figures
    br = sns.barplot(x='density', y='proportion', data=plate, hue='phase',
                     dodge=True, errwidth=1.25, alpha=0.25, ci=None, ax=axes)
    scat = sns.swarmplot(x='density', y='proportion', data=plate,
                         hue='phase', dodge=True, ax=axes)

    # To generate custom error bars
    sample_list = list(set(plate['density']))
    number_groups = len(list(set(plate['Hue position'])))

    bars = br.patches
    xvals = [(bar.get_x() + bar.get_width()/2) for bar in bars]
    xvals.sort()
    # collect mean, sd for each bar
    yvals = plate.groupby(
        ["density", "phase", "Hue position"]).mean().T[samples].T
    yvals.reset_index(inplace=True)
    yvals.rename(columns={'proportion': 'mean'}, inplace=True)
    yvals['error'] = list(plate.groupby(
        ["density", "phase", "Hue position"]).std().T[samples].T['proportion'])
    yvals = yvals.sort_values(
        ["density", "Hue position"])

    (_, caps, _) = axes.errorbar(x=xvals, y=yvals['mean'],
                                    yerr=yvals['error'], capsize=2, elinewidth=1.25, ecolor="grey", linewidth=0)
    for cap in caps:
        cap.set_markeredgewidth(2)
    axes.set_ylabel("Confluency (%)")

    # To only label once in legend
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles[0:number_groups], labels[0:number_groups],
                   bbox_to_anchor=(1.26, 1.05), title='Phase')

    # rotate tick labels

    for label in axes.get_xticklabels():
        label.set_rotation(45)

axes.set_ylabel("Confluency (%)")
axes.set_xlabel(r'Density (x 10$^5$)')


plt.tight_layout()
plt.autoscale()
#plt.show()
plt.savefig(f'{output_path}scattbar_plot.png')

# Generate line-plot

fig = plt.subplots()
for phase in phase_name:
    sns.lineplot(summary['density'], summary[phase], label=phase, ci='sd')
plt.ylabel("Proportion of cells in phase")
plt.xlabel('r'Density(x 10$^ 5$)'')
plt.title('Phase distribution')
plt.legend(bbox_to_anchor=(1.1, 1.0), title='Phase')
plt.tight_layout()
plt.autoscale()
plt.savefig(f'{output_path}line_plot.png')


