import os
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from GEN_Utils import FileHandling
from loguru import logger

logger.info("Import OK")

# Set sample-specific variables
input_path = 'examples/python/gauss_models/'
output_path = 'examples/python/phase_plotting/'

plate_sample = ['TPE only', '1', '1.5', '2', '3', '4']*4
plate_cords = [f'{x}{y}' for x in string.ascii_uppercase[0:4]
               for y in range(1, 7)]
sample_map = dict(zip(plate_cords, plate_sample))

if not os.path.exists(output_path):
    os.mkdir(output_path)

# Read in summary df and preview
summary = pd.read_excel(f'{input_path}summary.xlsx')

# Assign sample-specific descriptors to summary table
summary['plate'] = summary['sample'].str[0]
summary['well'] = summary['sample'].str[1:]
summary['sample'] = summary['well'].map(sample_map)

phase_name = ['G', 'S', 'M']
phase_num = [1, 2, 3]
phase_map = dict(zip(phase_name, phase_num))

# Generate line-plot
fig = plt.subplots()
for phase in phase_name:
    sns.lineplot(summary['sample'], summary[phase], label=phase, ci='sd')
plt.ylabel("Proportion of cells in phase")
plt.xlabel(r'Density(x 10$^ 5$)')
plt.title('Phase distribution')
plt.legend(bbox_to_anchor=(1.1, 1.0), title='Phase')
plt.tight_layout()
plt.autoscale()
plt.savefig(f'{output_path}line_plot.png')
