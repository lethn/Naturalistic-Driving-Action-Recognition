# Import Libraries
import os
import pandas as pd
import numpy as np

# Import Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define paths
root_path = os.path.join('/content/drive', 'MyDrive/AIC2022')
csv_dir = os.path.join(root_path, 'Results/csv')
path_out = os.path.join(csv_dir, 'overall_result.txt')

csv_name = ['42271_3_merged.csv', '42271_4_merged.csv', '56306_2_merged.csv', '56306_3_merged.csv',
            '65818_1_merged.csv', '65818_2_merged.csv', '72519_2_merged.csv', '72519_3_merged.csv',
            '79336_0_merged.csv', '79336_2_merged.csv']

path_in = [os.path.join(csv_dir, name) for name in csv_name]

# Generate Text
def text_from_csv(path_in, path_out):
    text = open(path_out, 'w')

    for id, csv in zip(range(1, 11), path_in):
        res = pd.read_csv(csv)
        res = res[res['Final'] > -1]
        
        for i in range(18):
            current_label = res[res['Final'] == i]
            if not current_label.empty: 
                start = current_label['Second Count'].iloc[0]
                end = current_label['Second Count'].iloc[-1]
                text.write('{} {} {} {}\n'.format(id, i, start, end))

    text.close()

text_from_csv(path_in, path_out)