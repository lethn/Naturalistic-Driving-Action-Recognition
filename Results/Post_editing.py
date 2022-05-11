# Import libraries
import pandas as pd
import numpy as np
import os

# Import Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define paths
root_path = os.path.join('/content/drive', 'MyDrive/AIC2022')

csv_path_in = os.path.join(root_path, 'Results', 'Dashboard_User_id_79336_NoAudio_2_tested.csv')
csv_path_out = os.path.join(root_path, 'Results/csv', '79336_2_merged.csv')
start_time = 13   # Start time to filter, avoid N/A at the beginning of videos

print(os.path.exists(csv_path_in))
print(os.path.exists(csv_path_out))

class_labels = ['Normal Forward Driving', 'Drinking', 'Phone Call(right)', 'Phone Call(left)', 'Eating',
                'Text (Right)', 'Text (Left)', 'Hair_makeup', 'Reaching behind', 'Adjust control panel', 'Pick up from floor (Driver)',
                'Pick up from floor (Passenger)', 'Talk to passenger at the right', 'Talk to passenger at backseat', 'yawning',
                'Hand on head', 'Singing with music', 'shaking or dancing with music']


# Get list of periods of activites: [[start_1, end_1], [start_2, end_2],...]
def get_periods(class_id, target, start_time):
  print(class_labels[target])
  periods = []
  start = end = -1
  for i in range(start_time, class_id.shape[0]):  # Start from start_time, to avoid N/A at the begining of video
    if class_id[i] == target:
      if i == start_time or class_id[i-1] != target:
        start = i
      if i == class_id.shape[0] - 1 or class_id[i+1] != target:
        end = i
        period = [start, end]   
        periods.append(period)
  return periods

# Merge periods which are adjacent
def merge_periods(periods):
  if len(periods) == 0:
    return []
  merged_periods = []
  merged_periods.append(periods[0])
  for i in range(1, len(periods)):
    prev = merged_periods[-1]
    if periods[i][0] - prev[1] <= 5:  # If the start of next period - end of previous period <= 5  => merge 2 period
      prev[1] = periods[i][1]         # Merge by assigning the end of current = the end of next period
    else:                             
      merged_periods.append(periods[i])
  return merged_periods

# Return the longest period
def get_longest_period(periods):
  if len(periods) == 0:
    return []
  max = 0
  max_idx = -1
  for i in range(len(periods)):
    if periods[i][1] - periods[i][0] > max:   # If end - start > max => max = this period
      max = periods[i][1] - periods[i][0]
      max_idx = i
  return periods[max_idx]

# Join all functions above to a pipeline
def filter(csv_path_in, csv_path_out, start_time):
  df = pd.read_csv(csv_path_in)
  class_id = df['Class ID'].values              # Get list of class_id predicted
  final_periods = [-1]*class_id.shape[0]        # Initialize a list of -1

  for i in range(18):
    periods = get_periods(class_id, i, start_time)  # Get periods of ith activity
    print(periods)
    periods = merge_periods(periods)            # Merge adjacent periods
    print(periods)
    longest_period = get_longest_period(periods)    # Get the longest one
    print(longest_period)
    if len(longest_period) > 0:
      for j in range(longest_period[0], longest_period[1]+1):   
        final_periods[j] = i                    # Assign class_id to appropriate positions, other are still -1
  df['Final'] = final_periods     
  df.to_csv(csv_path_out, index=False)
  print(final_periods)

filter(csv_path_in, csv_path_out, start_time)

df = pd.read_csv(csv_path_in)
class_id = df['Class ID'].values 
periods = get_periods(class_id, 5, 115)

print(periods)
print(class_id)