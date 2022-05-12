# Import libraries
import numpy as np
from PIL import Image
import cv2
import os
import random

# Import Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set rootpath
root_path = '/content/drive/My Drive/AIC2022/Dataset6/A1'
print(os.listdir(root_path))

# Set 18 class labels
class_labels = ['Normal Forward Driving', 'Drinking', 'Phone Call(right)', 'Phone Call(left)', 'Eating',
                'Text (Right)', 'Text (Left)', 'Hair_makeup', 'Reaching behind', 'Adjust control panel', 'Pick up from floor (Driver)',
                'Pick up from floor (Passenger)', 'Talk to passenger at the right', 'Talk to passenger at backseat', 'yawning',
                'Hand on head', 'Singing with music', 'shaking or dancing with music']

# Set and count the number of images from each class folder
datasetA1_path = []
datasetA2_path = []
for i in range(0,18):
    tempA1_path = root_path + '/A1/Class' + str(i)
    tempA2_path = root_path + '/A2/Class' + str(i)
    classA1_i = []
    classA2_i = []
    for j in os.listdir(tempA1_path):
        class_i.append(os.path.join(temp_path, j)) # '.../Dataset/Class0/0.png'
    for j in os.listdir(tempA2_path):
        class_i.append(os.path.join(temp_path, j)) # '.../Dataset/Class0/0.png'
    dataA1_path.append(classA1_i)
    dataA2_path.append(classA2_i)

count = 0
print("Dataset A1:\n")
for i in range(0,18):
  print("{} = {}".format(class_labels[i], len(dataA1_path[i])))
  count = count + len(os.listdir(tempA1_path))
print("Total: " + str(count))

count = 0
print("Dataset A2:\n")
for i in range(0,18):
  print("{} = {}".format(class_labels[i], len(dataA2_path[i])))
  count = count + len(os.listdir(tempA2_path))
print("Total: " + str(count))

# Check the size of the image
img = Image.open(data_path[12][305])
img.size

# Spliting to train, test and validation sets
def list_splitter(list_to_split, ratio = 0.5):
  length = len(list_to_split)
  pivot = int(length * ratio)
  return [list_to_split[:pivot], list_to_split[pivot:]]

def train_test_split(datasetA1, datasetA2, ratio = 0.5):
  train_set = datasetA1
  val_set = []
  test_set = []
  for data in datasetA2:
    list_i = data.copy()
    val, test = list_splitter(list_i, ratio)
    val_set.append(val)
    test_set.append(test)
  return train_set, val_set, test_set
  
train_set, val_set, test_set = train_test_split(dataA1_path, dataA2_path)

# Convert list to numpy array
train_set = np.asarray(train_set)
val_set = np.asarray(val_set)
test_set = np.asarray(test_set)

# Create directories corresponding to class name
train_path = os.path.join(root_path, 'Training')
val_path = os.path.join(root_path, 'Validation')
test_path = os.path.join(root_path, 'Testing')

for s in class_labels:
  os.makedirs(os.path.join(train_path, s))
  os.makedirs(os.path.join(val_path, s))
  os.makedirs(os.path.join(test_path, s))

# Distribute to Training directory
for i in range(train_set.shape[0]):
  target_path = os.path.join(train_path, class_labels[i])
  for j in range(len(train_set[i])):
    img = Image.open(train_set[i][j])
    img.save(target_path + '/' + str(j) + '.png')

# Distribute to Validation directory
for i in range(val_set.shape[0]):
  print('Class: {}'.format(class_labels[i]))
  target_path = os.path.join(val_path, class_labels[i])
  for j in range(len(val_set[i])):
    img = Image.open(val_set[i][j])
    img.save(target_path + '/' + str(j) + '.png')

# Distribute to Testing directory
for i in range(test_set.shape[0]):
  print('Class: {}'.format(class_labels[i]))
  target_path = os.path.join(test_path, class_labels[i])
  for j in range(len(test_set[i])):
    img = Image.open(test_set[i][j]) 
    img.save(target_path + '/' + str(j) + '.png')
