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
data_path = []
for i in range(0,18):
  temp_path = root_path + '/Class' + str(i)  
  class_i = []
  for j in os.listdir(temp_path):
   class_i.append(os.path.join(temp_path, j)) # '.../Dataset/Class0/0.png'
  data_path.append(class_i)

count = 0
for i in range(0,18):
  print("{} = {}".format(class_labels[i], len(data_path[i])))
  count = count + len(os.listdir(temp_path))
print("Total: " + str(count))

# Shuffle Function
def shuffle_data(dataset):
  for data in dataset:
    random.shuffle(data)
  return np.asarray(dataset)

data_path = shuffle_data(data_path)

# Checking after shuffle
for i in range(data_path.shape[0]):
  print('class {}'.format(i))
  print(data_path[i])

# Create directories corresponding to class name
train_path = '/content/drive/My Drive/AIC2022/Dataset6/Training'
for s in class_labels:
  os.makedirs(os.path.join(train_path, s))

# Distribute to Training directory
for i in range(data_path.shape[0]):
  print('Class: {}'.format(class_labels[i]))
  target_path = os.path.join(train_path, class_labels[i])
  for j in range(len(data_path[i])):
    img = Image.open(data_path[i][j])
    img.save(target_path + '/' + str(j) + '.png')