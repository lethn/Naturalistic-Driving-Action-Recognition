# Import libraries
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

# Import Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define folders path
root_path = os.path.join('/content/drive', 'MyDrive/AIC2022')

train_dir = os.path.join(root_path, 'Dataset3/Training')
test_dir = os.path.join(root_path, 'Dataset3/Testing')
a2_test = os.path.join(root_path, 'FrameTest/user_id_42271')

# Load model
model_path = os.path.join(root_path, 'Model')
json_file = open(os.path.join(model_path, 'MobileNet_model_accuracy_94,55%.json'), 'r') #Change model here
json = json_file.read()
mobileNet_model = model_from_json(json)
mobileNet_model.summary()

mobileNet_model.load_weights(os.path.join(model_path, 'Model for Dataset3/model-27-1.13.h5'))  #Change model here

# Modify some constants
activity_dict = ['Adjust control panel', 'Drinking', 'Eating', 'Hair_makeup', 'Hand on head',
                 'Normal Forward Driving', 'Phone Call(left)', 'Phone Call(right)', 'Pick up from floor (Driver)',
                 'Pick up from floor (Passenger)', 'Reaching behind', 'Singing with music', 'Talk to passenger at backseat', 
                 'Talk to passenger at the right', 'Text (Left)', 'Text (Right)', 'shaking or dancing with music', 'yawning']

target_width = 128
target_height = 128

def load_test_data(TESTING_DIRECTORY, BATCH_SIZE, IMAGE_SIZE):
  test_data_generator = ImageDataGenerator(rescale = 1./255)
  test_generator = test_data_generator.flow_from_directory(directory = TESTING_DIRECTORY,
                                                                  batch_size = BATCH_SIZE,
                                                                  class_mode = 'categorical',
                                                                  target_size = IMAGE_SIZE)
  return test_generator

test_generator = load_test_data(test_dir, 32, (target_width, target_height))

mobileNet_model.compile(loss='categorical_crossentropy',
          optimizer=Adam(learning_rate=0.001),
          metrics=['accuracy'])
eval = mobileNet_model.evaluate(test_generator)

# Normalize image function
def preprocess_img(images):
  images = np.array(images).astype(np.float32)
  images /= 255.0
  return images

#----------------------------------------------------------------------
# Read an image + resize + normalize

example_path = os.path.join(a2_test, 'frame97.png') # Change and try it
example = cv2.imread(example_path)
example = cv2.resize(example, (target_width, target_height))
h,w,n_c = example.shape
print(h, w, n_c)
cv2_imshow(example)

example = preprocess_img(example)	#Normalize

# Expand dimension, (1, 128, 128, 3) means 1 image in batch
example = np.expand_dims(example, 0)
print(example.shape) #(1, 128, 128, 3)

res = mobileNet_model.predict(example)
idx = np.argmax(res)
print('Predicted: class {}'.format(idx))
print('Label: ' + activity_dict[idx])

#---------------------------------------------------------------------------------------
# Read and test on multiple images
raw_images = [cv2.imread(os.path.join(a2_test, image)) for image in os.listdir(a2_test)]
images = [cv2.resize(image, (target_width, target_height)) for image in raw_images]

cv2_imshow(images[14])

images = preprocess_img(images)
print(images.shape)

# Prediction
res = mobileNet_model.predict(images)
print(res)

idx = np.argmax(res, axis = 1)
j = 0
for i in idx:
  print('Frame {}'.format(j))
  j = j + 1
  print('Predicted: class {}'.format(i))
  print('Label: ' + activity_dict[i])
  print('---------------------------------')