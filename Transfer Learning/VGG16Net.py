## Import libraries
import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import shutil
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Import Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set root path
root_path = os.path.join('/content/drive', 'MyDrive/AIC2022')

# Set train and test directory
train_dir = os.path.join(root_path, 'Dataset6/Training')
val_dir = os.path.join(root_path, 'Dataset6/Validation')

# Training and Validation Generators
def train_val_generator(TRAINING_DIRECTORY, VALIDATION_DIRECTORY, BATCH_SIZE, IMAGE_SIZE):
    train_data_generator = ImageDataGenerator(
      rescale = 1./255,
      width_shift_range = 0.2,
      height_shift_range = 0.2,
      zoom_range = 0.2,
      fill_mode = 'nearest')
    
    validation_data_generator = ImageDataGenerator(rescale = 1. / 255)

    train_generator = train_data_generator.flow_from_directory(directory = TRAINING_DIRECTORY, 
                                                               batch_size = BATCH_SIZE,
                                                               class_mode = 'categorical',
                                                                target_size = IMAGE_SIZE)
    
    validation_generator = validation_data_generator.flow_from_directory(directory = VALIDATION_DIRECTORY,
                                                             batch_size = BATCH_SIZE,
                                                             class_mode = 'categorical',
                                                              target_size = IMAGE_SIZE)
    
    # test_generator = validation_data_generator.flow_from_directory(directory = TESTING_DIRECTORY,
    #                                                               batch_size = BATCH_SIZE,
    #                                                               class_mode = 'categorical',
    #                                                               target_size = IMAGE_SIZE)
    
    return train_generator, validation_generator

train_generator, validation_generator = train_val_generator(train_dir, val_dir, 32, (128, 128))

# Transfer Learning - VGG16Net
from tensorflow.keras.applications.vgg16 import VGG16

vgg = VGG16(input_shape = (128, 128, 3),
                        include_top = False,
                        weights = 'imagenet')

for layer in vgg.layers:
    layer.trainable = False

vgg.summary()

# Callback
early_stop = EarlyStopping(monitor='val_loss', patience=5)

filepath = root_path + '/Model/VGG16Net for Dataset6/model-{epoch:02d}-{val_accuracy:.2f}.h5'
model_checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

callbacks = [model_checkpoint]

# Add more layers
last_output = vgg.output

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(18, activation='softmax')(x)

model=tf.keras.Model(inputs = vgg.input, outputs = x)
    
model.compile(optimizer=Adam(learning_rate = 0.001),
              loss='categorical_crossentropy',
               metrics=['accuracy'])

model.summary()

NUM_TRAIN = train_generator.samples
BATCH_SIZE = 32
VERBOSE = 1
NUM_EPOCH = 30

history = model.fit(train_generator, 
                    epochs = NUM_EPOCH, 
                    validation_data = validation_generator, 
                    verbose = VERBOSE,
                    steps_per_epoch = NUM_TRAIN // BATCH_SIZE,
                    callbacks = callbacks)

def display_model_history(hist):
  # To draw the Accuracy curve and Loss curve
  fig, axs = plt.subplots(1,2,figsize=(15,5))

  # Accuracy subplot
  axs[0].plot(range(1, len(hist.history['accuracy'])+1), hist.history['accuracy'])
  axs[0].plot(range(1, len(hist.history['val_accuracy'])+1), hist.history['val_accuracy'])
  axs[0].set_title('Model Accuracy')
  axs[0].set_xlabel('Epoch')
  axs[0].set_ylabel('Accuracy')
  axs[0].legend(['Train', 'Test'], loc='best')

  # Loss subplot
  axs[1].plot(range(1, len(hist.history['loss'])+1), hist.history['loss'])
  axs[1].plot(range(1, len(hist.history['val_loss'])+1), hist.history['val_loss'])
  axs[1].set_title('Model Loss')
  axs[1].set_xlabel('Epoch')
  axs[1].set_ylabel('Loss')
  axs[1].legend(['Train', 'Test'], loc='best')
  fig.savefig('plot.png')
  plt.show()

display_model_history(history) # Drawing

train_acc = history.history['accuracy']  # Get the model accuracy

# Define path to save weights (.h5 file) and model structure (.json file)
model_path = os.path.join(root_path, 'Model/VGG16Net for Dataset6')
weight_path = model_path + '/VGG16Net_model_accuracy_' + str(train_acc).replace('.', ',') + '%.h5'     #e.g. ../Model/model_accuracy_90%.h5
json_path = model_path + '/VGG16Net_model_accuracy_' + str(train_acc).replace('.', ',') + '%.json'

# Saving process
model.save_weights(weight_path)
model_json = model.to_json()
with open(json_path, "w") as json_file:
  json_file.write(model_json)

# Save training history as csv file using Pandas
hist_df = pd.DataFrame(history.history)
hist_csv_file_path = model_path + '/history.csv'
hist_df.to_csv(hist_csv_file_path, index=False)