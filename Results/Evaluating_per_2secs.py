# Import Libraries
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
import numpy as np
import pandas as pd

# Import Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
root_path = os.path.join('/content/drive', 'MyDrive/AIC2022')
model_dir = os.path.join(root_path, 'Model')

# Load Model
def load_model(model_path, weights_path):
    json_file = open(model_path, 'r')
    json = json_file.read()
    json_file.close()
    model = model_from_json(json)
    model.load_weights(weights_path)
    return model

# Load Dashboard Model
dashboard_model_path = os.path.join(model_dir, "Model for Dataset3/MobileNet_Dashboard.json")
dashboard_weights_path = os.path.join(model_dir, "MobileNet for Dataset6/MobileNet_Dashboard.h5")
dashboard_model = load_model(dashboard_model_path, dashboard_weights_path)
dashboard_model.summary()

# Define Classes
class_labels = ['Normal Forward Driving', 'Drinking', 'Phone Call(right)', 'Phone Call(left)', 'Eating',
                'Text (Right)', 'Text (Left)', 'Hair_makeup', 'Reaching behind', 'Adjust control panel', 'Pick up from floor (Driver)',
                'Pick up from floor (Passenger)', 'Talk to passenger at the right', 'Talk to passenger at backseat', 'yawning',
                'Hand on head', 'Singing with music', 'shaking or dancing with music']

activity_dict = {'Adjust control panel': 9, 'Drinking': 1, 'Eating': 4, 'Hair_makeup': 7,
                 'Hand on head': 15, 'Normal Forward Driving': 0, 'Phone Call(left)': 3,
                 'Phone Call(right)': 2, 'Pick up from floor (Driver)': 10, 'Pick up from floor (Passenger)': 11,
                 'Reaching behind': 8, 'Singing with music': 16, 'Talk to passenger at backseat': 13,
                 'Talk to passenger at the right': 12, 'Text (Left)': 6, 'Text (Right)': 5, 'shaking or dancing with music': 17, 'yawning': 14}

true_id = [9, 1, 4, 7, 15, 0, 3, 2, 10, 11, 8, 16, 13, 12, 6, 5, 17, 14]

target_width = 128
target_height = 128

# Preprocess: resize -> normalize -> expand_dims
def rescale_input(x):
    x = x.astype('float32')
    x /= 255.
    return x

def preprocess(image, target_dim = (128, 128)):
    img = image.copy()
    img = cv2.resize(img, target_dim)
    img = rescale_input(img)
    img = np.expand_dims(img, 0)
    return img

# Evaluate on video, only one prediction every 2 seconds
def predict_once_per_2secs(video_path_in, model):
    vidcap = cv2.VideoCapture(video_path_in)
    if (vidcap.isOpened() == False):
        print("Error reading video file")
        exit()
    
    count = 0
    sec = 0
    predictions = []
    timeframe = []      # list of [second, activity_id, label]
    label = ''
    while True:
        success, frame = vidcap.read() 
        if not success:
          break

        preprocess_frame = preprocess(frame)
        prediction = model.predict(preprocess_frame)
        predictions.append(prediction)

        count += 1
        if count % 60 == 0:
            pred = np.sum(predictions, axis=0) / 60
            idx = np.argmax(pred)
            class_id = true_id[idx]
            label = class_labels[class_id]
            count = 0
            predictions.clear()
            for i in range(0,2):
              temp_list = []
              temp_list.append(sec+i)
              temp_list.append(class_id)
              temp_list.append(label)
              timeframe.append(temp_list)
            sec += 2

        # cv2.putText(frame, label, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 224, 127), 2)
        # out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vidcap.release()
    cv2.destroyAllWindows()
    return timeframe

#Path for results
video_path_in = os.path.join(root_path, '2022/A2/user_id_42271/Dashboard_user_id_42271_NoAudio_4.MP4')
csv_path_out = os.path.join(root_path, 'Results/csv/Dashboard_user_id_42271_NoAudio_4_tested_2.csv')

# print(os.path.exists(video_path_out))
print(os.path.exists(csv_path_out))

timeframe = predict_once_per_2secs(video_path_in, dashboard_model)
print(timeframe)

for i in range(len(timeframe)):
  time = timeframe[i][0]
  minute = time // 60
  second = time - minute*60
  s = str(minute) + ':' + str(second)
  timeframe[i].append(s)


tf = pd.DataFrame(timeframe, columns=['Second Count', 'Class ID', 'Class Label', 'Time']) #Dataframe of timeframe
tf.to_csv(csv_path_out) #Export CSV from pandas dataframe