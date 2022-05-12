import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
import numpy as np


def load_model():
    """Load model json and h5 files"""
    model_path = './models'
    json_file = open(os.path.join(model_path, 'MobileNet_model.json'), 'r')
    json = json_file.read()
    json_file.close()
    model = model_from_json(json)
    model.load_weights(os.path.join(model_path, 'MobileNet_model_66%.h5'))
    return model


"""Set up classes information"""
activity_dict = ['Adjust control panel', 'Drinking', 'Eating', 'Hair_makeup',
                 'Hand on head', 'Normal Forward Driving', 'Phone Call(left)',
                 'Phone Call(right)', 'Pick up from floor (Driver)', 'Pick up from floor (Passenger)',
                 'Reaching behind', 'Singing with music', 'Talk to passenger at backseat',
                 'Talk to passenger at the right', 'Text (Left)', 'Text (Right)', 'shaking or dancing with music', 'yawning']

target_width = 128
target_height = 128

def rescale_input(x):
    x = x.astype('float32')
    x /= 255.0
    return x

def preprocess(image, target_dim=(128,128)):
    img = image.copy()
    img = cv2.resize(img, target_dim)
    img = rescale_input(img)
    img = np.expand_dims(img, 0)
    return img

def extractVideo(vid_path_in, vid_path_out, model):
    vidcap = cv2.VideoCapture(vid_path_in)
    if (vidcap.isOpened() == False):
        print("Error reading video file")
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    out = cv2.VideoWriter(vid_path_out,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             10, (frame_width, frame_height))
    success, frame = vidcap.read()
    success = True
    while success:
        success, frame = vidcap.read()
        preprocess_frame = preprocess(frame)
        prediction = model.predict(preprocess_frame)
        idx = np.argmax(prediction)
        label = activity_dict[idx]
        cv2.putText(frame, label, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 224, 127), 2)
        out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vidcap.release()
    cv2.destroyAllWindows()

def predict_once_per_2secs(vid_path_in, vid_path_out, model):
    vidcap = cv2.VideoCapture(vid_path_in)
    if (vidcap.isOpened() == False):
        print("Error reading video file")
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    out = cv2.VideoWriter(vid_path_out,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          30, (frame_width, frame_height))
    success = True
    count = 0
    predictions = []
    label = ''
    while success:
        success, frame = vidcap.read()
        preprocess_frame = preprocess(frame)
        prediction = model.predict(preprocess_frame)
        predictions.append(prediction)
        count += 1
        if count % 60 == 0:
            pred = np.sum(predictions, axis=0) / 60
            idx = np.argmax(pred)
            label = activity_dict[idx]
            count = 0
            predictions.clear()
        cv2.putText(frame, label, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 224, 127), 2)
        out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    mobileNet_model = load_model()
    path_in = './video/Dashboard_user_id_72519_NoAudio_2_1.MP4'
    path_out = './video/Dashboard_user_id_72519_NoAudio_2_extracted_1.mp4'
    extractVideo(path_in, path_out, mobileNet_model)


