#    Class type        Activity Label
#      0           Normal Forward Driving
#      1           Drinking
#      2           Phone Call(right)
#      3           Phone Call(left)
#      4           Eating
#      5           Text (Right)
#      6           Text (Left)
#      7           Hair_makeup
#      8           Reaching behind
#      9           Adjust control panel
#      10          Pick up from floor (Driver)
#      11          Pick up from floor (Passenger)
#      12          Talk to passenger at the right
#      13          Talk to passenger at backseat
#      14          yawning
#      15          Hand on head
#      16          Singing with music
#      17          shaking or dancing with music

# Import libraries
import os
import cv2

# Import Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create directory for 18 classes
def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")

classtype = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"]
for i in range(0, 18):
  path = "/content/drive/MyDrive/AIC2022/Dataset2/Class" + classtype[i]
  create_dir(path)

# Set FPS (Default 30 FPS, total frame = 30 / gap)
gap = 6

# Set MP4 file
vidcap = cv2.VideoCapture("/content/drive/MyDrive/AIC2022/2022/A2/user_id_42271/Dashboard_user_id_42271_NoAudio_3.MP4")

# Set classtype and the time range (ms) from video
classtype = ["3", "6", "2", "14", "4", "5", "17", "16", "9", "15", "11", "13", "8", "1", "10", "0", "12", "7"]
start_time_ms = [8,  41, 69, 96,  130, 157, 189, 214, 248, 282, 314, 341, 369, 395, 427, 455, 483, 515]
stop_time_ms  = [31, 59, 86, 120, 147, 179, 204, 238, 272, 304, 331, 359, 385, 417, 445, 473, 505, 530]

# Get frames from video
for i in range(0, 18):
	path = "/content/drive/MyDrive/AIC2022/Dataset6/A1/Class" + classtype[i]
	start_time_ms[i] = start_time_ms[i] * 1000
	stop_time_ms[i] = stop_time_ms[i] * 1000

	count = 0
	success = True
	vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms[i])
	while success and vidcap.get(cv2.CAP_PROP_POS_MSEC) <= stop_time_ms[i]:
	    success, image = vidcap.read()
	    print('Read a new frame: ', success)

	    if success == False:
	        vidcap.release()
	        break

	    if count == 0:
	        cv2.imwrite((path + "/frame%d.png") % (count / gap), image)
	    else:
	    	if count % gap == 0:
	        	cv2.imwrite((path + "/frame%d.png") % (count / gap), image)

	    count += 1
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break