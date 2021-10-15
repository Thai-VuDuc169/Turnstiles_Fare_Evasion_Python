from __future__ import division, print_function, absolute_import
import sys
import ctypes
import os
import shutil
import threading
import cv2 as cv
import numpy as np
sys.path.append(r"/home/thaivu/Projects/Turnstiles_Fare_Evasion_Python")

from detections.yolov5.yolov5_trt import YoLov5TRT

from tracking.deep_sort import nn_matching
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker

from utils import generate_detections as gdet
from utils.preprocessing import tlbr2tlwh, cropImage, VirtualFence

################################### path to plugin and engine of the customized human detection 
PLUGIN_LIBRARY = r"plugins/human_yolov5_plugin/libmyplugins.so"
ENGINE_FILE_PATH = r"plugins/human_yolov5_plugin/yolov5s_custom.engine"
ctypes.CDLL(PLUGIN_LIBRARY)

################################### path to a model of traking and initialize a tracker
NN_BUDGET = None
MAX_COSINE_DISTANCE = 0.3
TRACKING_MODEL = r"models/deep_sort_model/mars-small128.pb"
encoder = gdet.create_box_encoder(TRACKING_MODEL, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
tracker = Tracker(metric)

################################### declare a instance of VirtualFence
POINTS = ((100, 100), (300, 300)) # A and B
virtual_fence = VirtualFence(POINTS[0], POINTS[1])

################################### path to folder containing cropped images
if os.path.exists('output/'):
   shutil.rmtree('output/')
os.makedirs('output/')
CROPPED_IMAGES_PATH = r"/home/thaivu/Projects/Turnstiles_Fare_Evasion_Python/output"

################################### path to video demo and capture frames
VIDEO_PATH = r"/home/thaivu/Projects/Turnstiles_Fare_Evasion_Python/videos/test_tracking_human1.mp4"
input_cap = cv.VideoCapture(VIDEO_PATH)
frame_width = int(input_cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)
print("Size: {}".format(size))
################################### create output folder to store the video
# output_video = cv.VideoWriter( "output/detection_demo.avi", cv.VideoWriter_fourcc(*'MJPG'), 20 , size, isColor = True)

# color for tracked objects 
COLOR = [(255, 255, 0), (204, 0, 153), (26, 209, 255), (71, 107, 107)]

try:   
   # create a YoLov5TRT instance
   yolov5_wrapper = YoLov5TRT(ENGINE_FILE_PATH)
   # while input_cap.isOpened():
   for i in range(20):
      is_read, frame = input_cap.read()
      if not is_read:
         break
      result_boxes, result_scores, result_classid, _ = yolov5_wrapper.inferOneImage(frame, drawable= None)
      tlbr2tlwh(result_boxes)
      detect_box = np.copy(result_boxes)
      # 1 Detection gồm (tlwh, conf, feature)
      # frame đầu vào ở đây là ảnh chưa được xử lý (frame sau khi xử lý không bị thay đổi), boxes định dạng tlwh
      features = encoder(frame, boxes= result_boxes)
      detections = [Detection(box, confidence, feature)
                     for box, confidence, feature 
                     in zip(detect_box, result_scores, features)]
      # Update tracker
      tracker.predict()
      tracker.update(detections)
      # Update current state of each track in tracks
      virtual_fence.updateCurrentStates(tracker.tracks)
      for track in tracker.tracks:
         if not track.is_confirmed() or track.time_since_update > 1:
            continue

         if track.pre_state == None:
            track.pre_state = track.current_state
            continue
         if track.pre_state == track.current_state:
            continue
         else:
            temp_img = cropImage(frame, track.to_tlbr())
            cv.imwrite(CROPPED_IMAGES_PATH, temp_img)
            print("image is saved!")
            track.pre_state = track.current_state
            continue
         # track_box = track.to_tlbr()
         # cv.rectangle(frame, (int(track_box[0]), int(track_box[1])), (int(track_box[2]), int(track_box[3])),
         #                COLOR[track.track_id%3], 6)
         # cv.putText(frame,"ID: " + str(track.track_id), (int(track_box[0]), int(track_box[1])), 0, 2, (0, 0, 255), 5)
      # output_video.write(frame)

finally:
   # destroy the instance
   yolov5_wrapper.destroy()

input_cap.release()
# output_video.release()
print("OK!")