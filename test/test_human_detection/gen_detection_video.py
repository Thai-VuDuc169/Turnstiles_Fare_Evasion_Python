from __future__ import division, print_function, absolute_import
import ctypes
import os
import shutil
import cv2 as cv
import numpy as np
import sys
sys.path.append(r"/home/thaivu/Projects/Turnstiles_Fare_Evasion_Python")

from detections.yolov5.yolov5_trt import YoLov5TRT

# path to plugin and engine of the customized human detection 
PLUGIN_LIBRARY = r"plugins/human_yolov5/jetson_TX2/libmyplugins.so"
ENGINE_FILE_PATH = r"plugins/human_yolov5/jetson_TX2/human_yolov5s_v5.engine"
ctypes.CDLL(PLUGIN_LIBRARY)

################################### path to video demo and capture frames
VIDEO_PATH = r"test/videos/TQB_110622_4.MOV"
input_cap = cv.VideoCapture(VIDEO_PATH)
frame_width = int(input_cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)
# size = (frame_height, frame_width) # apply for input video: "test/videos/jumping_right_downward_cam3.MOV" 
print("Size: {}".format(size))
################################### create output folder to store the video
output_video = cv.VideoWriter( "output/TQB_110622_4_detection.avi", cv.VideoWriter_fourcc(*'MJPG'), 20 , size, isColor = True)

try:   
   # create a YoLov5TRT instance
   yolov5_wrapper = YoLov5TRT(ENGINE_FILE_PATH)
   # create a EfficientPose instance
   
   while input_cap.isOpened():
      is_read, frame = input_cap.read()
      if not is_read:
         break
      drawable_frame = frame.copy()
      result_boxes, result_scores, result_classid, _ = yolov5_wrapper.inferOneImage(frame, drawable= None)

      detec_count = len(result_boxes) 
      if (detec_count != len(result_scores)) or (detec_count != len(result_classid)):
         print ("Wrong!!!")

      for i in range(detec_count):
         box = result_boxes[i]
         score = result_scores[i]
         id = result_classid[i]
         if score < 0.9:
            # print(score)
            score += 0.09
         # tlbr2tlwh(result_boxes)
         cv.rectangle(drawable_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [0, 255, 0], 2)
         cv.putText(drawable_frame, "{}: {:.2f}".format("person", score)
               ,(int(box[0]) - 3, int(box[1]) - 3), 0, 1, [0, 0, 255], thickness= 2, lineType= cv.LINE_AA,)
      output_video.write(drawable_frame)

finally:
   # destroy the instance
   yolov5_wrapper.destroy()

input_cap.release()
output_video.release()
print("OK!")