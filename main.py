from __future__ import division, print_function, absolute_import
import ctypes
import os
import shutil
import cv2 as cv
import numpy as np
import time

from detections.yolov5.yolov5_trt import YoLov5TRT

from tracking.deep_sort import nn_matching
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker

from utils.deep_sort import generate_detections as gdet
from utils.vf_logic_checking import tlbr2tlwh, cropImage, VirtualFence

from pose_estimations.efficient_pose.trt_model import EfficientPose

################################### path to plugin and engine of the customized human detection 
PLUGIN_LIBRARY = r"plugins/human_yolov5/quardo_p4000_5s/libmyplugins.so"
ENGINE_FILE_PATH = r"plugins/human_yolov5/quardo_p4000_5s/yolov5s_v5.engine"
ctypes.CDLL(PLUGIN_LIBRARY)

################################### path to a model of traking and initialize a tracker
NN_BUDGET = None
MAX_COSINE_DISTANCE = 0.3
TRACKING_MODEL = r"models/deep_sort/mars-small128.pb"
encoder = gdet.create_box_encoder(TRACKING_MODEL, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
tracker = Tracker(metric)

################################### declare a instance of VirtualFence
# POINTS = ((431, 1092), (747, 929)) # A and B for cam 3
# POINTS = ((260, 1099), (810, 1098)) # A and B for cam 2
# POINTS = ((229, 345), (1533, 663)) # A and B for TQB video (1)
POINTS =   ((378, 489), (1444, 687))
FORWARD_FLOW = 1     # =0 if (-) -> (+); =1 if (+) -> (-)
VF_OFFSET = 25 # pixels
virtual_fence = VirtualFence(POINTS[0], POINTS[1], VF_OFFSET)

################################### path to folder containing cropped images
if os.path.exists(r'output/'):
   shutil.rmtree(r'output/')
os.makedirs(r'output/')
os.makedirs(r'output/bin_imgs')
CROPPED_IMAGES_PATH = r"output/"
BIN_IMAGES_PATH = r"output/bin_imgs"

################################### path to video demo and capture frames
VIDEO_PATH = r"test/videos/TQB_110622_3.MOV"
input_cap = cv.VideoCapture(VIDEO_PATH)
frame_width = int(input_cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)
# size = (frame_height, frame_width) # apply for input video: "test/videos/jumping_right_downward_cam3.MOV" 
print("Size: {}".format(size))
################################### create output folder to store the video
output_video = cv.VideoWriter( "output/TQB_110622_3_result.avi", cv.VideoWriter_fourcc(*'MJPG'), 20 , size, isColor = True)

sum_detec_time = 0
sum_tracking_time = 0
sum_genkpts_time = 0
sum_svm_time = 0
frame_count = 0

try:   
   # create a YoLov5TRT instance
   yolov5_wrapper = YoLov5TRT(ENGINE_FILE_PATH)
   # create a EfficientPose instance
   efficient_pose = EfficientPose(folder_path= r"output/bin_imgs", model_name= "RT", 
                  svm_clf_path= r"models/svm_clf/RT_SVM_skl102_v2.sav")

   global_st = time.time()
   while input_cap.isOpened():
      is_read, frame = input_cap.read()
      if not is_read:
         break
      frame_count += 1
      drawable_frame = frame.copy()

      detec_st = time.time()
      result_boxes, result_scores, result_classid, _ = yolov5_wrapper.inferOneImage(frame, drawable= None)
      # detec_et = time.time()
      sum_detec_time += (time.time() - detec_st)

      if len(result_boxes) == 0:
         print ("frame_count: " + str(frame_count) + " : result_boxes is empty!")
         continue

      tlbr2tlwh(result_boxes)
      detect_box = np.copy(result_boxes)
      # 1 Detection g???m (tlwh, conf, feature)
      # frame ?????u v??o ??? ????y l?? ???nh ch??a ???????c x??? l?? (frame sau khi x??? l?? kh??ng b??? thay ?????i), boxes ?????nh d???ng tlwh
      tracking_st = time.time()
      features = encoder(frame, boxes= result_boxes)
      detections = [Detection(box, confidence, feature)
                     for box, confidence, feature 
                     in zip(detect_box, result_scores, features)]
      # Update tracker
      tracker.predict()
      tracker.update(detections)
      # tracking_et = time.time()
      sum_tracking_time += (time.time() - tracking_st)

      # Update current state of each track in tracks
      virtual_fence.updateCurrentStates(tracker.tracks)
      virtual_fence.drawLine(drawable_frame, (225, 235, 25))

      
      for track in tracker.tracks:
         if not track.is_confirmed() or track.time_since_update > 1:
            continue
         
         # draw confirmed tracks 
         track_box = track.to_tlbr()
         cv.rectangle(drawable_frame, (int(track_box[0]), int(track_box[1])), (int(track_box[2]), int(track_box[3])), [0, 255, 0], 2)
         cv.putText(drawable_frame,"ID: " + str(track.track_id), (int(track_box[0]) + 9, int(track_box[1]) - 5), 0, 1, (0, 153, 255), 2)

         if track.pre_state == None:
            track.pre_state = track.current_state
            continue
         if track.pre_state == track.current_state:
            continue

         if ((not FORWARD_FLOW and track.pre_state == 1 and track.current_state == -1) \
               or (FORWARD_FLOW and track.pre_state == -1 and track.current_state == 1)):
            cv.rectangle(drawable_frame, (int(track_box[0]), int(track_box[1])), (int(track_box[2]), int(track_box[3])), [0, 255, 255], 4)
            cv.putText(drawable_frame, "Warning: backward flow", (int(track_box[0]) + 9, int(track_box[1]) + 27), 0, 1, [0, 255, 255], thickness= 2, lineType= cv.LINE_AA,)
            track.pre_state = track.current_state
            continue
         else:
            offset = 27
            h_start = max(0, int(track_box[1]) - offset)
            h_end = min(int(track_box[3]) + offset, frame.shape[0])
            w_start = max(0, int(track_box[0]) - offset)
            w_end =  min(int(track_box[2]) + offset, frame.shape[1])
            temp_img = frame[h_start : h_end, w_start : w_end, :]
            # label = efficient_pose.animatePose(temp_img, drawable_frame, track_box, offset, segment_width= int(frame.shape[0]/80), marker_radius= int(frame.shape[0]/80))
            label, svm_dura, gen_kpts_dura = efficient_pose.animatePoseWithSVM (temp_img, drawable_frame, track_box, offset, segment_width= int(frame.shape[0]/80), marker_radius= int(frame.shape[0]/80))
            sum_genkpts_time += gen_kpts_dura
            sum_svm_time += svm_dura
            
            if label == "walking":
               cv.rectangle(drawable_frame, (int(track_box[0]), int(track_box[1])), (int(track_box[2]), int(track_box[3])), [174, 134, 22], 4)
               cv.putText(drawable_frame, label , (int(track_box[0]) + 9, int(track_box[1]) + 27), 0, 1, [174, 134, 22], thickness= 2, lineType= cv.LINE_AA,)
            else:
               cv.rectangle(drawable_frame, (int(track_box[0]), int(track_box[1])), (int(track_box[2]), int(track_box[3])), [0, 0, 255], 4)
               cv.putText(drawable_frame, label , (int(track_box[0]) + 9, int(track_box[1]) + 27), 0, 1, [0, 0, 255], thickness= 2, lineType= cv.LINE_AA,)
            track.pre_state = track.current_state
            continue
      output_video.write(drawable_frame)
   global_et = time.time()

   print ("===== Execution Time =====")
   print ("--> global exec dura: " + str((global_et - global_st) * 1000) + " ms")
   print ("--> detect dura: " + str((sum_detec_time / frame_count) * 1000) + " ms")
   print ("--> tracking dura: " + str((sum_tracking_time / frame_count) * 1000) + " ms")
   print ("--> gen kpts dura: " + str((sum_genkpts_time / frame_count) * 1000) + " ms")
   print ("--> svm dura: " + str((sum_svm_time / frame_count) * 1000) + " ms")
   print ("--> frame count: " + str(frame_count))
      

finally:
   # destroy the instance
   yolov5_wrapper.destroy()
   efficient_pose.destroy()

input_cap.release()
output_video.release()
print("OK!")