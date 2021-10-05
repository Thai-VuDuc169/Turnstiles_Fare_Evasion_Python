import ctypes
import os
import shutil
import sys
import threading
from detections.yolov5.yolov5_trt import YoLov5TRT, get_img_path_batches, plot_one_box, inferThread, warmUpThread

# load custom plugin and engine
PLUGIN_LIBRARY = r"plugins/human_yolov5_plugin/libmyplugins.so"
ENGINE_FILE_PATH = r"plugins/human_yolov5_plugin/yolov5s_custom.engine"

# if len(sys.argv) > 1:
#    ENGINE_FILE_PATH = sys.argv[1]
# if len(sys.argv) > 2:
#    PLUGIN_LIBRARY = sys.argv[2]

ctypes.CDLL(PLUGIN_LIBRARY)

if os.path.exists('output/'):
   shutil.rmtree('output/')
os.makedirs('output/')
# a YoLov5TRT instance
yolov5_wrapper = YoLov5TRT(ENGINE_FILE_PATH)
try:
   print('batch size is', yolov5_wrapper.batch_size)
   
   image_dir = "/home/thaivu/Projects/TestImages"
   image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size, image_dir)

   for i in range(10):
      # create a new thread to do warm_up
      thread1 = warmUpThread(yolov5_wrapper)
      thread1.start()
      thread1.join()
   for batch in image_path_batches:
      # create a new thread to do inference
      thread1 = inferThread(yolov5_wrapper, batch)
      thread1.start()
      thread1.join()
finally:
   # destroy the instance
   yolov5_wrapper.destroy()