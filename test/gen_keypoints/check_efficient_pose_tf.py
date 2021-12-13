import sys
import os
import cv2 as cv
import numpy as np
from PIL import Image
sys.path.append(r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python")


from pose_estimations.efficient_pose.tf_model import EfficientPose

IMAGE_FILE_NAME = r"thai_jumping4.jpg"

image = np.array(Image.open(os.path.join(r"test/images", IMAGE_FILE_NAME )))
print ("image.shape= " + str(image.shape))
print ("image[0,0,0]= " + str(image[0,0,0]))

efficient_pose = EfficientPose(folder_path= r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python/output/bin_imgs", model_name= "III")
print ("Result: " + efficient_pose.estimatePose(image, file_name= IMAGE_FILE_NAME))
# efficient_pose.performPoseEstimation(image, r"bin_image_test.png")
