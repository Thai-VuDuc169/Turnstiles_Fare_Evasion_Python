import cv2 as cv
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# refer: https://github.com/tranleanh/data-augmentation/blob/main/data_augment_tool.py
def colorjitter(img, cj_type="b"):
   '''
   ### Different Color Jitter ###
   img: image
   cj_type: {b: brightness, s: saturation, c: constast}
   '''
   if cj_type == "b":
      # value = random.randint(-50, 50)
      value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
      hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
      h, s, v = cv.split(hsv)
      if value >= 0:
         lim = 255 - value
         v[v > lim] = 255
         v[v <= lim] += value
      else:
         lim = np.absolute(value)
         v[v < lim] = 0
         v[v >= lim] -= np.absolute(value)

      final_hsv = cv.merge((h, s, v))
      img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
      return img
   
   elif cj_type == "s":
      # value = random.randint(-50, 50)
      value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
      hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
      h, s, v = cv.split(hsv)
      if value >= 0:
         lim = 255 - value
         s[s > lim] = 255
         s[s <= lim] += value
      else:
         lim = np.absolute(value)
         s[s < lim] = 0
         s[s >= lim] -= np.absolute(value)

      final_hsv = cv.merge((h, s, v))
      img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
      return img
   
   elif cj_type == "c":
      brightness = 10
      contrast = random.randint(40, 100)
      dummy = np.int16(img)
      dummy = dummy * (contrast/127+1) - contrast + brightness
      dummy = np.clip(dummy, 0, 255)
      img = np.uint8(dummy)
      return img

def cutout(img, gt_boxes, num_cuts= 2, factor= 0.3,):
   '''
   ### Cutout ###
   img: image
   gt_boxes: format [[obj x1 y1 x2 y2],...]
   num_cuts: the number of cuts 
   factor: factor for size of black boxes
   '''
   out = img.copy()
   for _ in range(num_cuts):
      for box in gt_boxes:
         x1 = int(box[1])
         y1 = int(box[2])
         x2 = int(box[3])
         y2 = int(box[4])
         mask_w = int( (x2 - x1) * factor )
         mask_h = int( (y2 - y1) * factor )
         mask_x1 = random.randint(x1, x2 - mask_w)
         mask_y1 = random.randint(y1, y2 - mask_h)
         mask_x2 = mask_x1 + mask_w
         mask_y2 = mask_y1 + mask_h
         cv.rectangle(out, (mask_x1, mask_y1), (mask_x2, mask_y2), (0, 0, 0), thickness=-1)
   return out

def noisy(img, noise_type="gauss"):
   '''
   ### Adding Noise ###
   img: image
   cj_type: {gauss: gaussian, sp: salt & pepper}
   '''
   if noise_type == "gauss":
      image=img.copy() 
      mean=0
      st=0.7
      gauss = np.random.normal(mean,st,image.shape)
      gauss = gauss.astype('uint8')
      image = cv.add(image,gauss)
      return image
   
   elif noise_type == "sp":
      image=img.copy() 
      prob = 0.05
      if len(image.shape) == 2:
         black = 0
         white = 255            
      else:
         colorspace = image.shape[2]
         if colorspace == 3:  # RGB
               black = np.array([0, 0, 0], dtype='uint8')
               white = np.array([255, 255, 255], dtype='uint8')
         else:  # RGBA
               black = np.array([0, 0, 0, 255], dtype='uint8')
               white = np.array([255, 255, 255, 255], dtype='uint8')
      probs = np.random.random(image.shape[:2])
      image[probs < (prob / 2)] = black
      image[probs > 1 - (prob / 2)] = white
      return image

def filters(img, f_type = "blur"):
   '''
   ### Filtering ###
   img: image
   f_type: {blur: blur, gaussian: gaussian, median: median}
   '''
   if f_type == "blur":
      image=img.copy()
      fsize = 9
      return cv.blur(image,(fsize,fsize))
   elif f_type == "gaussian":
      image=img.copy()
      fsize = 9
      return cv.GaussianBlur(image, (fsize, fsize), 0)
   elif f_type == "median":
      image=img.copy()
      fsize = 9
      return cv.medianBlur(image, fsize)

def augmentImages(img):
   yield "normal", img # be not transform
   yield "horiz_flip", cv.flip(img, 1) # flip horizontally
   yield "noised", noisy(img) # be not transform
   # yield "color_jitter", colorjitter(img, cj_type= "s")
   yield "cutout_n1", cutout(img, [[0, 13, 13, img.shape[1] - 13, img.shape[0] - 13]], num_cuts= 1, factor= 0.4)
   yield "cutout_n2", cutout(img, [[0, 0, 0, img.shape[1] - 1, img.shape[0] - 1]], num_cuts= 2, factor= 0.3)
   # yield "fat_resize", cv.resize(img, dsize= None, fx= 1.4, fy = (img.shape[1] * 1.4)/ img.shape[0])
