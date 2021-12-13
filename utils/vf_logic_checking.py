import numpy as np
import cv2 as cv

def tlbr2tlwh(bbox):
   if bbox.ndim < 2:
      bbox = np.atleast_2d(bbox)
   bbox[:, 2:] -= bbox[:, :2]

def cropImage(img, bbox):
   """Crop a part of image with given bbox
   Parameters
   ----------
      img : cv.Mat
         Input image
      bbox : ndarray 
         The bounding box (tlbr format)
   Returm
   ----------
      mat : cv.Mat
         a cropped output image  
   """
   return img[bbox[1] : bbox[3], bbox[0] : bbox[2]]

class VirtualFence:
   
   def __init__(self, A_point, B_point):
      """Initialize a virtual fence between A and B
      Parameters
      ----------
         A_point: a tuple (x, y) is coor of A point
         B_point: a tuple (x, y) is coor of B point
      """
      self.A_point = A_point   
      self.B_point = B_point
      self.remainder = self.A_point[0] * self.B_point[1] - self.B_point[0] * self.A_point[1]
      
   def drawLine(self, frame, color):
      cv.line(frame, self.A_point, self.B_point, color, thickness= 4)
      
   def updateCurrentStates(self, tracks):
      """Update current state of each track in tracks (on which side of the virtual fence?)
      Parameters
      ----------
         tracks : List[Track]
            The list of active tracks at the current time step.  
      """
      for track in tracks:
         bbox = track.to_tlbr()
         xy_center = (int((bbox[0] + bbox[2]) / 2), 
                        int((bbox[1] + bbox[3]) / 2))
         track.current_state = self.checkSide(xy_center)

   
   def checkSide(self, xy_center):
      """Check which side of the virtual fence the current point is on
      Parameters
      ----------
         xy_center : a tuple (x, y)
            (x, y) is the coordinate in the center of the bounding box
      Return
      ----------
         1 : if the point at the side of the increasing vertical axes
         -1 :  otherwise
      """
      # tan_center_A = (xy_center[0] - self.A_point[0]) / (xy_center[1] - self.A_point[1])
      # tan_line = (self.B_point[0] - self.A_point[0]) / (self.B_point[1] - self.A_point[1]) 
      left_side = (self.A_point[0] - self.B_point[0]) * xy_center[1]
      right_side = (self.A_point[1] - self.B_point[1]) * xy_center[0] + self.remainder
      return 1 if (left_side - right_side) > 0 else -1

