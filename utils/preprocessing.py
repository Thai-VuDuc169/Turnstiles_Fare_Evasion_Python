import numpy as np

def tlbr2tlwh(bbox):
   bbox[:, 2:] -= bbox[:, :2]
