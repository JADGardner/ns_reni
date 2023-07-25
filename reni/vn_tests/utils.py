import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as Rotation

def rand_rotation_tensor():
  R = Rotation.from_rotvec(Rotation.random().as_euler('zxy', degrees=True))
  R = torch.tensor(R.as_matrix()).to(torch.float32)
  return R

def test_close_eps(a, b, eps=1e-7):
    return (abs(a-b)<eps).all()

def test_close(a, b):
    # loop through from 1e-1 to 1e-1 and 
    # return (true, eps) with the smallest epsilon that makes the test pass
    # else return (False, None)
    for i in range(15, 1, -1):
        eps = 10**(-i)
        if test_close_eps(a, b, eps):
            return True, eps
    return False, None
