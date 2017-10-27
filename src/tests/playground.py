import path_magic
import unittest
import os
from function_space import FunctionSpace
from mesh import CrazyMesh
import numpy as np
import numpy.testing as npt
from mesh import CrazyMesh
from forms import Form, ExtGaussForm_0
import matplotlib.pyplot as plt
import time


def ffun(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


lhs = np.random.random((10000, 10000))
rhs = np.random.random(10000)
t0 = time.time()
solution = np.linalg.solve(lhs, rhs)
t1 = time.time()
print("time :", t1 - t0)
