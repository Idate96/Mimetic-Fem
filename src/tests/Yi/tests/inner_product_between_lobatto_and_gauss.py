# -*- coding: utf-8 -*-
"""
(SHORT NAME EXPLANATION)
    
>>>DOCTEST COMMANDS 
(THE TEST ANSWER)

@author: Yi Zhang. Created on Mon Jul 10 20:12:27 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
         
#SUMMARY----------------

#INPUTS-----------------
    #ESSENTIAL:
    #OPTIONAL:

#OUTPUTS----------------

#EXAMPLES---------------

#NOTES------------------
"""

# -*- coding: utf-8 -*-
"""
(SHORT NAME EXPLANATION)
    
>>>DOCTEST COMMANDS 
(THE TEST ANSWER)

@author: Yi Zhang （张仪）. Created on Thu Jul  6 16:00:33 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
         
#SUMMARY----------------

#INPUTS-----------------
    #ESSENTIAL:
    #OPTIONAL:

#OUTPUTS----------------

#EXAMPLES---------------

#NOTES------------------
"""
from function_space import FunctionSpace
import numpy as np
from mesh import CrazyMesh
from forms import Form
from hodge import hodge
from coboundaries import d
from assemble import assemble
from _assembling import assemble_, integral1d_
import matplotlib.pyplot as plt
from quadrature import extended_gauss_quad
from scipy.integrate import quad
from sympy import Matrix
import scipy.io
from scipy import sparse
import scipy as sp
from inner_product import inner

# %% exact solution define
# u^{(1)} = { u,  v }^T
def u(x,y):
	return   +np.cos(np.pi*x) * np.sin(np.pi*y)

def v(x,y):
	return   -np.sin(np.pi*x) * np.cos(np.pi*y)

def r_u(x,y):
    return   -2* np.pi**2 * np.cos(np.pi*x) * np.sin(np.pi*y)

def r_v(x,y):
    return    2* np.pi**2 * np.sin(np.pi*x) * np.cos(np.pi*y)

# %% define the mesh
mesh = CrazyMesh( 2, (2, 2), ((-1, 1), (-1, 1)), 0.05 )
func_space_gauss1   = FunctionSpace(mesh, '1-gauss', (5, 5), is_inner=False)
func_space_lobatto1 = FunctionSpace(mesh, '1-lobatto', (5, 5), is_inner=False)

form_1_gauss   = Form(func_space_gauss1)
form_1_lobatto = Form(func_space_lobatto1)

M = inner(form_1_lobatto.basis,form_1_gauss.basis)
