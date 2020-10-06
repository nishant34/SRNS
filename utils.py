import numpy as  np
import torch
import cv2

import torch.functional.nn as F

from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def parse_intrinsics(file_path,trgt_sidelength=None):
    txt_file = open(file_path,'r')
    lines = txt_file.readline()
    f = float(lines.split()[0])
    cx  =float(lines.split()[1])
    cy  =float(lines.split()[2])
    
    line2  = txt_file.readline()
    bary_center = torch.tensor(list(map(float,line2.split())))
    
    scale = float(txt_file.readline())
    
    line4 = txt_file.readline()
    h = float(line4.split()[0])
    w  = float(line4.split()[1])
    
    if trgt_sidelienth is not None:
        cx = cx/width*trgt_sidelength
        cy  = cy/height*trgt_sidelength
        f = trget_sidelength/height*f
    intrinsic_matrix = np.array([[f, 0., cx, 0.],
                               [0., f, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return intrinsic_matrix,grid_barycenter,scale

def straight_to_2D_image(tensor):
    batches,num_samples,channels = tensor.shape()
    image_length = np.sqrt(num_samples).astype(int)

    final_2d_image = tensor.permute(0,2,1).view(batch_size,image_length,image_length)

    return final_2d_image


     




def show_images(images,titles=None):
     n_images = len(images)

     n_cols = np.ceil(np.sqrt(n_images))
     n_rows = n_images/n_cols
     fig,ax  =plt.subplots(n_rows,n_cols,figsize = (12,4))
     count = 0
     for i in range(n_rows):
         for j in range(n_cols):
             im = ax[i][j].imshow(images[count])
             count+=1
             ax[i][j].axes.get_xaxis().set_visible(False)
             ax[i][j].axes.get_yaxis().set_visible(False)
     
     plt.tight_layout()         
     return fig


def get_extrinsic_matrix(file_name):

    txt_file = open(file_name)
    lines = txt_file.readline()
    lines_arr = lines.split()
    a =  np.random.randn(4,4) 
    for i in range(1,5):
        for j in range(1,5):
          a[i-1][j-1]  = float(lines_arr[i*4+j-1])


    return torch.from_numpy(a)



    