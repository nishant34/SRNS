import numpy as  np
import torch
import cv2
import torch.nn.functional as F
from comon import *
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def parse_intrinsics(file_path,targt_sidelength=None):
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
    
    if targt_sidelength is not None:
        cx = cx/width*targt_sidelength
        cy  = cy/height*targt_sidelength
        f = target_sidelength/height*f
    intrinsic_matrix = np.array([[f, 0., cx, 0.],
                               [0., f, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return intrinsic_matrix,bary_center,scale

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

#def convert_extrinsic_to_tensor(extrinsics):

def get_world_coordinates(R,K,t,x,d):
    #y1  =[]
    #y = []
    y = torch.zeros(batch_size,x.shape[1],x.shape[2],3)
    
    K = K.cpu()
    R  = R.cpu()
    #print("the typ of d is ")
    #print(type(d))
    print(x.shape)
    #R2 = torch.randn(batch_size,4,3)
    R2 = R[:,:,0:2]
    t = t.cpu()
    t = np.array(t)
    for k in range(x.shape[0]):
     for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            #print(j)
            curr = np.array([i,j,d])
            R1 = R2
            #curr_world_coordinates = np.matmul(np.array(R1),np.matmul((np.linalg.inv(np.array(K))),curr)-t)
            curr_world_coordinates  = curr
            y[k,i,j] = torch.tensor(curr)
            #y.ap   pend(curr_world_coordinates)
     #y_final1 = np.array(y)
     #y1.append(y_final1)
     #print(type(y1))
    #y_final = np.array(y1)
    #print(type(y_final))
    #y_final = np.float32(y_final)
    #print(type(y_final[0]))
    #y_final = torch.tensor(y_final,dtype = torch.float32)
    #y_final = y_final.to(device)
    y = y.to(device)
    return y


    



    
