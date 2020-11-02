import numpy as  np
import torch
import torch.nn.functional as F
import cv2
from comon import *
#def get_intrinsic(file_name,image_length):
    
 #   text_file = open('file_name','r') 


def world_coordinate_generator(R,K,t,u,v,d):
    vect = np.array([u,v,d])
    a = R.T
    b = np.linalg.inv(K)
    ans = a*(np.matmul(b,vect)-t)
    return ans



def get_normal_map_from_depth_map(d_im):
    h,w,d = d_im.shape
    normals = torch.zeros(h,w,d)
    for i in range(1,w-1):
      for j in range(1,h-1):
        t = np.array([i,j-1,d_im[j-1,i,0]],dtype="float64")
        f = np.array([i-1,j,d_im[j,i-1,0]],dtype="float64")
        c = np.array([i,j,d_im[j,i,0]] , dtype = "float64")
        #d = np.cross(f-c,t-c)
        d = torch.cross(torch.tensor(f-c),torch.tensor(t-c))
        #print("cross_passed")
        #n = d / np.sqrt((np.sum(d**2)))
        n = F.normalize(d,dim=-1)
        normals[j,i,:] = n
    return normals*255

def depth_map_to_cam_coordinates(x_im,y_im,z,intrinsics):
    fx,fy,cx,cy = get_intrinsic_params(intrinsics)
    #print(fx.shape)
    #print(fy.shape)
    #print(cx.shape)
    #print(cy.shape)
    #print(x_im.shape)
    #print(y_im.shape)
    #print(z.shape)
    num_pixels = x_im.shape[1]
    #x_cam = torch.zeros(x_im.shape[0],num_pixels)
    #y_cam = torch.zeros(x_im.shape[0],num_pixels)
    cx_1 = cx[:,None]
    cy_1 = cy[:,None]
    fx_1 = fx[:,None]
    fy_1 = fy[:,None]
    

    cx_1 = cx_1.expand(4,num_pixels)
    cy_1 = cy_1.expand(4,num_pixels)
    fx_1 = fx_1.expand(4,num_pixels)
    fy_1 = fy_1.expand(4,num_pixels)
    cx_1  = cx_1.to(device)
    cy_1 =  cy_1.to(device)
    fx_1  = cx_1.to(device)
    fy_1 =  cy_1.to(device)
    x_im = x_im.to(device)
    y_im = y_im.to(device)
    x_cam  = ((x_im-cx_1)*z)/fx_1
    y_cam  =  ((y_im-cy_1)*z)/fy_1
    x_cam  = x_cam.float()
    y_cam =  y_cam.float()
    #print("barrier_passed")
    return torch.stack((x_cam,y_cam,z),dim=-1)
    




def get_intrinsic_params(intrinsic_matrix):
    fx =  intrinsic_matrix[:,0,0]
    fy = intrinsic_matrix[:,1,1]
    cx = intrinsic_matrix[:,0,2]
    cy =  intrinsic_matrix[:,1,2]
    return fx,fy,cx,cy



def pixel_depth_to_world_coordinates(xy_grid,depth_map,intirnsics,extrinsics):
    x_cam =  xy_grid[:,:,0].view(batch_size,-1)
    y_cam = xy_grid[:,:,1].view(batch_size,-1)
    z_cam = depth_map.view(batch_size,-1)

    camera_3d_coordinates = depth_map_to_cam_coordinates(x_cam,y_cam,z_cam,intrinsics)

    world_coordinates =  extrinsics*camera_3d_coordinates

    return world_coordinates


def pixels_form_camera(x,y,z,intrinsics):
    fx,fy,cx,cy = get_intrinsic_params(instrinsics)

    u  = (x*fx)/z  + cx
    v =  (y*fy)/z  + cy


    return torch.stack((u,v,z),dim=-1)



def get_depths_from_world_coordinates(x,y,z,extrinsic_matrix):
    batch_size = x.shape[0]

    homogenous_points =  torch.cat((x,y,z,torch.ones(x).cuda()),dim=-1)

    #camera_coordinates = torch.inverse(extrinsic_matrix)*.bmm(homogenous_points)

    depth = camera_coordinates[:,:,2]
    depth = depth[:,:,None]

    return depth



    
