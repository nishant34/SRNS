import numpy as  np
import torch
import cv2
from common import *
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
    for i in range(1,w-1):
      for j in range(1,h-1):
        t = np.array([i,j-1,d_im[j-1,i,0]],dtype="float64")
        f = np.array([i-1,j,d_im[j,i-1,0]],dtype="float64")
        c = np.array([i,j,d_im[j,i,0]] , dtype = "float64")
        #d = np.cross(f-c,t-c)
        d = torch.cross(f-c,t-c)
        #n = d / np.sqrt((np.sum(d**2)))
        n = F.normalize(d,dim=-1)
        normals[j,i,:] = n
    return normals*255

def depth_map_to_cam_coordinates(x_im,y_im,z,intrinsics):
    fx,fy,cx,cy = get_intrinsic_params(intrinsics)
    x_cam  = ((x_im-cx)*z)/fx
    y_cam  =  ((y_im-cy)*z)/fy
    
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

    camera_coordinates = torch.inverse(extrinsic_matrix)*.bmm(homogenous_points)

    depth = camera_coordinates[:,:,2]
    depth = depth[:,:,None]

    return depth



    