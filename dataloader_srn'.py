
#do for few shhoots also


import torch
from torch.utils.data import Dataset,dataloader
import os
from glob import glob
import utils
from PIL import Image
import numpy as np

class object1(Dataset):
    #for a particular object
    def __init__(self,data_dir,image_length,instance_idx):
        self.instance_category = instance_idx
        self.obj_dir = data_dir
        self.extrinsic_dir = os.path.join(data_dir,"pose")
        self.image_dir  = os.path.join(data_dir,"rgb")
        self.intrinsics_dir = os.path.join(data_dir,"intrinsics")
        self.image_length  = image_length 
       
        if not os.isdir(self.image_dir):
            print("wrong dir")
            return 

        self.extrinsic_list = [x for x in glob.glob(os.path.join(self.extrinsic_dir,'*.txt'))]
        self.image_list = [x for x in glob.glob(os.path.join(self.image_dir,'*.txt'))]
        self.intrinsic_list = [x for x in glob.glob(os.path.join(self.intrinsics_dir,'*.txt'))]
        self.intrinsic_params = [] 
        for file_name in self.intrinsic_list:
          with open(file_name) as f:
             array = []
             for line in f:
               array.append([float(x) for x in line.split()])
             array_1 = np.reshape(array,(-1,1))
             self.intrinsic_params.append(array)
        self.extrinsic_params = [] 
        for file_name in self.extrinsic_list:
          with open(file_name) as f:
             array = []
             for line in f:
               array.append([float(x) for x in line.split()])
             array_1 = np.reshape(array,(-1,1))
             self.extrinsic_params.append(array)
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,index):
       rgb_image  = Image.open(os.path.join(self.image_dir,self.image_list(index)))
       image_final = adjustedresolution(np.array(rgb_image),self.image_length)
       intrinsics = self.intrinsic_params[index]
       extrinsics = self.extrinsic_params[index]
       
       return image_final,intrinsics,extrinsics

    def image_resolution(self,image_length):
      self.image_length = image_length

class class1(Dataset):

    def __init__(self,data_dir,number_of_instances,image_sidelength):
      #self.num_examples_per_instance = examples_per_instance

       self.object_list = []
       for dir_name  in os.listdir(data_dir):
           object_path = os.path.join(data_dir,dir_name)
           curr_object = object1(object_path)
           self.object_lists.append(curr_object)
       
       self.num_objects = len(self.object_lists)           
    def __len__(self):
      count = 0
      for obj in self.object_lists:
        count+=len(obj)
      return count
            
    def image_resolution(self,Image_length):
      for obj in self.object_lists:
        obj.image_resolution(Image_length)   
    def __getitem__(self,obj_index):
           curr_object_list = self.object_lists[index]
           x =[]
           for i in range(self.num_samples_per_instance):
               x.append(curr_object_list[np.random.randint(len(curr_object_list))])
           

           return x

        
    #def collate_fn(slef,batch_list):


        