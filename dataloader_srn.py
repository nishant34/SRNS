
#do for few shhoots also


import torch
from torch.utils.data import Dataset,DataLoader
import os
from glob import glob
#import utils
from PIL import Image
import numpy as np
import torch.nn.functional as F

def adjustedresolution(Image1,length1):
 if length1>Image1.shape[0]:
   print("change resolution")
   return 
 a = Image1.shape[0]
 top1 = (a-length1)/2
 top = int(top1)
 left = top 
 final_image = Image1[top:top+length1,left:left+length1]
 return final_image

class object1(Dataset):
    #for a particular object
    def __init__(self,data_dir,image_length,instance_idx):
        self.instance_category = instance_idx
        self.obj_dir = data_dir
        self.extrinsic_dir = os.path.join(data_dir,"pose")
        self.image_dir  = os.path.join(data_dir,"rgb")
        self.intrinsics_dir = os.path.join(data_dir,"intrinsics")
        self.image_length  = image_length 
        #print(self.image_dir)
        if not os.path.isdir(self.image_dir):
            print("wrong dir")
            return 

        self.extrinsic_list = [x for x in glob(os.path.join(self.extrinsic_dir,'*.txt'))]
        self.image_list = [x for x in glob(os.path.join(self.image_dir,'*.png'))]
        self.intrinsic_list = [x for x in glob(os.path.join(self.intrinsics_dir,'*.txt'))]
        self.intrinsic_params = [] 
        for file_name in self.intrinsic_list:
          #print("accessed")
          with open(file_name) as f:
             array = []
             for line in f:
               array.append([float(x) for x in line.split()])
             array_1 = np.reshape(array,(-1,1))
             array = np.reshape(array,(3,3))
             self.intrinsic_params.append(array)
        self.extrinsic_params = [] 
        for file_name in self.extrinsic_list:
          with open(file_name) as f:
             array = []
             for line in f:
               array.append([float(x) for x in line.split()])
             array_1 = np.reshape(array,(-1,1))
             array = np.reshape(array,(4,4))
             self.extrinsic_params.append(array)
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,index):
       #print("accessed")
       rgb_image  = Image.open(os.path.join(self.image_dir,self.image_list[index]))
       d = np.array(rgb_image)
       d1 = d[:,:,1:]
       #print(d1.shape)
       #print(d[0][0][0])
       #print("1111111111111111111111")
       #image_final = adjustedresolution(d1,self.image_length)
       image_final = d1
       intrinsics = self.intrinsic_params[index]
       extrinsics = self.extrinsic_params[index]
       
       return image_final,intrinsics,extrinsics

    def image_resolution(self,image_length):
      self.image_length = image_length

class class1(Dataset):

    def __init__(self,data_dir,number_of_instances,image_sidelength):
      #self.num_examples_per_instance = examples_per_instance

       self.object_list = []
       self.num_samples_per_instance = 10
       for dir_name  in os.listdir(data_dir):
           #print("accessed1")
           object_path = os.path.join(data_dir,dir_name)
           curr_object = object1(object_path,5,image_sidelength)
           self.object_list.append(curr_object)
       
       self.num_objects = len(self.object_list) 
       #print("length is:{}".format(self.num_objects))
       #print(self.object_list[0].__getitem__(0))          
    #def __len__(self):
    #  count = 0
    #  for obj in self.object_list:
    #    count+=len(obj)
    #  return count
    def __len__(self):
      return len(self.object_list)-1
            
    def image_resolution(self,Image_length):
      for obj in self.object_list:
        obj.image_resolution(Image_length)   
    def __getitem__(self,obj_index):
           #print("accessed_item")
           #print("accessed number is:{}".format(obj_index))
           curr_object_list = self.object_list[obj_index]
           x =[]
           for i in range(self.num_samples_per_instance):
               x.append(curr_object_list[np.random.randint(len(curr_object_list))])
           

           return x

        
    #def collate_fn(slef,batch_list):
if __name__ == "__main__":
  data_loaded = class1("C:/Users/DELL/Desktop/Current_fields/SRN/dataset/shepard_metzler_train/",20,64)
  final_loaded = DataLoader(data_loaded,batch_size = 4,shuffle=True,num_workers=0)
  for id,images in enumerate(final_loaded):
    #print(len(images[0]))
    #print(id)
    print((images[0][0]).shape)



        
