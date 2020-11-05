import torch.nn as nn
import utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from comon import *
import utils
from hypernetwork import *
from geormetry import *
from PIL import Image
import scipy.misc
class SRNS(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,pixel_output):
          super(SRNS,self).__init__()
          self.layer1 = nn.Linear(input_size,256)
          self.norm1 = nn.BatchNorm1d(256)
          self.layer2  = nn.Linear(256,256)
          self.norm2 = nn.BatchNorm1d(256)
          self.layer3  = nn.Linear(256,output_size)
          self.norm3 = nn.BatchNorm1d(output_size)
          self.activation = nn.ReLU()
          #self.norm = nn.BatchNorm1d()
          self.pixel_layer1 = nn.Linear(32,256)
          self.pixel_layer2 = nn.Linear(256,256)
          self.pixel_layer3 = nn.Linear(256,256)
          self.pixel_layer4 = nn.Linear(256,256)
          self.pixel_layer5 = nn.Linear(256,pixel_output)
          #self.lstm_marcher =  nn.LSTM(32,hidden_size,1)  
          self.lstm_marcher =  nn.LSTM(32,hidden_size)  
          self.d = initial_distance
          self.latent_codes = nn.Embedding(num_instances,latent_dim).cuda()
          self.hyper_scene_representer = fullnetwork(latent_size,output_size,hidden_layer_dim,num_layers) 
          self.h0 = torch.randn(1,16384,32)
          self.c0 = torch.randn(1,16384,32)
          self.h0 = self.h0.to(device)
          self.c0 = self.c0.to(device)
          self.delta_to_d_layer = nn.Linear(32,1)
          self.logs = list()
           


    def scene_representer(self,x,latent_embed):
          x = self.layer1(x)
          x = self.activation(x)
          x = self.norm1(x)
          x = self.layer2(x)
          x = self.activation(x)
          x = self.norm2(x)
          x = self.layer3(x)
          return x
    def pixel_generator(self,x):
         #print(x.shape)
         #1w_shape = int(x.shape[1]/32)
         #print("harshad_mehta:{}".format(w_shape))
         #1w = torch.zeros(batch_size,w_shape,3)
         #1count = 0
         #1for i in range(0,x.shape[-1],32):
         #1 c= x[:,i:i+32]
          #print(c.shape)
          #print("11111")
          a = self.pixel_layer1(x)
          a = self.activation(a)
          #a = self.norm1(a)
          a = self.pixel_layer2(a)
          a = self.activation(a)
          #a = self.norm1(a)
          
          a = self.pixel_layer3(a)
          a = self.activation(a)
          #a = self.norm1(a)
         
          a = self.pixel_layer4(a)
          a = self.activation(a)
          #a = self.norm1(a)
          
          a = self.pixel_layer5(a)
          #1w[:,count,:] = a
         #1return w
          return a


    def forward(self,id1,x,R,k,t):
      #print("the shape of x is :{}".format(x.shape))
      curr_latent_code = self.latent_codes(id1)
      self.latent = curr_latent_code
      self.latent = self.latent.to(device)
      #print("and the shape is:{}".format(x.shape))
      x1 = x.reshape(x.size(0),-1)
      d_shape = x1.shape[1]/3
      d_shape = int(d_shape)
      #1self.d = torch.zeros(4,d_shape)
      self.d = torch.zeros(16384,32)
      self.d = self.d.to(device)
      
      for i in range (2):
         self.d1 = initial_distance
         y = utils.get_world_coordinates(R,k,t,x,self.d1)
         y = torch.reshape(y,[4,-1])
         w_shape = y.shape[-1]
         w_shape = w_shape/3
         w_shape = w_shape*32
         w_shape =int(w_shape)
         w =torch.zeros(4,w_shape)
         #print("and the shape for w is {}".format(w.shape))
         w = w.to(device)
         count = 0
         network = self.hyper_scene_representer(curr_latent_code)
         #new approach
         y = y.to(device)
         y = y.view(y.shape[0],-1,3)
         w_correct = network(y)
         w_correct = w_correct.view(-1,feature_vector_size)
         w_correct = w_correct[None,:,:]
         self.delta,(self.h1,self.c1) = self.lstm_marcher(w_correct,(self.h0,self.c0)) 
         self.h0 = self.h1
         self.c0 = self.c1
         self.delta_1 = self.delta_to_d_layer(self.delta)
         self.d = self.d + self.delta_1

             
         #1for i in range(0,x1.shape[-1],3):
         #1    if  count==1000:
         #1            break
         #1    a = y[:,i:i+3]
         #1    a = a.to(device)
             #network = self.hyper_scene_representer(curr_latent_code)
             #print(a.shape)
             #print(w[:,i:i+feature_vector_size].shape)
             #print("got it")
          #1   w[:,i:i+feature_vector_size] = network(a)
             #b = network(a)
             #print(b.shape)
             #print("222222")
          #1   w1 = w[None,:,:]
          #1   self.delta,(self.h1,self.c1) = self.lstm_marcher(w1[:,:,i:i+feature_vector_size],(self.h0,self.c0))
          #1   self.h0 = self.h1
          #1   self.c0 = self.c1
          #1   self.delta_1 = self.delta_to_d_layer(self.delta)
             #print(self.d.shape)
             #print(self.delta_1.shape)
          #1   self.d[:,count] = self.d[:,count]+self.delta_1[0,:,0]
          #1   count+=1
      w = w_correct.view(4,-1,32)
      pixel_rendered = self.pixel_generator(w)
      with torch.no_grad():
            #batch_size = uv.shape[0]
            batch_size = x.size(0)
            uv = np.mgrid[0:target_length,0:target_length]
            uv = np.reshape(uv,[2,-1]).transpose(1,0)
            uv = uv[None,:,:]
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
            x_cam = uv[:,:,0].view(batch_size,-1)
            y_cam = uv[:,:,1].view(batch_size,-1)
            z_cam  = self.d.view(batch_size,-1)
            t = x_cam.shape[1]
            z_cam1 = z_cam[:,0:t]


            #normals = compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
            depth_image = depth_map_to_cam_coordinates(x_cam,y_cam,z_cam1,k)
            #print("depth_cam_passed")
            normals = get_normal_map_from_depth_map(depth_image)
            #print("normals passed")
            normals = np.array(normals)
            #print("the uniqueness in the normals")
            #print(np.unique(normals))
            normals1 = normals.reshape(4,32,32,3)
            normals2 = normals1[0]
            #print(normals2)
            #print(type(normals2))
            normals3 = Image.fromarray(normals2.astype(np.uint8))

            normals3.save('output_file.jpg')
            #print("the shape of normals is :{}".format(normals.shape))
            #scipy.misc.imsave('output_file.jpg',normals)
            #print("image_saved")
            #r = t
            self.logs.append(("image", "normals",
                           torchvision.utils.make_grid(torch.tensor(normals), scale_each=True, normalize=True), 100))
      self.input1 = x 
      self.input1 = self.input1.to(device)
      self.output1 = torch.reshape(pixel_rendered,x.shape)  
      self.output1 = self.output1.to(device)
      
      return pixel_rendered,self.d


    #def image_loss_l2(self,I,E,K):
    #def image_loss_l2(self,I,id1):
    def image_loss_l2(self): 
            #z = self.get_latent_embedding(I)
            #z = self.latent_codes(id1)
            
            #output1 = self.pixel_generator(self.hyper_scene_representer(z))
            #input1 = I
            loss = nn.MSELoss()
            return loss(self.input1,self.output1)
      

    def depth_reg_loss_l2(self,d,lambda1):
            #return lambda1*(min(d,0))*min(d,0)
            loss = nn.MSELoss()
            return lambda1*loss(d,torch.zeros_like(d))  

    def  latent_loss_embed(self,lambda1):
            loss = nn.MSELoss()
            #return lambda1*(self.latent)*(self.latent)
            return lambda1*loss(self.latent,torch.zeros_like(self.latent))



