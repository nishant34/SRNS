import torch.nn as nn
import utils
import torch.nn as nn
import torch.nn.functional as F
from comon import *
import utils
from hypernetwork import *
from geormetry import *
class SRNS(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,pixel_output):
          super(SRNS,self).__init__()
          self.layer1 = nn.Linear(input_size,256)
          self.layer2  = nn.Linear(256,256)
          self.layer3  = nn.Linear(256,output_size)
          self.activation = nn.ReLU()
          #self.norm = nn.BatchNorm1d()
          self.pixel_layer1 = nn.Linear(32,256)
          self.pixel_layer2 = nn.Linear(256,256)
          self.pixel_layer3 = nn.Linear(256,256)
          self.pixel_layer4 = nn.Linear(256,256)
          self.pixel_layer5 = nn.Linear(256,pixel_output)
          self.lstm_marcher =  nn.LSTM(32,hidden_size,1)  
          self.d = initial_distance
          self.latent_codes = nn.Embedding(num_instances,latent_dim).cuda()
          self.hyper_scene_representer = fullnetwork(latent_size,output_size,hidden_layer_dim,num_layers) 
          self.h0 = torch.randn(1,batch_size,32)
          self.c0 = torch.randn(1,batch_size,32)
          self.h0 = self.h0.to(device)
          self.c0 = self.c0.to(device)
           
           
           


    def scene_representer(self,x,latent_embed):
          x = self.layer1(x)
          x = self.activation(x)
          #x = self.norm(x)
          x = self.layer2(x)
          x = self.activation(x)
          #x = self.norm(x)
          x = self.layer3(x)
          return x
    def pixel_generator(self,x):
         #print(x.shape)
         w_shape = int(x.shape[2]/32)
         #print("harshad_mehta:{}".format(w_shape))
         w = torch.zeros(batch_size,w_shape,3)
         count = 0
         for i in range(0,x.shape[-1],32):
          c= x[0,:,i:i+32]
          #print(c.shape)
          #print("11111")
          a = self.pixel_layer1(c)
          a = self.activation(a)
         #x = self.norm(x)
          a = self.pixel_layer2(a)
          a = self.activation(a)
         #x = self.norm(x)
          
          a = self.pixel_layer3(a)
          a = self.activation(a)
         #x = self.norm(x)
         
          a = self.pixel_layer4(a)
          a = self.activation(a)
         #x = self.norm(x)
          
          a = self.pixel_layer5(a)
          w[:,count,:] = a
         return w


    def forward(self,id1,x,R,k,t):
      curr_latent_code = self.latent_codes(id1)
      self.latent = curr_latent_code
      self.latent = self.latent.to(device)
      #print("and the shape is:{}".format(x.shape))
      #x = x.reshape(x.size(0),-1)
      for i in range (2):
         self.d = initial_distance
         y = utils.get_world_coordinates(R,k,t,x,self.d)
         y = torch.reshape(y,[4,-1])
         w_shape = y.shape[-1]
         w_shape = w_shape/3
         w_shape = w_shape*32
         w_shape =int(w_shape)
         w =torch.zeros(4,w_shape)
         #print("and the shape for w is {}".format(w.shape))
         w = w.to(device)
         for i in range(0,x.shape[-1],3):
             a = y[:,i:i+3]
             a = a.to(device)
             network = self.hyper_scene_representer(curr_latent_code)
             #print(a.shape)
             w[:,i:i+feature_vector_size] = network(a)
             #b = network(a)
             #print(b.shape)
             #print("222222")
             w = w[None,:,:]
             self.delta,(self.h1,self.c1) = self.lstm_marcher(w[:,:,i:i+feature_vector_size],(self.h0,self.c0))
             self.h0 = self.h1
             self.c0 = self.c1
             self.d = self.d+self.delta
      
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


            #normals = compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
            #depth_image = depth_map_to_cam_coordinates(x_cam,y_cam,z_cam,k)
            #normals = get_normal_map_from_depth_map(deth_image)
            #self.logs.append(("image", "normals",
                           #torchvision.utils.make_grid(normals, scale_each=True, normalize=True), 100))
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



