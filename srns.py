import torch.nn as nn
import util
import torch.nn as nn
import torch.nn.functional as F
import common
import util
class SRNS(nn):
    def __init__(self,input_size,output_size,hidden_size,pixel_output,rotation,trans,intriinsics):
          self.layer1 = nn.linear(input_size,256)
          self.layer2  = nn.linear(256,256)
          self.layer3  = nn.linear(256,output_size)
          self.activation = nn.ReLU()
          self.norm = nn.BatchNorm1d()
          self.pixel_layer1 = nn.linear(,output_size,256)
          self.pixel_layer2 = nn.linear(256,256)
          self.pixel_layer3 = nn.linear(256,256)
          self.pixel_layer4 = nn.linear(256,256)
          self.pixel_layer5 = nn.linear(256,pixel_output)
          self.lstm_marcher =  nn.LSTM(output_size,hidden_size,1)  
          self.d = common.initial_distance
          self.latent_codes = nn.Embedding(num_instances,latent_dim).cuda()
          self.hyper_scene_representer() = fullnetwork(latent_size,output_size,hidden_layer_dim,num_layers) 


    def scene_representer(self,x,latent_embed):
          x = self.layer1(x)
          x = self.activation(x)
          x = self.norm(x)
          x = self.layer2(x)
          x = self.activation(x)
          x = self.norm(x)
          x = self.layer3(x)
          
   def pixel_generator(self,x):
         x = self.pixel_layer1(x)
         x = self.activation(x)
         x = self.norm(x)
         x = self.pixel_layer2(x)
         x = self.activation(x)
         x = self.norm(x)
          
         x = self.pixel_layer3(x)
         x = self.activation(x)
         x = self.norm(x)
         
         x = self.pixel_layer4(x)
         x = self.activation(x)
         x = self.norm(x)
         
         x = self.pixel_layer5(x)


    def forward(self,x):
      x = x.reshape(x.size(0),-1)
      for i in range common.max_iters:
         self.d = common.initial_distance
         y = util.get_world_coordinates(R,K,t,x,self.d)
         for i in range(0,x.shape(-1),3):
             w[i:i+common.feature_vector_size] = self.hyper_scene_representer(y[i:i+2])
             self.delta,(self.h1,self.c1) = self.lstm_layer(w[i:i+common.feature_vector_size],(self.h0,self.c0))
             self.h0 = self.h1
             self.c0 = self.c1
             self.d = self.d+self.delta
      
      pixel_rendered = pixel_generator(w)
      with torch.no_grad():
            batch_size = uv.shape[0]
            x_cam = uv[:,:,0].view(batch_size,-1)
            y_cam = uv[:,:,1].view(batch_size,-1)
            z_cam  = self.d.view(batch_size,-1)


            normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
            self.logs.append(("image", "normals",
                              torchvision.utils.make_grid(normals, scale_each=True, normalize=True), 100))

      return pixel_rendered,self.d


      def image_loss_l2(self,I,E,K):
            z = self.get_latent_embedding(I)
            output1 = self.pixel_generator(self.scene_representer(z))
            input1 = I
            loss = nn.MSELoss()
            return loss(input1,output1)
      

      def depth_reg_loss_l2(self,d,lambda):
            return lambda*(min(d,0))*min(d,0)


      def  latent_loss_embed(self,embed,lambda):
            return lambda*(embed)*(embed)



