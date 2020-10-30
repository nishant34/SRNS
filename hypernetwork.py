
#class autoencoder(self,)


#class autoencoder(self,)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import os
import sys
from comon import  *

class fullnetwork(nn.Module):
    def __init__(self,latent_size,output_size,hidden_layer_dim,num_layers):
        super(fullnetwork,self).__init__()
        self.latent_dim  = latent_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_layer_dim
        self.size_list = []
        self.weights_list = []
        self.output_size_list = []
        self.size_list.append(3*hidden_layer_dim)
        self.output_size_list.append(hidden_layer_dim)
        for i in range(num_layers-2):
            self.size_list.append(hidden_layer_dim*hidden_layer_dim)
            self.output_size_list.append(hidden_layer_dim)

        self.size_list.append(hidden_layer_dim*32)
        self.output_size_list.append(32)
        self.modules = []
        self.layer1 = nn.Linear(3,256)
        self.layer2  = nn.Linear(256,256)
        self.layer22  = nn.Linear(256,256)
        self.layer3  = nn.Linear(256,32)           
        self.modules.append(self.layer1)
        self.modules.append(self.layer2)
        self.modules.append(self.layer22)
        self.modules.append(self.layer3)
        self.hyper_users = []
        for k in range(num_layers):
            a1  =  usehypernetwork(self.size_list[k],self.latent_dim,self.output_size_list[k])         
            self.hyper_users.append(a1)
        
    def forward(self,x):
            net = []
            for j in range(num_layers):
                #print("pakdo scam")
                #print(j)
                #self.hyper_user = usehypernetwork(self.size_list[j],self.latent_dim,self.output_size_list[j])
                #net.append(self.hyper_user(x))     
                self.weights_list.append(self.hyper_users[j](x))
                #print(self.modules[j].weight.shape)
                #print(self.weights_list[j].shape)
                self.modules[j].weight = nn.Parameter(self.weights_list[j])
            return nn.Sequential(*self.modules)


class usehypernetwork(nn.Module):
    def __init__(self,num_weights,latent_size,output_size):
        super(usehypernetwork,self).__init__()
        self.num_weights = num_weights
        self.latent_size = latent_size
        self.hyper_net  = hypernetwork(latent_size,self.num_weights,num_layers)
        self.hyper_net = self.hyper_net.to(device)
        self.output_size = output_size
    def forward(self,x):
        x = x.to(device)
        weights =  self.hyper_net(x)
        weights  =torch.reshape(weights,[self.output_size,-1])
        return weights
        


class hypernetwork(nn.Module):
       def __init__(self,latent_size,num_output_param,num_layers_representation):
             super(hypernetwork,self).__init__()
             self.layer1 = nn.Linear(32,256)
             self.layer2  = nn.Linear(256,256)
             self.layer3  = nn.Linear(256,256)
             self.layer4  = nn.Linear(256,256)
             self.layer5  = nn.Linear(256,num_output_param)
                           
       
       def forward(self,x):
             
             x  = self.layer1(x)
             x = self.layer2(x)
             x = self.layer3(x)
             x = self.layer4(x)
             x = self.layer5(x)
             return x
                     


             
      






    
