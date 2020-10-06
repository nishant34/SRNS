
#class autoencoder(self,)

class fullnetwork(nn):
    def __init__(self,latent_size,output_size,hidden_layer_dim,num_layers):
        self.latent_dim  = latent_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_layer_dim
        self.size_list = []
        self.wieghts_list = []
        self.size_list.append(latent_size*hidden_layer_dim)
        for i in range(num_layers-1):
            self.size_list.append(hidden_layer_dim*hidden_layer_dim)

        self.size_list.append(hidden_layer_dim*output_size)
        net = []
        #self.hyper_user = usehypernetwork()
        def forward(self,x):
            for j in range(num_layers):
                self.hyper_user = usehypernetwork(self.size_list[j],self.latent_dim)
                net.append(self.hyper_user(x))     
            return nn.Sequential(*net)


class usehypernetwork(nn):
    def __init__(self,num_weights,latent_size):
        self.num_weights = num_weights
        self.latent_size = latent_size
        self.hyper_net  = hypernetwork(latent_size,self.num_weights,num_layers)
    def forward(self,x):
        weights =  self.hypernet(x)
        return weights
        


class hypernetwork(nn):
       def __init__(self,latent_size,num_output_param,num_layers_representation):
             self.layer1 = nn.linear(latent_size,256)
             self.layer2  = nn.linear(256,256)
             self.layer3  = nn.linear(256,256)
             self.layer4  = nn.linear(256,256)
             self.layer5  = nn.linear(256,num_output_param)
             
       
       def forward(self,x):
             x  = self.layer1(x)
             x = self.layer2(x)
             x = self.layer3(x)
             x = self.layer4(x)
             x = self.layer5(x)
             return x
                     


             
      






    
