import torch
from torch.utils.data import Dataset
from srns import *
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataloader_srn import *
from torch.utils.data import DataLoader
import utils
from comon import *
import logging

if __name__== '__main__':
    intrinsic_matrix = utils.parse_intrinsics(intrinsics_file_path) 
     
    
    if validat1:
        val_dataset = class1(val_data_dir,num_instances,image_sidelength)


        val_dataloader  =Dataloader(val_dataset,batch_size = batch_size,shuffle=false,drop_last= true)
    
    id1 = torch.LongTensor([[2]])
    id1 = id1.to(device)
    model = SRNS(input_size,output_size,hidden_state_size,pixel_output)
    if torch.cuda.is_available():
        model = model.cuda()
    #train_dataset  = class1(data_dir,num_instances,image_sidelength)
    train_dataset = class1("C:/Users/DELL/Desktop/Current_fields/SRN/dataset/shepard_metzler_train/",20,64)
    train_dataloader = DataLoader(train_dataset,batch_size = 4,shuffle=True,num_workers=0)


    
    model.train()
    model.cuda()
    #results_path = os.path.join(root_dir,"results")
    if not os.path.isdir(results_path):
      os.mkdir(results_path)
    #log_path = os.path.join(results_path,"log_files")
    #tensorboard_dir = os.path.join(results_path,'tensorboard_files')
    #checkpoints_dir   = os.path.join(results_path,"checkpoints")
    if not os.path.isdir(log_path):
      os.mkdir(log_path)
      os.mkdir(tensorboard_dir)
    if not os.path.isdir(checkpoints_dir):
     os.mkdir(checkpoints_dir)
    if load_model is not False:
        print("Loading model from %s"%save_path)
        model.load_state_dict(save_path)
    
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    logger = logging.getLogger('LOG')
    logger.info("learning rate:{}".format(lr))
    logger.info("start epoch:{}".format(start_epoch))
    logger.info("end epoch:{}".format(end_epoch))
    logger.info("Batch_size:{}".format(train_batch_size))
    print("training started")
    if torch.cuda.is_available():
        print("using cuda..........")
    train_dataloader =  DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
    writer = SummaryWriter(log_dir = log_path)
    val_writer = SummaryWriter(log_dir = log_path)
    for epoch in range(0,n_epochs):
        iter = 0
        for inputs in train_dataloader:
         for objects in inputs:
            iter+=1
            if torch.cuda.is_available():    
                #print('using cuda........')
                #images,extrinsics,intrinsics = objects
                images = objects[0]
                extrinsics = objects[2]
                intrinsics =objects[1]
                #inputs = inputs.to(device)
                images = images.to(device)
                extrinsics = extrinsics.to(device)
                intrinsics = intrinsics.to(device)
                
                
                #labels = labels.to(device)
            translations = extrinsics[:,:,-1]
            pixels,depths  =model(id1,images,extrinsics,intrinsics,translations)

            optimizer.zero_grad()

            #image_loss  = model.image_loss_l2(inputs,outputs)
            #image_loss = model.image_loss_l2(inputs,id1)
            image_loss = model.image_loss_l2()
            latent_loss =  model.latent_loss_embed(lamb1)
            depth_reg_loss  = model.depth_reg_loss_l2(depths,lamb1)
            #weight2*latent_loss + weight3*depth_reg_loss
            total_loss = weight1*image_loss 
            total_loss.backward(retain_graph=True)
                 
            optimizer.step()
            print('____________________________________________________________________________________________________________________________________')
            print('iter:{}'.format(iter),"||",'epoch:{}'.format(epoch),"||",'lr:{:.6f}'.format(lr),"||",'recons_loss:{:.6f}'.format(image_loss),"||",'embed_loss:{}'.format(latent_loss),"||",'depth_reg_loss:{}'.format(depth_reg_loss))
            #print('iter:{}'.format(iter),'epoch:{}'.format(epoch),'lr:{:.6f}'.format(lr),'recons_loss:{:.6f}'.format(image_loss))
            writer.add_scalar('recon_loss',image_loss,epoch)
            writer.add_scalar('embed_loss',latent_loss,epoch)
            writer.add_scalar('depth_reg_loss',depth_reg_loss,epoch) 
    if epoch%SAVE_MODEL_EPOCH==0:
        print("running val set...........")

        model.eval()
        with torch.no_grad():
         dist_losses = []  
         for inputs in val_dataloader:
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                #labels = labels.to(device)
            ouputs = model(inputs)
            dist_loss =  model.image_loss_l2(inputs,outputs)
            dist_losses.append(dist_losses)

          
        model.train()
        print("validation_loss:{:.6f}".format(np.mean(dist_loss)))
        val_writer.add_scalar('recon_loss',np.mean(dist_loss),epoch)
        torch.save(model.state_dict,os.path.join(checkpoints_dir,'epoch%d.pth'%epoch))

                






 
