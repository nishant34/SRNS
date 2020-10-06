import torch
from torchvision import Dataset
from srns import *
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataloader_srn import *
from torch.utils.data import DataLoader
import utils
from common import *
import logging

if __name__== '__main__':
    intrinsic_matrix = utils.parse_instrinsics(intrinsics_file_path) 
    
    if torch.cuda.is_available():
        model = model.cuda()
    train_dataset  = class1(data_dir,num_instances,image_sidelength)

    if validat1:
        val_dataset = class1(val_data_dir,num_instances,image_sidelength)


        val_dataloader  =Dataloader(val_dataset,batch_size = batch_size,shuffle=false,drop_last= true)
    

    model = SRNS(input_size,out_put_size,hidden_state_size,pixel_output,rotation,translation,intrinsics)
    
    model.train()
    model.cuda()
    results_path = os.path.join(root_dir,"results")
    log_path = os.path.join(results_path,"log_files")
    tensorboard_dir = os.path.join(results_path,'tensorboard_files')
    checkpoints_dir   = os.path.join(results_path,"checkpoints")
    os.mkdir(log_path)
    os.mkdir(tensorboard_dir)
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
    train_dataloader =  Dataloader(train_dataset,batch_size=batch_size,shuffle=False)
    writer = SummaryWriter(log_dir = LOG_PATH)
    val_writer = SummaryWroter(log_dir = LOG_PATH)
    for epoch in range(0,n_epochs):
        iter = 0
        for inputs,labels in train_dataloader:
            iter+=1
            if torch.cuda.is_available():
                print('using cuda........')

                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs  =model(inputs)

            optimizer.zero_grad()

            image_loss  = model.image_loss_l2(inputs,outputs)
            latent_loss =  model.latent_loss_embed()
            depth_reg_loss  = model.depth_reg_loss_l2(inputs,outputs)

            total_loss = weight1*image_loss + weight2*latent_loss + weight3*depth_reg_loss
            total_loss.backward()
                 
            optimizer.step()

            print('iter:{}'.format(iter),'epoch:{}'.format(epoch),'lr:{:.6f}'.format(lr),'recons_loss:{:.6f}'.format(image_loss),'embed_loss:{}'.format(latent_loss),'depth_reg_loss:{}'.format(depth_reg_loss))
            writer.add_scalar('recon_loss'image_loss,epoch)
            writer.add_scalar('embed_loss',latent_loss,epoch)
            writer.add_scalar('dpeth_reg_loss',depth_reg_loss,epoch) 
    if epoch%SAVE_MODEL_EPOCH==0:
        print("running val set...........")

        model.eval()
        with torch.no_grad():
         dist_losses = []  
         for inputs,labels in val_dataloader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            ouputs = model(inputs)
            dist_loss =  model.image_loss_l2(inputs,outputs)
            dist_losses.append(dist_losses)

          
        model.train()
        print("validation_loss:{:.6f}".format(np.mean(dist_loss)))
        val_writer.add_summary('recon_loss',np.mean(dist_loss),epoch)
        torch.save(model.state_dict,os.path.join(checkpoints_dir,'epoch%d.pth'%epoch))

                






 
