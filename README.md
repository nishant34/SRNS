# SRNS
* Implementation of idea similar to what is described in Scene representation networks (NeurIPS 2019).
* The model is capable of representing scenes as a funcitons of coordinates which implies continuity instead of discrete representations such as point clouds and voxelgrids.
* Hypernetworks have been used to predict weights of the model responsible for representing points latent space feature vectors.
* MLP network is used as a  neural renderer for predictng colors from feature vectors.
* The paper describes a differentiable ray marching algorithm using lstm for predicting depth.

# Requirements
For package management install Anaconda and then to create and activate an environment :
```javascript
conda create -n myenv
source activate myenv

```

Now all the packages required  can be installed. Requirements  are:
* Python  3.6 or greater 
* Pytorch 1.14 or greater
* scipy
* matplotlib
* skimage

Requirements.txt file willl be uploaded soon .


# Training the model
Set teh desired values in comon.py file and after that in the command line:
```javascript
python train.py
```
* A logger can also be used to save the logs into a txt file .
* Model will be saved in the specified directory
* Tensorboard logs will also be saved in the specified directory. 

For visualising the tensorboard losses:
```javascript
python -m tensorboard.main --logdir=[PATH_TO_LOGDIR]
```
test.py file will be uploaded soon for generating results using saved model weights and evaluating them.

Visualisations for novel view synthesis will also be uploaded soon along with few shot reconstruction.

# Structure:
* dataloader_srn.py : For loading the data as objects of a articula class of shapes.
* hypernetwork.py : Code for using hypernetworks to predict model weights taking latent feature vetcor of shape as input.
* geormetry.py : For 3D projections and related functions.
* comon.py  : Hyperparameter values , path values and other parameter values.
* train.py : Contains the ocde for trianing the model.
* utils.py : Contains utility functions helpful during training and forward pass of the model.
* srns.py : Contains the complete code of the model.


