# CSE251B_WI21_BrainMRI
 
### Changelog
March 7th - Initial code dump of all files.






### How to train a model
The syntax for training a model is python main_run_exp.py <exp_name> <model_type>, where <exp_name> is a string used to name the storage directory and <model_type> is a string used to define the model you want. The <exp_name> should be relevant to the model used. Acceptable values for <model_type> are below:


1. 'resnet50' - model is a pretrained ResNet50 CNN
2. 'vgg16' - model is a pretrained VGG16 CNN with batch norm
3. 'densenet121' - model is a pretrained DenseNet121 CNN

## Changes pending
Need to create config files to implement more generic training and setting of hyperparameters, model definition.