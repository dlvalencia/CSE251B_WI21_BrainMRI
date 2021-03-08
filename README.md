# CSE251B_WI21_BrainMRI
 
### Changelog
March 7th - Initial code dump of all files.

March 7th - Fixed code to put labels onto CUDA, uploaded VGG16 training results

March 7th - Fixed code in model_factory.py to fix sizing issues with DenseNet121 forward pass

March 8th - Added experiment_v03.py, updated main_run_exp.py to use updated experiment_v03.py.


### How to train a model
The syntax for training a model is python main_run_exp.py <config_file>, where <config_file> is a config .json file. See the example Resnet50_LR0225.json file for an example.


## JSON File Format
The following entries should be changed:

* experiment_name : Used to set the name of the experiment, a directory will be created to store results.

* num_epochs : defines the number of epochs for training.

* learning_rate : defines the learning rate for training.

* model_type : defines the model. Acceptable values are:
    1. 'resnet50' - model is a pretrained ResNet50 CNN
    2. 'vgg16' - model is a pretrained VGG16 CNN with batch norm
    3. 'densenet121' - model is a pretrained DenseNet121 CNN

## Changes pending
* Need to create config files to implement more generic training and setting of hyperparameters, model definition. - Done [ March 8th]

* Add additional models to model_factory.

* Add pixel intensification to dataloader for additional augmentation.