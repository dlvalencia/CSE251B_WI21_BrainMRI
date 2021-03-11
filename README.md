# CSE251B_WI21_BrainMRI
 
### Changelog
March 7th - Initial code dump of all files.

March 7th - Fixed code to put labels onto CUDA, uploaded VGG16 training results

March 7th - Fixed code in model_factory.py to fix sizing issues with DenseNet121 forward pass

March 8th - Added experiment_v03.py, updated main_run_exp.py to use updated experiment_v03.py.

March 11th - Added experiment_v04.py, updated main_run_exp.py, and added dataloader2.py. Now requires definition of image transforms in the json configuration file.

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

* transforms : defines the transforms and their parameters
    1. 'RotateAngle' - the absolute range edges that the rotation transformation will apply. Expects float value.
    2. 'HorzFlip' - Expects string "true" or "false", wether flip will be applied with 50% probability.
    3. 'ContrastFactor' - Float. Factor by which we increase the contrast of the image. 1.0 returns original image. Less than 1 makes image more dull, greater than 1 increases the contrast.
    4. 'SharpnessFactor' - Float. Factor by which we increase the sharpness of the image. Less than 1 blurs the image, greater than 1 sharpens the image.
    5. 'BrightnessFactor' - Float. Factor by which we increase the brightness of the image. Less than 1 darkens the image, greater than 1 brightens the image.

## Changes pending
* Need to create config files to implement more generic training and setting of hyperparameters, model definition. - Done [ March 8th]

* Add additional models to model_factory.

* Add pixel intensification to dataloader for additional augmentation. - Done [March 11th]