################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from experiment_v02 import Experiment
import sys
import ssl

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py custom`
if __name__ == "__main__":
    print("Running Experiment: ")
    ssl._create_default_https_context = ssl._create_unverified_context
    #exp = Experiment("Exp_ResNet50", 'resnet50')
    exp = Experiment(sys.argv[1], sys.argv[2])
    exp.run()
    exp.test()