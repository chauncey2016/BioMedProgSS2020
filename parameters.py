import getopt
import sys
import numpy as np

######################### PARAMS ########################
EPOCHS = 100
VAL_INTERVAL = 1
BATCH_SIZE = 48                 #48/32?
NUM_WORKERS = 2
RESIZE_TO_PIXEL = 224
SETTING_MODEL = "resnet"        # "resnet", "densenet" or "inceptionv4"
LEARNING_RATE = 1e-4
SETTING_OPTIMIZER = "adam"                   # "adam" or "sgd"
CROSS_VALIDATION = False
K_FOLDS = 3

DATA_DIR = "/storage/images"    # 
LOG_DIR = "logs"
PLOT_DIR = "plots"


def parse_arguments():
    global SETTING_MODEL
    global SETTING_OPTIMIZER
    global EPOCHS
    global RESIZE_TO_PIXEL
    global CROSS_VALIDATION
    global K_FOLDS

    try:
        opts, argv = getopt.getopt(sys.argv[1:], "m:o:e:i:c:")
    except getopt.GetoptError as err:
        print("\nUsage of training.py:")
        print(f"\t-m resnet|densenet|inceptionv4\t Choose the model (default: {SETTING_MODEL})")
        print(f"\t-o adam|sgd\t\t\t Choose the optimizer (default: {SETTING_OPTIMIZER})")
        print(f"\t-e N\t\t\t\t Number of epochs (default: {EPOCHS})")
        print(f"\t-i N\t\t\t\t Resize data images to N x N pixels (default: {RESIZE_TO_PIXEL})")
        print(f"\t-c N\t\t\t\t Number of cross validation folds - '0' means percentage split (default: {K_FOLDS if CROSS_VALIDATION else '0 (percentage split)'})")
        print()
        exit()

    for k, v in opts:
        if k == '-m':
            SETTING_MODEL = v
        elif k == '-o':
            SETTING_OPTIMIZER = v
        elif k == '-e':
            EPOCHS = int(v)
        elif k == '-i':
            RESIZE_TO_PIXEL = int(v)
        elif k == '-c':
            if int(v) > 1:
                CROSS_VALIDATION = True
                K_FOLDS = np.minimum(5, np.maximum(int(v), 0)) # between 0 and 5
            else:
                CROSS_VALIDATION = False
            
    
    return (
        SETTING_MODEL,
        SETTING_OPTIMIZER,
        EPOCHS,
        RESIZE_TO_PIXEL,
        CROSS_VALIDATION,
        K_FOLDS
    )