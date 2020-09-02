# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from datetime import datetime
import numpy as np
import torch
import monai

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from BioMedClasses import *
from plotter import *
from pipeline import *
from parameters import *

datetime_now = datetime.now()

def train(trainX, trainY, valX, valY, testX=None, testY=None, num_class=3, plot=True, fold=-1):
        train_transforms, val_transforms = define_transforms(SETTING_MODEL, RESIZE_TO_PIXEL)

        # create a training data loader
        train_ds = BioMedDataset(trainX, trainY, train_transforms)
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

        # create a validation data loader
        val_ds = BioMedDataset(valX, valY, val_transforms)
        val_loader = DataLoader(val_ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

        # model, loss function and optimizer
        device = torch.device("cuda:0")
        model = define_model(SETTING_MODEL, num_class=num_class, device=device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = define_optimizer(SETTING_OPTIMIZER, model.parameters(), LEARNING_RATE)

        # start training and validating
        epoch_loss_train_values, epoch_loss_val_values, epoch_acc_train_values, epoch_acc_val_values, epoch_sensi_train_values, epoch_sensi_val_values, epoch_speci_train_values, epoch_speci_val_values, epoch_mem_usage_values = train_model(EPOCHS, train_ds, train_loader, val_loader, VAL_INTERVAL, model, optimizer, loss_function, device)

        if plot:
            if fold == -1: # percentage split
                # to plot training and validation metrics, added by changqia
                metrics_plot_file = "{}/epoch_metrics_{}_{}_{}_{}_{}.png".format(PLOT_DIR, SETTING_MODEL, SETTING_OPTIMIZER, BATCH_SIZE, RESIZE_TO_PIXEL, datetime_now.strftime("%Y.%m.%d_%H-%M-%S"))
                memory_usage_plot_file = "{}/epoch_mem_usage_{}_{}_{}_{}_{}.png".format(PLOT_DIR, SETTING_MODEL, SETTING_OPTIMIZER, BATCH_SIZE, RESIZE_TO_PIXEL, datetime_now.strftime("%Y.%m.%d_%H-%M-%S"))
                last_epoch_plot_file = "{}/metrics_last_epoch_{}_{}_{}_{}_{}.png".format(PLOT_DIR, SETTING_MODEL, SETTING_OPTIMIZER, BATCH_SIZE, RESIZE_TO_PIXEL, datetime_now.strftime("%Y.%m.%d_%H-%M-%S"))
            else:
                metrics_plot_file = "{}/epoch_metrics_{}_{}_{}_{}_{}_fold{}.png".format(PLOT_DIR, SETTING_MODEL, SETTING_OPTIMIZER, BATCH_SIZE, RESIZE_TO_PIXEL, datetime_now.strftime("%Y.%m.%d_%H-%M-%S"), fold)
                memory_usage_plot_file = "{}/epoch_mem_usage_{}_{}_{}_{}_{}_fold{}.png".format(PLOT_DIR, SETTING_MODEL, SETTING_OPTIMIZER, BATCH_SIZE, RESIZE_TO_PIXEL, datetime_now.strftime("%Y.%m.%d_%H-%M-%S"), fold)
                last_epoch_plot_file = "{}/metrics_last_epoch_{}_{}_{}_{}_{}_fold{}.png".format(PLOT_DIR, SETTING_MODEL, SETTING_OPTIMIZER, BATCH_SIZE, RESIZE_TO_PIXEL, datetime_now.strftime("%Y.%m.%d_%H-%M-%S"), fold)

            last_epoch_plot(last_epoch_plot_file, EPOCHS, epoch_acc_val_values[-1], epoch_sensi_val_values[-1], epoch_speci_val_values[-1], SETTING_MODEL, BATCH_SIZE, RESIZE_TO_PIXEL)
            metrics_plot(metrics_plot_file, EPOCHS, epoch_loss_train_values, epoch_loss_val_values, epoch_acc_train_values, epoch_acc_val_values, epoch_sensi_train_values, epoch_sensi_val_values, epoch_speci_train_values, epoch_speci_val_values, SETTING_MODEL, SETTING_OPTIMIZER, fold, K_FOLDS)
            memory_usage_plot(memory_usage_plot_file, EPOCHS, epoch_mem_usage_values, SETTING_MODEL, BATCH_SIZE, RESIZE_TO_PIXEL) 

        if testX != None:
            # create a test data loader
            test_ds = BioMedDataset(testX, testY, val_transforms)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
            # testing
            test(test_loader, model, device)
        
        return (epoch_loss_train_values, epoch_loss_val_values, epoch_acc_train_values, epoch_acc_val_values, epoch_sensi_train_values, epoch_sensi_val_values, epoch_speci_train_values, epoch_speci_val_values, epoch_mem_usage_values)


def main():
    global SETTING_MODEL
    global SETTING_OPTIMIZER
    global EPOCHS
    global RESIZE_TO_PIXEL
    global CROSS_VALIDATION
    global K_FOLDS
    SETTING_MODEL, SETTING_OPTIMIZER, EPOCHS, RESIZE_TO_PIXEL, CROSS_VALIDATION, K_FOLDS = parse_arguments()

    log_file = "{}/training_{}_{}_{}_{}_{}_{}.txt".format(LOG_DIR, SETTING_MODEL, SETTING_OPTIMIZER, EPOCHS, BATCH_SIZE, RESIZE_TO_PIXEL, datetime_now.strftime("%Y.%m.%d_%H-%M-%S"))
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    sys.stdout = PrintToFileAndConsole(open(log_file, "w"), sys.stdout)

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print(f"Running with settings: model = {SETTING_MODEL}, optimizer = {SETTING_OPTIMIZER}, cross validation = {K_FOLDS if CROSS_VALIDATION else CROSS_VALIDATION}, epochs = {EPOCHS}, batch size = {BATCH_SIZE}, learning rate = {LEARNING_RATE}, resize to ({RESIZE_TO_PIXEL}, {RESIZE_TO_PIXEL})")

    class_names, image_file_list, image_label_list = load_directory(DATA_DIR, debug=True)
    num_class = len(class_names)

    if CROSS_VALIDATION:
        print("cross validation")

        # result list
        validation_results = []
        # shuffle images
        imagesX, imagesY = shuffle_x_y(image_file_list, image_label_list)

        # split images up
        folds = np.array_split(list(zip(imagesX, imagesY)), K_FOLDS)
        fold_indices = list(range(len(folds)))

        loss_train_values = []
        loss_val_values = []
        acc_train_values = []
        acc_val_values = []
        sensi_train_values = []
        sensi_val_values = []
        speci_train_values = []
        speci_val_values = []
        mem_values = []

        for i in fold_indices:
            print(f"\nrunning fold {i+1} of {len(fold_indices)}...")

            training = np.concatenate([folds[x] for x in fold_indices if x!=i],axis=0)
            validation = folds[i]

            trainX, trainY, valX, valY = list(), list(), list(), list()
            
            for j in range(len(training)):
                trainX.append(str(training[j][0]))
                trainY.append(int(training[j][1]))
            
            for j in range(len(validation)):
                valX.append(str(validation[j][0]))
                valY.append(int(validation[j][1]))
            
            epoch_loss_train_values, epoch_loss_val_values, epoch_acc_train_values, epoch_acc_val_values, epoch_sensi_train_values, epoch_sensi_val_values, epoch_speci_train_values, epoch_speci_val_values, epoch_mem_usage_values = train(trainX, trainY, valX, valY, fold=i, num_class=num_class, plot=True)
            loss_train_values.append(epoch_loss_train_values)
            loss_val_values.append(epoch_loss_val_values)
            acc_train_values.append(epoch_acc_train_values)
            acc_val_values.append(epoch_acc_val_values)
            sensi_train_values.append(epoch_sensi_train_values)
            sensi_val_values.append(epoch_sensi_val_values)
            speci_train_values.append(epoch_speci_train_values)
            speci_val_values.append(epoch_speci_val_values)
            mem_values.append(epoch_mem_usage_values)
        
        # calculate average over folds
        avg_loss_train_values, avg_loss_val_values, avg_acc_train_values, avg_acc_val_values, avg_sensi_train_values, avg_sensi_val_values, avg_speci_train_values, avg_speci_val_values, avg_mem_usage_values = [], [], [], [], [], [], [], [], []
        for i in range(EPOCHS):
            avg_loss_train_values.append(np.mean([ loss_train_values[j][i] for j in range(len(loss_train_values)) ]))
            avg_loss_val_values.append(np.mean([ loss_val_values[j][i] for j in range(len(loss_val_values)) ]))
            avg_acc_train_values.append(np.mean([ acc_train_values[j][i] for j in range(len(acc_train_values)) ]))
            avg_acc_val_values.append(np.mean([ acc_val_values[j][i] for j in range(len(acc_val_values)) ]))
            avg_sensi_train_values.append(np.mean([ sensi_train_values[j][i] for j in range(len(sensi_train_values)) ]))
            avg_sensi_val_values.append(np.mean([ sensi_val_values[j][i] for j in range(len(sensi_val_values)) ]))
            avg_speci_train_values.append(np.mean([ speci_train_values[j][i] for j in range(len(speci_train_values)) ]))
            avg_speci_val_values.append(np.mean([ speci_val_values[j][i] for j in range(len(speci_val_values)) ]))
            avg_mem_usage_values.append(np.mean([ mem_values[j][i] for j in range(len(mem_values)) ]))

        metrics_plot_file = "{}/epoch_metrics_{}_{}_{}_{}_{}_combined.png".format(PLOT_DIR, SETTING_MODEL, SETTING_OPTIMIZER, BATCH_SIZE, RESIZE_TO_PIXEL, datetime_now.strftime("%Y.%m.%d_%H-%M-%S"))
        memory_usage_plot_file = "{}/epoch_mem_usage_{}_{}_{}_{}_{}_combined.png".format(PLOT_DIR, SETTING_MODEL, SETTING_OPTIMIZER, BATCH_SIZE, RESIZE_TO_PIXEL, datetime_now.strftime("%Y.%m.%d_%H-%M-%S"))
        last_epoch_plot_file = "{}/metrics_last_epoch_{}_{}_{}_{}_{}_combined.png".format(PLOT_DIR, SETTING_MODEL, SETTING_OPTIMIZER, BATCH_SIZE, RESIZE_TO_PIXEL, datetime_now.strftime("%Y.%m.%d_%H-%M-%S"))
        acc_plot_file = "{}/accuracy_single_{}_{}_{}_{}_{}_combined.png".format(PLOT_DIR, SETTING_MODEL, SETTING_OPTIMIZER, BATCH_SIZE, RESIZE_TO_PIXEL, datetime_now.strftime("%Y.%m.%d_%H-%M-%S"))

        metrics_plot(metrics_plot_file, EPOCHS, avg_loss_train_values, avg_loss_val_values, avg_acc_train_values, acc_val_values, avg_sensi_train_values, avg_sensi_val_values, avg_speci_train_values, avg_speci_val_values, SETTING_MODEL, SETTING_OPTIMIZER, PLOT_CROSS_VAL_COMBINED, K_FOLDS)
        memory_usage_plot(memory_usage_plot_file, EPOCHS, avg_mem_usage_values, SETTING_MODEL, BATCH_SIZE, RESIZE_TO_PIXEL)
        last_epoch_plot(last_epoch_plot_file, EPOCHS, avg_acc_val_values[-1], avg_sensi_val_values[-1], avg_speci_val_values[-1], SETTING_MODEL, BATCH_SIZE, RESIZE_TO_PIXEL)
        accuracy_plot(acc_plot_file, EPOCHS, acc_val_values)
            
    else:
        print("percentage split")
        trainX, trainY, valX, valY, testX, testY = distribute_dataset(image_file_list, image_label_list, shuffle=True, debug=True)
        train(trainX, trainY, valX, valY, testX, testY, num_class=num_class)

if __name__ == "__main__":
    main()