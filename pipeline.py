### 
### Pipeline for the Bio Med practicum.
### Authors: Qiang Chang, Fabian Brain
###


import os
import random
import numpy as np

from sklearn import metrics

import monai
from monai.transforms import Compose, LoadPNG, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom, Resize

import torch
from torch.utils.tensorboard import SummaryWriter

from InceptionV4 import *
from BioMedClasses import *
from eval_metrics import *

def load_directory(data_dir, debug=False):
    class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
    image_files = [[os.path.join(data_dir, class_name, x) 
                    for x in os.listdir(os.path.join(data_dir, class_name))] 
                   for class_name in class_names]
    image_file_list = []
    image_label_list = []
    for i, class_name in enumerate(class_names):
        image_file_list.extend(image_files[i])
        image_label_list.extend([i] * len(image_files[i]))
    
    if debug:
        print('Total image count:', len(image_file_list))
        print("Label names:", class_names)
        print("Label counts:", [len(image_files[i]) for i in range(len(class_names))])

    return (class_names, image_file_list, image_label_list)


def distribute_dataset(image_file_list, image_label_list, valid_frac=0.3, test_frac=0.2, shuffle=False, debug=False):
    trainX, trainY = [], []
    valX, valY = [], []
    testX, testY = [], []

    for i in range(len(image_file_list)):
        rann = np.random.random()
        if rann < valid_frac:
            valX.append(image_file_list[i])
            valY.append(image_label_list[i])
        elif rann < test_frac + valid_frac:
            testX.append(image_file_list[i])
            testY.append(image_label_list[i])
        else:
            trainX.append(image_file_list[i])
            trainY.append(image_label_list[i])

    if shuffle:
        trainX, trainY = shuffle_x_y(trainX, trainY)

    if debug:
        print("Training count =",len(trainX),"Validation count =", len(valX), "Test count =",len(testX))
    
    return (trainX, trainY, valX, valY, testX, testY)

def shuffle_x_y(x, y):
    """Shuffle two lists x and y equally"""
    train_pairs = list(zip(x, y))
    random.shuffle(train_pairs)
    x, y = zip(*train_pairs)
    return x, y

def define_transforms(setting_model, resize_to_pixel):
    """Returns the array of transform functions required for the specified model and target image size"""
    # apply configured network (does not apply to Resize!)
    transform_load_func = None
    transform_addchannel_func = None

    if setting_model == "densenet" or setting_model == "resnet":
        transform_addchannel_func = AddChannel
        transform_load_func = LoadToGrayscale
    elif setting_model == "inceptionv4":
        transform_addchannel_func = EmptyTransform     # do nothing, since LoadToRGB already returns (C, W, H)
        transform_load_func = LoadToRGB

    # Define transforms
    if resize_to_pixel != 1024:
        train_transforms = Compose([
            transform_load_func(),
            transform_addchannel_func(),
            #Shape(),
            Resize((resize_to_pixel, resize_to_pixel)),
            #Shape(),
            ScaleIntensity(),
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            #RandRotate(range_x=15, prob=0.5, keep_size=True),
            #RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            ToTensor()
        ])

        val_transforms = Compose([
            transform_load_func(),
            transform_addchannel_func(),
            Resize((resize_to_pixel, resize_to_pixel)),
            ScaleIntensity(),
            ToTensor()
        ])
    else:
        train_transforms = Compose([
            transform_load_func(),
            transform_addchannel_func(),
            #Shape(),
            #Resize((resize_to_pixel, resize_to_pixel)),
            #Shape(),
            ScaleIntensity(),
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            #RandRotate(range_x=15, prob=0.5, keep_size=True),
            #RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            ToTensor()
        ])

        val_transforms = Compose([
            transform_load_func(),
            transform_addchannel_func(),
            ScaleIntensity(),
            ToTensor()
        ])

    return train_transforms, val_transforms


def define_model(setting_model, num_class, device):
    """Returns the model instance"""
    if setting_model == "densenet":
        model = monai.networks.nets.densenet.densenet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
    elif setting_model == "resnet":
        model = monai.networks.nets.senet.se_resnet50(spatial_dims=2, in_channels=1, num_classes=num_class).to(device)
    elif setting_model == "inceptionv4":
        model = inceptionv4(num_classes=num_class, pretrained=False).to(device)
    else:
        raise Exception("Invalid model!")
    
    return model


def define_optimizer(setting_optimizer, model_parameters, learning_rate):
    """Returns the optimizer instance"""
    if setting_optimizer == "adam":
        optimizer = torch.optim.Adam(model_parameters, learning_rate)
    elif setting_optimizer == "sgd":
        optimizer = torch.optim.SGD(model_parameters, learning_rate, momentum=0.9)
    else:
        raise Exception("Invalid optimizer!")
    
    return optimizer


def train_model(epochs, train_ds, train_dl, val_dl, val_interval, model, optimizer, loss_function, device):
    """
        Trains the given model over the specified epochs with the given datasets and runs the validation each `val_interval` epochs
        Arguments:
            epochs -- the amount of epochs to be run
            train_ds -- the training Dataset instance
            train_dl -- the training DataLoader instance
            val_dl -- the validation DataLoader instance
            val_interval -- run the validation each val_interval epochs
            model -- the model instance
            optimizer -- the optimizer instance
            loss_function -- the loss function instance
            device -- the device instance (e.g. cuda)

        Returns a list of lists of metrics of every epoch:
            -- training loss values
            -- validation loss 
            -- accuracy training values
            -- accuracy validation values
            -- sensitivity training values
            -- sensitivity validation values
            -- specificity training values
            -- specificity validation values
            -- memory usage values
    """
    # start training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_train_values = list()
    epoch_loss_val_values = list()
    epoch_acc_train_values = list()
    epoch_acc_val_values = list()
    epoch_sensi_train_values = list()
    epoch_sensi_val_values = list()
    epoch_speci_train_values = list()
    epoch_speci_val_values = list()
    epoch_mem_usage_values = list()

    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss_train = 0
        epoch_loss_val = 0
        epoch_acc_train = 0
        epoch_sensi_train = 0
        epoch_speci_train = 0
        train_step = 0
        val_step = 0
        train_predictions, train_actuals = list(), list()
        for batch_data in train_dl:
            train_step += 1
            epoch_len = len(train_ds) // train_dl.batch_size

            if train_step <= epoch_len:
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss_train += loss.item()
                print(f"{train_step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + train_step)
                #store 
                pred_labels = outputs.detach().cpu()
                pred_labels = torch.argmax(pred_labels, dim=1)
                pred_labels =pred_labels.reshape(len(pred_labels), 1)
                actual_labels = labels.cpu().numpy()
                actual_labels = actual_labels.reshape(len(actual_labels), 1)
                train_predictions.append(pred_labels)
                train_actuals.append(actual_labels)
        
        epoch_loss_train /= train_step
        epoch_loss_train_values.append(epoch_loss_train)                       ## to plot training loss, changed by changqia

        epoch_mem_usage_values.append(torch.cuda.memory_allocated(device))

        train_actuals, train_predictions = np.vstack(train_actuals), np.vstack(train_predictions)
        cm = metrics.confusion_matrix(train_actuals, train_predictions)
        
        ## to plot training metrics, added by changqia
        epoch_acc_train, epoch_sensi_train, epoch_speci_train = confusion_metrics(conf_matrix=cm, debug=True)
        epoch_acc_train_values.append(epoch_acc_train)
        epoch_sensi_train_values.append(epoch_sensi_train)
        epoch_speci_train_values.append(epoch_speci_train)

        print(f"epoch {epoch + 1} average loss: {epoch_loss_train:.4f}")

        if (epoch + 1) % val_interval == 0:
            print("validate...")
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                epoch_acc_val = 0
                epoch_sensi_val = 0
                epoch_speci_val = 0
                val_predictions, val_actuals = list(), list()
                for val_data in val_dl:
                    val_step += 1
                    val_images, val_labels = val_data[0].to(device=device, dtype=torch.float), val_data[1].to(device=device)
                    val_outputs = model(val_images)
                    val_loss = loss_function(val_outputs, val_labels)
                    epoch_loss_val += val_loss.item()
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()

                    #store 
                    val_pred_labels = val_outputs.detach().cpu()
                    val_pred_labels = torch.argmax(val_pred_labels, dim=1)
                    val_pred_labels =val_pred_labels.reshape(len(val_pred_labels), 1)
                    val_actual_labels = val_labels.cpu().numpy()
                    val_actual_labels = val_actual_labels.reshape(len(val_actual_labels), 1)
                    val_predictions.append(val_pred_labels)
                    val_actuals.append(val_actual_labels)
                epoch_loss_val /= val_step
                epoch_loss_val_values.append(epoch_loss_val)         ## to plot validation loss, added by changqia
                
                val_actuals, val_predictions = np.vstack(val_actuals), np.vstack(val_predictions)
                val_cm = metrics.confusion_matrix(val_actuals, val_predictions)
        
                ## to plot validation metrics, added by changqia
                epoch_acc_val, epoch_sensi_val, epoch_speci_val = confusion_metrics(conf_matrix=val_cm, debug=True)
                epoch_acc_val_values.append(epoch_acc_val)
                epoch_sensi_val_values.append(epoch_sensi_val)
                epoch_speci_val_values.append(epoch_speci_val)

                metric = num_correct / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", metric, epoch + 1)
        else:
            epoch_loss_val_values.append(np.nan)
            epoch_acc_val_values.append(np.nan)
            epoch_sensi_val_values.append(np.nan)
            epoch_speci_val_values.append(np.nan)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

    return (epoch_loss_train_values, epoch_loss_val_values, 
            epoch_acc_train_values, epoch_acc_val_values,
            epoch_sensi_train_values, epoch_sensi_val_values, 
            epoch_speci_train_values, epoch_speci_val_values, epoch_mem_usage_values)
    

def test(test_dl, model, device):
    """Peforms the testing of the trained model on the given test DataLoader instance"""
    model.load_state_dict(torch.load('best_metric_model.pth'))
    model.eval()
    y_true = list()
    y_pred = list()
    with torch.no_grad():
        for test_data in test_dl:
            test_images, test_labels = test_data[0].to(device=device, dtype=torch.float), test_data[1].to(device=device, dtype=torch.float)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())

    print("test completed")
    print("y_pred: ", len(y_pred))
    print("y_true: ", len(y_true))
    