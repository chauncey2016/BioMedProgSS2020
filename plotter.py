__author__ = 'changqia'

import matplotlib.pyplot as plt
import os
import numpy as np

PLOT_CROSS_VAL_COMBINED = -2
PLOT_PERCENTAGE_SPLIT = -1

colors = [ 'b', 'g', 'r', 'c', 'm', 'y', 'k' ]

def mkdir(file_path):
    parent_path = os.path.dirname(file_path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

def accuracy_plot(file_path, *args):
    """
    create the accuracy plot singlely

    Arguments:
        -- file path
        -- args
            1. number of epochs
            2. validation accuracy values
    """
    mkdir(file_path)

    fig = plt.figure(figsize=(9,7))
    epochs = list(range(1,args[0]+1,1))

    #accuracy
    if isinstance(args[1], list):
        for i in range(len(args[1])):
            plt.plot(epochs, args[1][i], colors[i+2] + ',-', label='Validation (fold ' + str(i+1) + ')')
    else:
        pass

    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.title('Accuracy of Epochs')

    fig.legend(loc='center right')
    
    plt.savefig(file_path)
    plt.close()

def metrics_plot(file_path, *args):
    """
        Create the metrics plot

        Arguments:
        -- file path
        -- args
            1. number of epochs
            2. training loss values
            3. validation loss values
            4. training accuracy values
            5. validation accuracy values OR list of lists of accuracy values for each fold
            6. training sensitivity values
            7. validation sensitivity values
            8. training specificity values
            9. validation specificity values
            10. model name
            11. optimizer name
            12. current fold
            12. number of folds
    """
    mkdir(file_path)

    fig = plt.figure(figsize=(9,7))
    epochs = list(range(1,args[0]+1,1))

    #loss
    plt.subplot(221)
    plt.plot(epochs, args[1], colors[1] + ',-', label='Training')
    plt.plot(epochs, args[2], colors[0] + ',-', label='Validation (avg.)' if args[11] == PLOT_CROSS_VAL_COMBINED else 'Validation')
    plt.xlabel('Epoch')
    plt.title('Loss')

    #accuracy
    plt.subplot(222)
    
    if isinstance(args[4][0], list):
        for i in range(len(args[4])):
            plt.plot(epochs, args[4][i], colors[i+2] + ',-', label='Validation (fold ' + str(i+1) + ')')
    else:
        plt.plot(epochs, args[3], colors[1] + ',-')
        plt.plot(epochs, args[4], colors[0] + ',-')

    plt.xlabel('Epoch')
    plt.title('Accuracy')

    #Sensitivity
    plt.subplot(223)
    plt.plot(epochs, args[5], colors[1] + ',-')
    plt.plot(epochs, args[6], colors[0] + ',-')
    plt.xlabel('Epoch')
    plt.title('Sensitivity')

    #Specificity
    plt.subplot(224)
    plt.plot(epochs, args[7], colors[1] + ',-')
    plt.plot(epochs, args[8], colors[0] + ',-')
    plt.xlabel('Epoch')
    plt.title('Specificity')

    plt.subplots_adjust(hspace=0.45)
    if args[11] > PLOT_PERCENTAGE_SPLIT:
        plt.suptitle('Metrics of Epochs with model:{} and optimizer:{} (fold {}/{})'.format(args[9], args[10], args[11]+1, args[12]))
    elif args[11] == PLOT_PERCENTAGE_SPLIT:
        plt.suptitle('Metrics of Epochs with model:{} and optimizer:{}'.format(args[9], args[10]))
    elif args[11] == PLOT_CROSS_VAL_COMBINED:
        plt.suptitle('Metrics of Epochs with model:{} and optimizer:{} ({} folds combined)'.format(args[9], args[10], args[12]))

    fig.legend(loc='center')
    
    plt.savefig(file_path)
    plt.close()

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def last_epoch_plot(file_path, epoch, accuracy, sensitivity, specificity, model, batch_size, image_size):
    """Create the plot of metrics for the last epoch of a run"""
    mkdir(file_path)

    labels = ['Accuracy', 'Sensitivity', 'Specificity']
    metrics_last_epoch = []
    metrics_last_epoch.append(np.round(accuracy, 3))
    metrics_last_epoch.append(np.round(sensitivity, 3))
    metrics_last_epoch.append(np.round(specificity, 3))

    x = np.arange(len(labels))  # the label locations
    width = 0.45  # the width of the bars

    fig, ax = plt.subplots()
    rects = ax.bar(x , metrics_last_epoch, width)

    ax.set_ylabel('Values')
    ax.set_title('Metrics of epoch {}\nmodel: {}, batch size: {}, image size: {}'.format(epoch, model, batch_size, image_size))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    autolabel(rects,ax)

    fig.tight_layout()
    plt.savefig(file_path)
    plt.close()
   

def memory_usage_plot(file_path, epochs, memory_allocated, model, batch_size, image_size):
    """Create the plot of GPU memory usage"""
    mkdir(file_path)

    fig = plt.figure(figsize=(9,7))
    epochs = list(range(1, epochs + 1, 1))

    plt.plot(epochs, memory_allocated, colors[0] + 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Memory allocated (MiB)')
    plt.title('Memory')
    plt.suptitle(f"GPU Memory usage with model {model}, batch size {batch_size}, image size {image_size}")

    fig.legend(loc='center')

    plt.savefig(file_path)
    plt.close()



def architecture_metrics_summary_plot(optimizer):
    """Create the summary plot of architecture metrics"""

    if optimizer != "sgd" and optimizer != "adam":
        raise Exception("wrong optimizer")

    file_path = f"plots/metrics_compare_architectures_{optimizer}.png"
    mkdir(file_path)

    architecture = ['Accuracy', 'Sensitivity', 'Specificity']
    
    if optimizer == "adam":
        metrics_densenet = [0.972, 0.964, 0.98]
        metrics_inception = [0.963, 0.952, 0.975]
        metrics_resenet = [0.975, 0.971, 0.979]
    else:
        metrics_densenet = [0.924, 0.98, 0.868]
        metrics_inception = [0.955, 0.946, 0.965]
        metrics_resenet = [0.943, 0.934, 0.953]

    x = np.arange(len(architecture))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, metrics_densenet, width, label='DenseNet-121')
    rects2 = ax.bar(x , metrics_inception, width, label='Inception v4')
    rects3 = ax.bar(x + width, metrics_resenet, width, label='ResNet-50')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage')
    ax.set_title('Average metrics by architecture with optimizer ' + optimizer)
    ax.set_xticks(x)
    ax.set_xticklabels(architecture)
    ax.legend(loc='lower right')

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    fig.tight_layout()

    plt.savefig(file_path)
    plt.close()


def architecture_gpu_usage_plot():
    """Create the GPU memory usage plot of architectures"""

    file_path = f"plots/memory_usage_compare_architectures.png"
    mkdir(file_path)

    xlabels = ['Architectures']

    usage_densenet_adam = [ 8310 ]
    usage_densenet_sgd = [ 8286 ]
    usage_resnet_adam = [ 6998 ]
    usage_resnet_sgd = [ 6972 ]
    usage_inceptionv4_adam = [ 6854 ]
    usage_inceptionv4_sgd = [ 6830 ]

    x = np.arange(len(xlabels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 2*width, usage_densenet_adam, width, label='DenseNet-121/Adam')
    rects2 = ax.bar(x - width, usage_densenet_sgd, width, label='DenseNet-121/SGD')
    rects3 = ax.bar(x , usage_inceptionv4_adam, width, label='Inception v4/Adam')
    rects4 = ax.bar(x + width, usage_inceptionv4_sgd, width, label='Inception v4/SGD')
    rects5 = ax.bar(x + 2*width, usage_resnet_adam, width, label='ResNet-50/Adam')
    rects6 = ax.bar(x + 3*width, usage_resnet_sgd, width, label='ResNet-50/SGD')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('GPU memory usage (MiB)')
    ax.set_title('GPU memory usage by architecture and optimizer')
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.legend(loc='lower right')

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    autolabel(rects4, ax)
    autolabel(rects5, ax)
    autolabel(rects6, ax)

    fig.tight_layout()

    plt.savefig(file_path)
    plt.close()


if __name__ == "__main__":
    '''used for testing'''
    #architecture_metrics_summary_plot("adam")
    #epochs = 5
    #loss_train1 = [[0.6567, 0.5988, 0.5598, 0.4877, 0.3433],[0.5567, 0.3988, 0.5498, 0.7877, 0.3433],[0.2367, 0.4988, 0.5598, 0.7877, 0.8433]]
    # loss_val1 = [0.5, 0.4, 0.3, 0.2, 0.1]
    # loss_train2 = [0.6567, 0.5988, 0.5598, 0.4877, 0.3433]
    # loss_val2 = [0.5, 0.4, 0.3, 0.2, 0.1]
    # loss_train3 = [0.6567, 0.5988, 0.5598, 0.4877, 0.3433]
    # loss_val3 = [0.5, 0.4, 0.3, 0.2, 0.1]
    # loss_train4 = [0.6567, 0.5988, 0.5598, 0.4877, 0.3433]
    # loss_val4 = [0.5, 0.4, 0.3, 0.2, 0.1]
    # metrics_plot('plots/epoch_loss.png', epochs, loss_train1, loss_val1,loss_train2,loss_val2,loss_train3,loss_val3,loss_train4,loss_val4,'Resnet','Adam')
    architecture_gpu_usage_plot()
    #last_epoch_plot('plots/metrics_last_epoch.png', 200, 0.8, 0.6, 0.5, 'Resenet', 48, 224)
    #accuracy_plot('plots/accuracy_epoch.png', epochs, loss_train1)
