# Final Project for Udacity AIPND course
# 11/1/2018
# by Jizong Liang
# load all packages and dependency
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchvision import datasets, transforms, models
import json
import collections
from PIL import Image
import numpy as np
from torch.autograd import Variable
import argparse
from time import time
import datetime

# main function
def main():
    # call argument function
    in_args = get_input_args()
    # call data loader function
    train_data,valid_data,trainloader,validloader = fileloader(in_args.data_dir)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    # set up model architecture
    if in_args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif in_args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024

    # set up parameters and classifier
    for param in model.parameters():
        param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                          ('linear1', nn.Linear(input_size, in_args.hidden_units[0])),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('linear2', nn.Linear(in_args.hidden_units[0], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier

    # set up criterion and optimizer for model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
    if torch.cuda.is_available():
        # Move model parameters to the GPU
        print("GPU")
        model.cuda()
    else:
        print("CPU")
        model.cpu()
    # train the network with input parameters

    do_deep_learning(model, trainloader,validloader, in_args.epochs, 40, criterion, optimizer, in_args.gpu)

    # shift back to cpu mode
    model.to('cpu')

    # check acuracy of model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in validloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    # save model checkpoint
    torch.save({'input_size': input_size,
                'output_size': 102,
                'hidden_layers': [in_args.hidden_units[0]],
                'state_dict': model.state_dict(),
                'optimizer.state_dict': optimizer.state_dict,
                'class_to_idx': train_data.class_to_idx,
                'arch':in_args.arch
                }, in_args.save_dir)


# function for calling arguments
def get_input_args():
    """
        Retrieves and parses the command line arguments created and defined using
        the argparse module. This function returns these arguments as an
        ArgumentParser object.
        Parameters:
         None - simply using argparse module to create & store command line arguments
        Returns:
         parse_args() -data structure that stores the command line arguments object
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Dataset directory')
    parser.add_argument('--save_dir', default='.', help='Checkpoint save directory')
    parser.add_argument('--arch', choices=['vgg16', 'densenet121'], default='densenet121', help='Architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Training learning rate')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[512], help='Hidden layers')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action="store_true", help='Use GPU')
    return parser.parse_args()


# set up file loading directory
# in_args.data_dir
def fileloader(path):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # set transforms for train and test sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # loading the train and valid datasets

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(test_dir, transform=valid_transforms)


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    return train_data, valid_data,trainloader,validloader

# function for training model
def do_deep_learning(model, trainloader, validloader,epochs, print_every, criterion, optimizer, device=False):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    # model.to('cuda')



    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                test_loss = 0
                for ii, (inputs, labels) in enumerate(validloader):
                # move input and label tensors to the GPU
                    if torch.cuda.is_available() and device:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    inputs = Variable(inputs)
                    labels = Variable(labels)

                    output = model.forward(inputs)
                    test_loss += criterion(output, labels).data[0]

                # calculate the accuracy
                # model's output is LogSoftmax, take exponential to get the probabilities
                    probabilities = torch.exp(output).data
                # class with highest probability is our predicted class, compare with true label
                    equality = (labels.data == probabilities.max(1)[1])
                # accuracy is the number of correct predictions divided by all the predictions, so just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()




                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader))
                     )

                running_loss = 0
                model.train()


# function for testing accuracy

def check_accuracy_on_test(validloader,model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in validloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    main()
