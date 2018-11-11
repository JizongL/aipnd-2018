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
from PIL import Image
import numpy as np
import json
from torch.autograd import Variable
import collections
import argparse

# main function
def main():
    # call argument function
    in_args = get_input_args()
    # load label_map
    with open(in_args.label_map, 'r') as f:
        cat_to_name = json.load(f)
    # load trained model checkpoint file
    model,class_to_idx = load_checkpoint(in_args.path_to_model)
    idx_to_class = {i:k for k, i in class_to_idx.items()}
    # set image path
    img_path = in_args.path_to_image# example 'flowers/test/101/image_07949.jpg'
    # set k prediction
    topk = in_args.topk
    # image processing
    image = process_image(Image.open(img_path))
    image = torch.FloatTensor([image])
    # shift to eval mode
    model.eval()
    # prepare image
    output = model.forward(Variable(image))
    ps = torch.exp(output).data.numpy()[0]
    # obtain prediction
    topk_index = np.argsort(ps)[-topk:][::-1]
    topk_class = [idx_to_class[x] for x in topk_index]
    topk_prob = ps[topk_index]


    # print prediction result
    print(topk_prob,'\n',topk_class)

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

    parser = argparse.ArgumentParser(description = 'A trained model to predict images')
    parser.add_argument('path_to_model', type = str,help='input path of saved model checkpoint')
    parser.add_argument('path_to_image', type = str, help='Enter path of image to the model ')
    parser.add_argument('--topk', default=5, type = int, help='print top k prediction')
    parser.add_argument("--label_map", type=str, help="json file containing mapping for integer label to string labels", default='cat_to_name.json')
    parser.add_argument('--gpu', action="store_true", help='enable gpu for prediction')
    return parser.parse_args()


# load checkpoint helper
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # load the right model
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)

    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)

    # setup classifier
    input_size = checkpoint['input_size']
    hidden_layers = checkpoint['hidden_layers']
    output_size = checkpoint['output_size']
    classifier = nn.Sequential(OrderedDict([
                          ('linear1', nn.Linear(input_size, hidden_layers[0])),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('linear2', nn.Linear(hidden_layers[0], output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier

    for param in model.parameters():
        param.requires_grad = False
    return model ,checkpoint['class_to_idx']







# function for processing images

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((
        size[0] //2 - (224/2),
        size[1] //2 - (224/2),
        size[0] //2 + (224/2),
        size[1] //2 + (224/2))
    )
    np_image = np.array(image)/255
    np_image[:,:,0] = (np_image[:,:,0] - 0.485)/(0.229)
    np_image[:,:,1] = (np_image[:,:,1] - 0.456)/(0.224)
    np_image[:,:,2] = (np_image[:,:,2] - 0.406)/(0.225)
    np_image = np.transpose(np_image, (2,0,1))

    return np_image

# image show helper
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (std * image) + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    if title:
        ax.set_title(title)
    ax.axis('off')
    ax.imshow(image)

    return ax





# function for prediction

def predict(image_path,idx_to_class, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = process_image(Image.open(image_path))
    image = torch.FloatTensor([image])
    model.eval()
    output = model.forward(Variable(image))
    ps = torch.exp(output).data.numpy()[0]

    topk_index = np.argsort(ps)[-topk:][::-1]
    topk_class = [idx_to_class[x] for x in topk_index]
    topk_prob = ps[topk_index]

    return topk_prob, topk_class






if __name__ == "__main__":
    main()
