import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#matplotlib inline

device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

#resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
resnet = torch.load('10.pt')

resnet.eval().to(device)

from torchvision import transforms
#
# Create a preprocessing pipeline
#
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
#
# Pass the image for preprocessing and the image preprocessed
#
img_cat = Image.open("Food_project/test_steak/test/temp/91432.jpg").convert('RGB')
img_cat_preprocessed = preprocess(img_cat)
#
# Reshape, crop, and normalize the input tensor for feeding into network for evaluation
#
batch_img_cat_tensor = torch.unsqueeze(img_cat_preprocessed, 0)

resnet.eval()
#
# Get the predictions of image as scores related to how the loaded image
# matches with 1000 ImageNet classes. The variable, out is a vector of 1000 scores
#
out = resnet(batch_img_cat_tensor)

# Load the file containing the 1,000 labels for the ImageNet dataset classes
#
with open('Food_project/test_steak/test/labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]
#
# Find the index (tensor) corresponding to the maximum score in the out tensor.
# Torch.max function can be used to find the information
#
_, index = torch.max(out, 1)
#
# Find the score in terms of percentage by using torch.nn.functional.softmax function
# which normalizes the output to range [0,1] and multiplying by 100
#
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
#
# Print the name along with score of the object identified by the model
#
print(labels[index[0]], percentage[index[0]].item())
#
# Print the top 5 scores along with the image label. Sort function is invoked on the torch to sort the scores.
#
_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
