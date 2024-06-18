from torchvision import models, transforms
import torch

import os

from PIL import Image


def resnet_model(file_path):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    directory_path = file_path 
    files = os.listdir(directory_path)

    all_results = {}

    for file_name in files:
        image_path = os.path.join(directory_path, file_name)
        image = Image.open(image_path)
        
        if image.mode == 'L':
            image = image.convert('RGB')
        
        batch = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(batch).squeeze(0).softmax(0)
            top5_prob, top5_catid = torch.topk(prediction, 5)
        
        categories = weights.meta["categories"]
        results = [(categories[catid], f"{100 * prob:.1f}%") for prob, catid in zip(top5_prob, top5_catid)]

        all_results[file_name] = results
    
    return all_results



def vgg16_model(file_path):
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    directory_path = file_path 
    files = os.listdir(directory_path)

    all_results = {}

    for file_name in files:
        image_path = os.path.join(directory_path, file_name)
        image = Image.open(image_path)
        
        if image.mode == 'L':
            image = image.convert('RGB')
        
        batch = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(batch).squeeze(0).softmax(0)
            top5_prob, top5_catid = torch.topk(prediction, 5)
        
        categories = weights.meta["categories"]
        results = [(categories[catid], f"{100 * prob:.1f}%") for prob, catid in zip(top5_prob, top5_catid)]

        all_results[file_name] = results
    
    return all_results




def inceptionv3_model(file_path):
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    
    directory_path = file_path 
    files = os.listdir(directory_path)

    all_results = {}

    for file_name in files:
        image_path = os.path.join(directory_path, file_name)
        image = Image.open(image_path)
        
        if image.mode == 'L':
            image = image.convert('RGB')
        
        batch = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(batch).squeeze(0).softmax(0)
            top5_prob, top5_catid = torch.topk(prediction, 5)
        
        categories = weights.meta["categories"]
        results = [(categories[catid], f"{100 * prob:.1f}%") for prob, catid in zip(top5_prob, top5_catid)]

        all_results[file_name] = results
    
    return all_results
