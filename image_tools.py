from torchvision import  transforms
import os

def resize_img(img):

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), 
    ])
    return transform(img)

def normalize_img(img):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return normalize(img)
