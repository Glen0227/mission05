import torchvision.models as models
import torch
import torch.nn as nn
from data.data_loader import data_ready, data_loader_set

def prep_finetune_GoogLeNet(class_names):
    model = models.googlenet(weights='DEFAULT')
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, len(class_names))

    for name, param in model.named_parameters():
        if 'fc' in name or 'inception5' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model