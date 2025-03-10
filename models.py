import torch
from torch import nn
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, encoded_img_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_img_size
        
        resnet = torchvision.models.resnet101(pretrained=True)

        #remove the last two layers because we don't classify
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_img_size, encoded_img_size))

        self.fine_tune()
    
    def forward(self, img):

        out = self.resnet(img)
        out = self.adaptive_pool(out)

        #(batch_size, encoded_img_size, encoded_img_size, 2048)
        out = out.permute(0, 2, 3, 1)

        return out
    
    def fine_tune(self, fine_tune=True):
        
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Loop over the children (layers) of the ResNet model, starting from the 6th child onwards
        # self.resnet.children() returns an iterator over the modules (layers) in the ResNet model
        # list(self.resnet.children())[5:] slices the list to only include layers starting from index 5
        for c in list(self.resnet.children())[5:]:
    
            # Loop over the parameters (weights and biases) of the current layer
            for p in c.parameters():
        
                # Set the 'requires_grad' attribute of the parameters
                # 'requires_grad' determines whether the parameters should be updated during backpropagation
                # If fine_tune is True, the parameters will be trainable and updated, otherwise they are frozen
                p.requires_grad = fine_tune