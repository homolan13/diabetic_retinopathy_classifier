import torch as th
from torch import nn
import torchvision.models as models

class SingleModalityClassifier(nn.Module):
    """
    Binary classifier for images of one modality.

    methods:
        forward: Forward pass of the classifier.
    """
    def __init__(self, feature_vector_size: int, backbone: models, weights=None):
        """
        Initializes the classifier.
        Removes the classifier part of the backbone and adds a new custom classifier with given feature vector size.
        Initializes the weights of the backbone with the given weights.

        args:
            feature_vector_size: int, size of the feature vector
            backbone: models, backbone model
            weights: models, weights of the backbone model
        """
        super(SingleModalityClassifier, self).__init__()

        # Save init parameters
        self.feature_vector_size = feature_vector_size
        self.weights = weights
        self.backbone = backbone(weights=self.weights)
        
        # Remove the classifier part of the backbone
        if 'classifier' in self.backbone._modules: # E.g. EfficientNet_V2
            self.in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity() # Remove the classifier part by setting it to Identity
        elif 'fc' in self.backbone._modules: # E.g. ResNet
            self.in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity() # Remove the fc part by setting it to Identity

        # Add a new custom classifier
        if self.feature_vector_size != 0:
            self.feature_vector = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=self.in_features, out_features=self.feature_vector_size, bias=True),
                nn.ReLU(inplace=True)
            )
            self.classifier = nn.Sequential(
                nn.Linear(in_features=self.feature_vector_size, out_features=2, bias=True),
                nn.Softmax(dim=1)
            )
        else:
            self.feature_vector = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=2, bias=True),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.backbone(x)
        x = self.feature_vector(x)
        x = self.classifier(x)
        return x
    

class MultiModalityClassifier(nn.Module):
    """
    Binary classifier for images of multiple modalities (fundus, horizontal and vertical).

    methods:
        forward: Forward pass of the classifier.
    """
    def __init__(self, feature_vector_size: int, backbone: models, weights={'fundus': None, 'horizontal': None, 'vertical': None}, stack_or_sum='stack', device=th.device('cpu')):
        """
        Initializes the classifier by initializing three seperate SingleModalityClassifier with the given weights and adding a final classifier.

        args:
            feature_vector_size: int, size of the feature vector
            backbone: models, backbone model
            weights: dict, weights of the backbone model
            stack_or_sum: str, method to combine the feature vectors of the modalities (stack or sum)
            device: str, device on which the model is evaluated
        """
        super(MultiModalityClassifier, self).__init__()

        # Save init parameters
        self.feature_vector_size = feature_vector_size
        self.backbone = backbone
        self.weights = weights
        self.stack_or_sum = stack_or_sum
        self.device = device

        # Initialize the three SingleModalityClassifier
        self.fundus = SingleModalityClassifier(self.feature_vector_size, self.backbone)
        if self.weights['fundus']:
            self.fundus.load_state_dict(th.load(self.weights['fundus'], map_location=self.device))
        self.horizontal = SingleModalityClassifier(self.feature_vector_size, self.backbone)
        if self.weights['horizontal']:
            self.horizontal.load_state_dict(th.load(self.weights['horizontal'], map_location=self.device))
        self.vertical = SingleModalityClassifier(self.feature_vector_size, self.backbone)
        if self.weights['vertical']:
            self.vertical.load_state_dict(th.load(self.weights['vertical'], map_location=self.device))
    
        # Add a final classifier
        if self.feature_vector_size != 0:
            if self.stack_or_sum == 'stack':
                self.final_classifier = nn.Sequential(
                    nn.Linear(in_features=3*self.feature_vector_size, out_features=2, bias=True),
                    nn.Softmax(dim=1)
                )
            elif stack_or_sum == 'sum':
                self.final_classifier = nn.Sequential(
                    nn.Linear(in_features=self.feature_vector_size, out_features=2, bias=True),
                    nn.Softmax(dim=1)
                )
        else:
            if self.stack_or_sum == 'stack':
                self.final_classifier = nn.Sequential(
                    nn.Linear(in_features=3*self.fundus.in_features, out_features=2, bias=True),
                    nn.Softmax(dim=1)
                )
            elif stack_or_sum == 'sum':
                self.final_classifier = nn.Sequential(
                    nn.Linear(in_features=self.fundus.in_features, out_features=2, bias=True),
                    nn.Softmax(dim=1)
                )

    def forward(self, x):
        if isinstance(x, dict):
            x_fundus = x['fundus']
            x_horizontal = x['horizontal']
            x_vertical = x['vertical']
        else: # x is a tensor. This is necessary for the visualization tools.
            x_fundus = x[:, :3, :, :]
            x_horizontal = x[:, 3:6, :, :]
            x_vertical = x[:, 6:, :, :]
        
        x_fundus = self.fundus.backbone(x_fundus)
        x_fundus = self.fundus.feature_vector(x_fundus)

        x_horizontal = self.horizontal.backbone(x_horizontal)
        x_horizontal = self.horizontal.feature_vector(x_horizontal)

        x_vertical = self.vertical.backbone(x_vertical)
        x_vertical = self.vertical.feature_vector(x_vertical)

        if self.stack_or_sum == 'stack':
            x = th.cat((x_fundus, x_horizontal, x_vertical), dim=1)
        elif self.stack_or_sum == 'sum':
            x = x_fundus + x_horizontal + x_vertical
            
        x = self.final_classifier(x)
        return x    

def kaiming_init(model: nn.Module):
    """
    Kaiming-He initialization of weights. Use only with ReLU!

    args:
        model: nn.Module, model to be initialized
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0.01)