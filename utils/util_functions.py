import torch as th
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score # Attention: Imbalanced data
from utils.DRDataset import DRDataset

def load_transform(img: th.Tensor, size: int):
    """
    Transforms the image to the input size of the model, scales from 0-255 to 0-1, and normalizes it. Has to be applied to each image before it is passed to the model.

    args:
        img: th.Tensor, image to be transformed
        size: int, size of the input of the model

    returns:
        th.Tensor, transformed image
    """
    input_size = (size, size)
    img = img/255.0 # Convert from [0, 255] to [0, 1]
    load = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return load(img)


def train_transform(img: th.Tensor, size: int): # No RandomHorizontalFlip (is already included in DRDataset)
    """
    Applies load_transform and adds augmentation (a random rotation and shift to the image).

    args:
        img: th.Tensor, image to be transformed
        size: int, size of the input of the model

    returns:
        th.Tensor, transformed image
    """
    augmentation = transforms.Compose([
        transforms.RandomAffine(10, translate=(0.05, 0.05)), # Rotation by max 10° and shift by max 5%
    ])
    return augmentation(load_transform(img, size))


def test_transform(img: th.Tensor, size: int):
    """
    Applies load_transform to the image.

    args:
        img: th.Tensor, image to be transformed
        size: int, size of the input of the model

    returns:
        th.Tensor, transformed image
    """
    return load_transform(img, size)


def load_datasets(root: str, size:int, fold: int=0, csv_suffix=''):
    """
    Loads the DRDatasets for training and validation (fold != 0) or testing (fold == 0). Sets the transforms, image size and the evaluation mode for the datasets. 

    args:
        root: str, path to the root directory of the dataset
        size: int, size of the input of the model
        fold: int, fold of the cross-validation (0 for testing)
        csv_suffix: str, suffix of the csv files (e.g. '_refined' for the refined dataset)

    returns:
        train: DRDataset, training dataset
        val: DRDataset, validation dataset (None if fold == 0)
        test: DRDataset, testing dataset (None if fold != 0)
    """
    labels = ('proliferation', {'NPDR': 0, 'PDR': 1}) # TODO: Allow for other labels without manually changing the code
    if fold != 0:
        train = DRDataset(f'{root}/images', f'{root}/sets/train_{fold}{csv_suffix}.csv', labels=labels, transform=(train_transform, size))
        val = DRDataset(f'{root}/images', f'{root}/sets/val_{fold}{csv_suffix}.csv', labels=labels, transform=(test_transform, size), eval=True)
        test = None
    else:
        train = DRDataset(f'{root}/images', f'{root}/sets/train{csv_suffix}.csv', labels=labels, transform=(train_transform, size))
        val = None
        test = DRDataset(f'{root}/images', f'{root}/sets/test{csv_suffix}.csv', labels=labels, transform=(test_transform, size), eval=True)
    return train, val, test


def load_backbone(backbone_str: str, script_type: str):
    """
    Loads the backbone model and the corresponding weights for the given backbone string. Note: The weights are only loaded if the script runs single modality training to save resources.

    args:
        backbone_str: str, string of the backbone model
        script_type: str, type of the script (single or multi)

    returns:
        backbone: models, backbone model
        init_weights: models, weights of the backbone model (None if script_type == 'multi')
    """
    init_weights = None
    if backbone_str == 'resnet18':
        backbone = models.resnet18
        if script_type == 'single':
            init_weights = models.ResNet18_Weights.IMAGENET1K_V1
    elif backbone_str == 'resnet50':
        backbone = models.resnet50
        if script_type == 'single':
            init_weights = models.ResNet50_Weights.IMAGENET1K_V2
    elif backbone_str == 'efficientnet':
        backbone = models.efficientnet_v2_l
        if script_type == 'single':
            init_weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f'Backbone {backbone_str} not implemented (yet).')
    
    return backbone, init_weights
        
        
@th.no_grad()
def evaluation(model, loader, script_type, image_method, device, threshold=0.5):
    """
    Evaluates a given model on a given dataset. Computes the accuracy, the ROC curve, the AUC, the precision-recall curve, the AP and the confusion matrix.
    The threshold is variable.

    args:
        model: models, model to be evaluated
        loader: DataLoader, dataloader of the dataset to be evaluated
        script_type: str, type of the script (single or multi)
        image_method: str, method to access the image in the batch (e.g. 'fundus')
        device: str, device on which the model is evaluated
        threshold: float, threshold for the prediction

    returns:
        dict, evaluation results
    """
    y_true = th.empty(0)
    y_soft = th.empty(0)
    y_pred = th.empty(0)

    for batch in loader:
        if script_type == 'single':
            batch_x, y_t = batch[image_method], batch['label']
            batch_x = batch_x.to(device)
        else:
            batch_x, y_t = batch, batch['label'] 
            batch_x = {k: v.to(device) for k, v in batch_x.items() if k != 'label'}
        y_true = th.cat((y_true,y_t))
        y_s = model(batch_x).cpu()[:,1] # Only for class 1
        y_soft = th.cat((y_soft,y_s))
        y_p = (y_s > threshold).int()
        y_pred = th.cat((y_pred,y_p))

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_soft)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_soft)

    return {'accuracy': accuracy_score(y_true, y_pred),
            'tpr': tpr,
            'fpr': fpr,
            'threshold_roc': thresholds_roc,
            'auc': roc_auc_score(y_true, y_soft),
            'precision': precision,
            'recall': recall,
            'threshold_pr': thresholds_pr,
            'ap': average_precision_score(y_true, y_soft),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0,1])}