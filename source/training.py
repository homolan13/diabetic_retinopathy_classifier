import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import time as tt
from tqdm import tqdm

import torch as th
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.Models import *
from utils.util_functions import load_datasets, load_backbone, evaluation

def get_device():
    """
    Returns the device to be used for training and evaluation.

    returns:
        device: torch.device
    """
    if th.cuda.is_available(): # Nvidia GPU (for cluster usage)
        device = th.device('cuda')
    elif th.backends.mps.is_available(): # Apple Silicon (for local usage)
        device = th.device('mps')
    else:
        device = th.device('cpu')
    
    return device

def get_args():
    """
    Parses the command line arguments.

    returns:
        TRAINING_TYPE: str, type of training ("full", "cv" or "none")
        BACKBONE: str, backbone to be used ("resnet18", "resnet50" or "efficientnet")
        FEATURE_VECTOR_SIZE: int, size of the feature vector (0 - 1280)
        IMAGE_METHOD: str, image modality or fusion method ("fundus", "horizontal", "vertical", "stack" or "sum")
        IMAGE_SIZE: int, image size (16 - 512)
        NAME_SUFFIX: str, suffix for the model name
        BATCH_SIZE: int, batch size (1 - 64)
        SCRIPT_TYPE: str, type of script ("single" or "multi")
    """
    parser = argparse.ArgumentParser(description='Train a model for DR classification')
    parser.add_argument('-t', '--type', type=str, help='Type of training ("full", "cv" or "none"). "none" is evaluation-only mode. Required.', required=True)
    parser.add_argument('-b', '--backbone', type=str, help='Backbone to be used ("resnet18", "resnet50" or "efficientnet"). Required.', required=True)
    parser.add_argument('-v', '--feature_vector_size', type=int, help='Size of the feature vector (0 - 1280). Required.', required=True)
    parser.add_argument('-m', '--image_method', type=str, help='Image modality or fusion method ("fundus", "horizontal", "vertical", "stack" or "sum"). Required.', required=True)
    parser.add_argument('-s', '--batch_size', type=int, help='Batch size (1 - 64). Default: 1.', required=False, default=1)
    parser.add_argument('-i', '--image_size', type=int, help='Image size (16 - 512). Default: 16.', required=False, default=16)
    parser.add_argument('-n', '--name_suffix', type=str, help='Suffix for the model name. Default: "".', required=False, default='')
    parser.add_argument('-d', '--data_suffix', type=str, help='Suffix for the data CSV files ("" or "_refined"). Default: "".', required=False, default='')
    args = vars(parser.parse_args())

    TRAINING_TYPE = args['type'].lower()
    BACKBONE = args['backbone'].lower()
    FEATURE_VECTOR_SIZE = args['feature_vector_size']
    IMAGE_METHOD = args['image_method'].lower()
    BATCH_SIZE = args['batch_size']
    IMAGE_SIZE = args['image_size']
    NAME_SUFFIX = args['name_suffix']
    DATA_SUFFIX = args['data_suffix']

    assert TRAINING_TYPE in ['full', 'cv', 'none'], 'Training type must be either "full", "cv" or "none"'
    assert BACKBONE in ['efficientnet', 'resnet50', 'resnet18'], 'Backbone must be either "efficientnet", "resnet-18" or "resnet-50"'
    assert FEATURE_VECTOR_SIZE in range(1281), 'Description vector size must be between 0 and 1280'
    assert IMAGE_METHOD in ['fundus', 'horizontal', 'vertical', 'stack', 'sum'], 'Image modality must be either "fundus", "horizontal", "vertical", "stack" or "sum"'
    assert BATCH_SIZE in range(1, 65), 'Batch size must be between 1 and 64'
    assert IMAGE_SIZE in range(16, 513), 'Image size must be between 16 and 512'
    assert NAME_SUFFIX == '' or NAME_SUFFIX[0] == '_', 'Suffix for the model name must be empty or start with an underscore'
    assert DATA_SUFFIX == '' or DATA_SUFFIX[0] == '_', 'Suffix for the data CSV files must be empty or start with an underscore'

    if IMAGE_METHOD in ['fundus', 'horizontal', 'vertical']:
        SCRIPT_TYPE = 'single'
    else:
        SCRIPT_TYPE = 'multi'

    return TRAINING_TYPE, BACKBONE, FEATURE_VECTOR_SIZE, IMAGE_METHOD, BATCH_SIZE, IMAGE_SIZE, NAME_SUFFIX, DATA_SUFFIX, SCRIPT_TYPE

def setup(fold:int=0):
    """
    Sets up the training and evaluation environment.

    args:
        fold: int, optional, default=0

    returns:
        config: dict

    fold is an optional parameter that is necessary for denoting the current fold in cross-validation (TRAINING_TYPE='cv') mode. fold is set to 0 for full training (TRAINING_TYPE='full').
    config is a dictionary containing the following keys:
        root: str, path to the root directory of the dataset
        output_path_model: str, path to the directory where the model state will be saved
        model_name: str, name of the model state file
        summary: SummaryWriter, summary writer for TensorBoard
    """
    root = f'../data'
    if TRAINING_TYPE == 'full':
        output_path_model = f'model_states{NAME_SUFFIX}/full'
        output_path_event = f'event_files{NAME_SUFFIX}/full'
        model_name = f'{output_path_model}/{BACKBONE}_{FEATURE_VECTOR_SIZE}_{IMAGE_METHOD}_{IMAGE_SIZE}.pt'
        filename_suffix = f'_{BACKBONE}_{FEATURE_VECTOR_SIZE}_{IMAGE_METHOD}_{IMAGE_SIZE}'
    else:
        output_path_model = f'model_states{NAME_SUFFIX}/cv'
        output_path_event = f'event_files{NAME_SUFFIX}/cv'
        model_name = f'{output_path_model}/{BACKBONE}_{FEATURE_VECTOR_SIZE}_{IMAGE_METHOD}_{IMAGE_SIZE}_{fold}.pt'
        filename_suffix = f'_{BACKBONE}_{FEATURE_VECTOR_SIZE}_{IMAGE_METHOD}_{IMAGE_SIZE}_{fold}'
    if TRAINING_TYPE == 'none':
        output_path_model = f'model_states{NAME_SUFFIX}/full' # In this case, an existing model is needed and no new model is saved
        output_path_event = f'event_files{NAME_SUFFIX}/inference_only'
        model_name = f'{output_path_model}/{BACKBONE}_{FEATURE_VECTOR_SIZE}_{IMAGE_METHOD}_{IMAGE_SIZE}.pt'
        filename_suffix = f'_{BACKBONE}_{FEATURE_VECTOR_SIZE}_{IMAGE_METHOD}_{IMAGE_SIZE}'
    summary = SummaryWriter(output_path_event, purge_step=0, filename_suffix=filename_suffix)

    config = {
        'root': root,
        'output_path_model': output_path_model,
        'model_name': model_name,
        'summary': summary
    }

    return config

def train(config:dict, fold:int=0, num_epochs:int=200):
    """
    Trains a model.

    args:
        config: dict
        fold: int, optional, default=0
        num_epochs: int, optional, default=200

    config is a dictionary containing the following keys:
        root: str, path to the root directory of the dataset
        output_path_model: str, path to the directory where the model state will be saved
        model_name: str, name of the model state file
        summary: SummaryWriter, summary writer for TensorBoard
    fold is an optional parameter that is necessary for denoting the current fold in cross-validation (TRAINING_TYPE='cv') mode. fold is set to 0 for full training (TRAINING_TYPE='full').
    num_epochs is an optional parameter that denotes the number of epochs to train the model for.
    """
    # Unpack config
    root = config['root']
    output_path_model = config['output_path_model']
    model_name = config['model_name']
    summary = config['summary']

    # Load data
    data_train, _, _ = load_datasets(root, fold=fold, size=IMAGE_SIZE, csv_suffix=DATA_SUFFIX)
    num_workers = 4 # 0 for debugging, else 1 per CPU core
    loader_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    epoch_length = len(loader_train)

    # Instantiate model
    backbone, init_weights = load_backbone(BACKBONE, SCRIPT_TYPE)
    if SCRIPT_TYPE == 'single':
        model = SingleModalityClassifier(FEATURE_VECTOR_SIZE, backbone, weights=init_weights).to(DEVICE)
    else:
        if fold == 0:
            model_weights = {k: f'{output_path_model}/{BACKBONE}_{FEATURE_VECTOR_SIZE}_{k}_{IMAGE_SIZE}.pt' for k in ['fundus', 'horizontal', 'vertical']}
        else:
            model_weights = {k: f'{output_path_model}/{BACKBONE}_{FEATURE_VECTOR_SIZE}_{k}_{IMAGE_SIZE}_{fold}.pt' for k in ['fundus', 'horizontal', 'vertical']} # Uses different folds as initialization
        model = MultiModalityClassifier(FEATURE_VECTOR_SIZE, backbone, weights=model_weights, stack_or_sum=IMAGE_METHOD, device=DEVICE).to(DEVICE)

    # Define optimizer and scheduler
    optimizer = th.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs//3, eta_min=1e-6, last_epoch=-1)

    # Add constant to summary writer
    summary.add_scalar('fold', fold)
    summary.add_scalar('num_params', sum(params.numel() for params in model.parameters() if params.requires_grad))
    summary.add_scalar('epoch_length', epoch_length)

    # Actual training
    it = 0
    start_time = tt.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        if fold == 0:
            print(f'\n{IMAGE_METHOD.upper()} - Starting epoch {epoch + 1} (of {num_epochs})...')
        else:
            print(f'\n{IMAGE_METHOD.upper()} - Starting epoch {epoch + 1} (of {num_epochs} in fold {fold}/5)...')
        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]:.1e}')
        summary.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
        for batch in tqdm(loader_train):
            it += 1
            model.train()
            if SCRIPT_TYPE == 'single':
                batch_x, batch_y = batch[IMAGE_METHOD], batch['label'] 
                batch_x = batch_x.to(DEVICE)
            else:
                batch_x, batch_y = batch, batch['label'] 
                batch_x = {k: v.to(DEVICE) for k, v in batch_x.items() if k != 'label'}
            y_true = batch_y.to(DEVICE).float()
            y_score = model(batch_x)[:,1] # Use only class 1
            loss = nn.functional.binary_cross_entropy(y_score, y_true)
            epoch_loss += loss.cpu().item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (it % 10) == 0: # Log the training loss once every x iterations
                summary.add_scalar('training_loss', loss.cpu().item(), it)
                scheduler.step()

        print(f'Mean training loss (epoch {epoch+1}): {epoch_loss/epoch_length:.3f}') # Print mean training loss for the epoch
        # Save model after each epoch (checkpointing)
        if not os.path.exists(output_path_model):
            os.makedirs(output_path_model)
        th.save(model.state_dict(), model_name)
    
    summary.add_scalar('training_time', tt.time() - start_time)
    print(f'{IMAGE_METHOD.upper()} - Training finished. Saving model state...\n')

    # Save model state after training
    if not os.path.exists(output_path_model):
        os.makedirs(output_path_model)
    th.save(model.state_dict(), model_name)

    return
    

def validate(config:dict, fold:int=0):
    """
    Evaluates a model.

    args:
        config: dict
        fold: int, optional, default=0

    config is a dictionary containing the following keys:
        root: str, path to the root directory of the dataset
        output_path_model: str, path to the directory where the model state will be saved
        model_name: str, name of the model state file
        summary: SummaryWriter, summary writer for TensorBoard
    fold is an optional parameter that is necessary for denoting the current fold in cross-validation (TRAINING_TYPE='cv') mode. fold is set to 0 for full training (TRAINING_TYPE='full').

    """
    # Unpack config
    root = config['root']
    output_path_model = config['output_path_model']
    model_name = config['model_name']
    summary = config['summary']

    # Load data
    num_workers = 4 # 0 for debugging, else 1 per CPU core
    _, data_val, data_test = load_datasets(root, fold=fold, size=IMAGE_SIZE, csv_suffix=DATA_SUFFIX)
    if TRAINING_TYPE == 'cv':
        loader_val = DataLoader(data_val, batch_size=BATCH_SIZE, num_workers=num_workers)
    else: # TRAINING_TYPE == 'full' or TRAINING_TYPE == 'none'
        loader_val = DataLoader(data_test, batch_size=BATCH_SIZE, num_workers=num_workers)
    
    # Instantiate model and load model state
    backbone, _ = load_backbone(BACKBONE, SCRIPT_TYPE) # Weights are not needed for evaluation, as the model state is loaded below
    if SCRIPT_TYPE == 'single':
        model = SingleModalityClassifier(FEATURE_VECTOR_SIZE, backbone).to(DEVICE)
    else:
        model = MultiModalityClassifier(FEATURE_VECTOR_SIZE, backbone, stack_or_sum=IMAGE_METHOD, device=DEVICE).to(DEVICE)
    model.load_state_dict(th.load(model_name, map_location=DEVICE))

    # Evaluation
    if fold == 0:
        print(f'\n{IMAGE_METHOD.upper()} - Starting evaluation...')
    else:
        print(f'\n{IMAGE_METHOD.upper()} - Starting evaluation (fold {fold}/5)...')
    model.eval()
    metrics = evaluation(model, loader_val, SCRIPT_TYPE, IMAGE_METHOD, DEVICE) # TODO: Option to adjust the threshold

    # Add metrics to summary writer
    summary.add_scalar('test_accuracy', metrics['accuracy'])
    for step, (tp, fp, thr) in enumerate(zip(metrics['tpr'], metrics['fpr'], metrics['threshold_roc'])):
        summary.add_scalar('test_tpr', tp, step)
        summary.add_scalar('test_fpr', fp, step)
        summary.add_scalar('test_threshold_roc', thr, step)
    summary.add_scalar('test_auc', metrics['auc'])
    for step, (pr, re, thr) in enumerate(zip(metrics['precision'], metrics['recall'], metrics['threshold_pr'])):
        summary.add_scalar('test_precision', pr, step)
        summary.add_scalar('test_recall', re, step)
        summary.add_scalar('test_threshold_pr', thr, step)
    summary.add_scalar('test_ap', metrics['ap'])
    summary.add_scalar('confusion_matrix_00', metrics['confusion_matrix'][0,0])
    summary.add_scalar('confusion_matrix_01', metrics['confusion_matrix'][0,1])
    summary.add_scalar('confusion_matrix_10', metrics['confusion_matrix'][1,0])
    summary.add_scalar('confusion_matrix_11', metrics['confusion_matrix'][1,1])

    summary.flush() # Flush the summary writer to write the data to the event file (otherwise the last scalars added to the summary writer might not be written to the event file)

    print(f'Evaluation AUC: {metrics["auc"]:.3f}')
    print('Evaluation finished.\n')
    
    return

if __name__ == '__main__':

    DEVICE = get_device()
    TRAINING_TYPE, BACKBONE, FEATURE_VECTOR_SIZE, IMAGE_METHOD, BATCH_SIZE, IMAGE_SIZE, NAME_SUFFIX, DATA_SUFFIX, SCRIPT_TYPE = get_args()

    if TRAINING_TYPE == 'full':
        config = setup()
        train(config)
        validate(config)

    elif TRAINING_TYPE == 'cv':
        for fold in range(1,6):
            config = setup(fold)
            train(config, fold)
            validate(config, fold)

    else: # TRAINING_TYPE == 'none', only works with full training models
        config = setup() # Model needs to be existing
        validate(config)
    