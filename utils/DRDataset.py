import ast
import pandas as pd
import torch as th
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torch.utils.data import Dataset

class DRDataset(Dataset):
    """
    Custom Dataset. Loads the images and labels from the given path and csv file.

    methods:
        __init__: Initializes the dataset. Stores the image names and labels in lists.
        __len__: Returns the size of the dataset.
        __getitem__: Loads image triplet and label based on the given index. Applies the transforms to the images and the labels. Flips laterality if necessary.
        get_item_name: Helper function to get the image names of the triplet based on the given index. Used in visualization tools.
        get_item_laterality: Helper function to get the laterality of the image triplet based on the given index.
    """
    def __init__(self, img_path: str, csv_file: str, labels: tuple=('proliferation', {'NPDR': 0, 'PDR': 1}), transform: tuple=(None, None), label_transorm=None, config: str='both', eval: bool=False):
        """
        Initializes the dataset. Stores the image names and labels in lists.

        args:
            img_path: str, path to the images
            csv_file: str, path to the csv file
            labels: tuple, label type and label dictionary
            transform: tuple, transform and size of the input of the model
            label_transform: function, transform for the labels
            config: str, configuration of the dataset (both, L or R) - both: random configuration flips, L/R: All images are flipped to the same laterality
            eval: bool, evaluation mode (True for test set) - eval == True => no laterality configuration flipping
        """

        # Save init parameters
        self.img_path = img_path
        self.annotations = pd.read_csv(csv_file, converters={'frame_of_reference_UID': ast.literal_eval, 'image_hash': ast.literal_eval, 'image_uuid': ast.literal_eval})
        assert all(item in self.annotations.columns for item in ['patient_hash', 'laterality', 'image_uuid']), 'CSV does not provide enough information'

        assert labels[0] in ['proliferation', 'laterality', 'image_type', 'image_orientation'], 'label_type must be one of: proliferation, laterality, image_type, image_orientation'
        self.label_type = labels[0]
        self.label_dict = labels[1]
        
        self.transform = transform
        self.label_transform = label_transorm
        self.config = config
        self.eval = eval

        # Store image names and labels
        self.img_names = self.annotations['image_uuid'].tolist()
        self.img_labels = [self.label_dict[label] for label in self.annotations[self.label_type]]
    

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Loads image triplet and label based on the given index. Applies the transforms to the images and the labels. Flips laterality if necessary.

        args:
            idx: int, index of the image triplet

        returns:
            dict, image triplet (th.Tensor, each) and label
        """
        fundus = read_image(self.img_path + '/' + self.img_names[idx]['fundus'] + '.png', ImageReadMode.RGB)
        horizontal = read_image(self.img_path + '/' + self.img_names[idx]['horizontal'] + '.png', ImageReadMode.RGB)
        vertical = read_image(self.img_path + '/' + self.img_names[idx]['vertical'] + '.png', ImageReadMode.RGB)
        if self.config == 'both': # Random configuration flips (fundus and horizontal are horizontally flipped).
            if not self.eval: # No flipping in evaluation mode.
                rand = th.rand(1)
                if rand < 0.5:
                    fundus = fundus.flip(2)
                    horizontal = horizontal.flip(2)
        else: # OS <-> OD => fundus and horizontal from one laterality are horizontally flipped
            if self.annotations['laterality'][idx] != self.config:
                fundus = fundus.flip(2)
                horizontal = horizontal.flip(2)
        if self.transform[0]:
                fundus = self.transform[0](fundus, self.transform[1])
                horizontal = self.transform[0](horizontal, self.transform[1])
                vertical = self.transform[0](vertical, self.transform[1])
        label = self.img_labels[idx]
        if self.label_transform:
            label = self.label_transform(label)

        return {'fundus': fundus, 'horizontal': horizontal, 'vertical': vertical, 'label': label}
    
    def get_item_name(self, idx):
        """
        Helper function to get the image names of the triplet based on the given index. Used in visualization tools.

        args:
            idx: int, index of the image triplet

        returns:
            dict, image names
        """
        fundus = self.img_path + '/' + self.img_names[idx]['fundus'] + '.png'
        horizontal = self.img_path + '/' + self.img_names[idx]['horizontal'] + '.png'
        vertical = self.img_path + '/' + self.img_names[idx]['vertical'] + '.png'
        return {'fundus': fundus, 'horizontal': horizontal, 'vertical': vertical}
    
    def get_item_laterality(self, idx):
        """
        Helper function to get the laterality of the image triplet based on the given index.

        args:
            idx: int, index of the image triplet

        returns:
            str, laterality of the image triplet
        """
        return self.annotations['laterality'][idx]
