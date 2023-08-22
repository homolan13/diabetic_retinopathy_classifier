import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class TensorboardReader():
    """
    Reads out Tensorboard event files and extracts the relevant information.

    methods:
    """
    def __init__(self, path_to_events_file, id=None):
        """
        Initialization of the TensorboardReader. Stores all values from the event file to variables.

        args:
            path_to_events_file: str, path to the event file
            id: str, id of the TensorboardReader
        """
        self.path_to_events_file = path_to_events_file
        self.filename = os.path.basename(path_to_events_file)
        assert os.path.isfile(path_to_events_file), 'File not found!'
        self.events = EventAccumulator(path_to_events_file)
        self.events.Reload()

        self.id = id

        # Get all tags
        self.tags = self.events.Tags()['scalars']

    def _smooth(self, scalars, weight):
        """
        Smooths the given list of scalars with the given weight.

        args:
            scalars: list, scalars to be smoothed
            weight: float, weight of the smoothing

        returns:
            list, smoothed scalars
        """
        smooth = []
        last = scalars[0]
        for point in scalars:
            s = last * weight + (1 - weight) * point
            smooth.append(s)
            last = s
        return smooth

    def get_fold(self):
        """
        Returns the fold of the cross-validation (0 --> full training or inference only).

        returns:
            int, fold
        """
        if 'fold' in self.tags:
            return int(self.events.Scalars('fold')[0].value)
        else:
            return None
        
    def get_num_params(self):
        """
        Returns the number of parameters of the model.

        returns:
            int, number of parameters
        """
        if 'num_params' in self.tags:
            return int(self.events.Scalars('num_params')[0].value)
        else:
            return None
        
    def get_epoch_length(self):
        """
        Returns the epoch length of the training.

        returns:
            int, epoch length
        """
        if 'epoch_length' in self.tags:
            return int(self.events.Scalars('epoch_length')[0].value)
        else:
            return None

    def get_learning_rate(self):
        """
        Returns the learning rate for each epoch as a list.

        returns:
            list, learning rates
        """
        if 'learning_rate' in self.tags:
            return [e.value for e in self.events.Scalars('learning_rate')]
        else:
            return None

    def get_training_time(self):
        """
        Returns the training time in seconds.

        returns:
            float, training time
        """
        if 'training_time' in self.tags:
            return self.events.Scalars('training_time')[0].value # In seconds
        else:
            return None
        
    def get_training_loss(self, smoothing=0.):
        """
        Returns the (smoothed) training loss (every x-th iteration).
        
        args:
            smoothing: float, weight of the smoothing (0. for no smoothing)

        returns:
            list, iterations
            list, losses

        """
        if 'training_loss' in self.tags:
            return [e.step for e in self.events.Scalars('training_loss')], self._smooth([e.value for e in self.events.Scalars('training_loss')], smoothing)
        else:
            return None, None
        
    def get_accuracy(self):
        """
        Returns the evaluation accuracy.

        returns:
            float, accuracy
        """
        if 'test_accuracy' in self.tags:
            return self.events.Scalars('test_accuracy')[0].value
        else:
            return None
        
    def get_roc(self):
        """
        Returns the ROC curve.

        returns:
            list, false positive rates
            list, true positive rates
            list, thresholds
        """
        if 'test_tpr' in self.tags and 'test_fpr' in self.tags:
            tpr = [e.value for e in self.events.Scalars('test_tpr')]
            fpr = [e.value for e in self.events.Scalars('test_fpr')]
            thr = [e.value for e in self.events.Scalars('test_threshold_roc')]
            return fpr, tpr, thr
        else:
            return None, None, None
        
    def get_auc(self):
        """
        Returns the AUC.

        returns:
            float, AUC
        """
        if 'test_auc' in self.tags:
            return self.events.Scalars('test_auc')[0].value
        else:
            return None
        
    def get_pr_curve(self):
        """
        Returns the precision-recall curve.

        returns:
            list, precisions
            list, recalls
            list, thresholds
        """
        if 'test_precision' in self.tags and 'test_recall' in self.tags:
            precision = [e.value for e in self.events.Scalars('test_precision')]
            recall = [e.value for e in self.events.Scalars('test_recall')]
            thr = [e.value for e in self.events.Scalars('test_threshold_pr')]
            return precision, recall, thr
        else:
            return None, None, None
        
    def get_ap(self):
        """
        Returns the AP.

        returns:
            float, AP
        """
        if 'test_ap' in self.tags:
            return self.events.Scalars('test_ap')[0].value
        else:
            return None

    def get_confusion_matrix(self):
        """
        Returns the confusion matrix as array.

        returns:
            np.array, confusion matrix
        """
        if all([f'confusion_matrix_{i}{j}' in self.tags for i in range(2) for j in range(2)]):
            cm = np.zeros((2, 2), dtype=int)
            for i in range(2):
                for j in range(2):
                    cm[i, j] = self.events.Scalars(f'confusion_matrix_{i}{j}')[0].value
            return cm
        else:
            return None
        
    def info(self):
        """
        Prints information about the event file.
        """
        if self.id:
            print(f'TensorboardReader ID: {self.id}')
        else:
            print('TensorboardReader ID n.a.')
        print(f'Event file: {self.filename} (on path: {"/".join(self.path_to_events_file.split("/")[:-1])})')
        print(f'Available tags:\n{self.tags}')