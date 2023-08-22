import matplotlib.pyplot as plt

import torch as th
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import Occlusion, visualization as viz

from utils.Models import SingleModalityClassifier, MultiModalityClassifier
        
class VisualizingTools:
    """
    Serves as base class for the visualization methods.
    Contains helper functions for the visualization methods.
    """
    def __init__(self, model: nn.Module):
        """
        Initializaiton.

        args:
            model: nn.Module, model to be visualized
        """
        self.modalities = ['fundus', 'horizontal', 'vertical']
        self.model = model
        self.model.eval()
        if isinstance(model, SingleModalityClassifier):
            self.model_type = 'single'
        elif isinstance(model, MultiModalityClassifier):
            self.model_type = 'multi'
        else:
            raise NotImplementedError(f'{type(model)} not implemented.')

    @staticmethod    
    def _range01(img: th.Tensor):
        """
        Normalizes the image/tensor to the range [0, 1].

        args:
            img: th.Tensor, image/tensor to be normalized

        returns:
            th.Tensor, normalized image/tensor
        """
        return (img - img.min()) / (img.max() - img.min())
    
    @staticmethod
    def _interpolate(img: th.Tensor, map: th.Tensor):
        """
        Interpolates the map to the size of the image.

        args:
            img: th.Tensor, image - shape (C, H, W)
            map: th.Tensor, map - shape (N, C, H, W)

        returns:
            th.Tensor, interpolated map - shape (C, H, W)
        """
        return F.interpolate(map, size=img.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
    
    @staticmethod
    def visualize():
        """
        Abstract method for the visualization of the maps.
        """
        raise NotImplementedError

class CustomHiResCAM(VisualizingTools):
    """
    HiResCAM visualization method.
    """
    def __init__(self, model: nn.Module, target_layers):
        """
        Initialization.

        args:
            model: nn.Module, model to be visualized
            target_layers: str or list, target layers of the model (usually the last convolutional layers)
        """
        super().__init__(model)
        self.target_layers_in = target_layers

        # SingleModalityClassifier
        if self.model_type == 'single':
            self.target_layers = None
            for name, module in self.model.named_modules():
                if name == target_layers: # str for SingleModalityClassifier
                    self.target_layers = module
            if not self.target_layers:
                print('Target layers not found in model!')
            self.hirescam = HiResCAM(self.model, self.target_layers)

        # MultiModalityClassifier
        elif self.model_type == 'multi':
            self.target_layers = []
            for name, module in self.model.named_modules():
                for t_layer in target_layers:
                    if name == t_layer:
                        self.target_layers.append(module)
            if not self.target_layers:
                print('Target layers not found in model!')
            self.hirescam = {}
            for i, k in enumerate(self.modalities):
                self.hirescam[k] = HiResCAM(self.model, self.target_layers[i])

    def _get_blend(self, img: th.Tensor, map: th.Tensor, map_alpha: float):
        """
        Blend tensor with image.
        Called by _single_cam and _multi_cam.

        args:
            img: th.Tensor, image
            map: th.Tensor, map
            map_alpha: float, alpha value for the map

        returns:
            th.Tensor, blended image
        """
        color_map = th.tensor(plt.cm.jet(map.detach().numpy())[..., :3]).permute(2, 0, 1)
        blend = th.clamp((1-map_alpha) * VisualizingTools._range01(img) + map_alpha * color_map, 0.0, 1.0)
        return blend
    
    def _single_cam(self, img: th.Tensor, true_label: int, permute: bool, alpha: float):
        """
        HiResCAM backend for SingleModalityClassifier.
        Called by get_cam.

        args:
            img: th.Tensor, image
            true_label: int, true label of the image
            permute: bool, if True, permute the dimensions of the blended image
            alpha: float, alpha value for the map

        returns:
            th.Tensor, blended image
        """
        target_class = ClassifierOutputTarget(true_label)
        cam = th.tensor(self.hirescam(input_tensor=img[None], targets=[target_class]).squeeze())
        blend = self._get_blend(img, cam, map_alpha=alpha)
        if permute:
            return blend.permute(1, 2, 0)
        else:
            return blend
        
    def _multi_cam(self, img_dict: dict, true_label: int, permute: bool, alpha: float):
        """
        HiResCAM backend for MultiModalityClassifier.
        Called by get_cam.

        args:
            img_dict: dict, image dictionary
            true_label: int, true label of the image
            permute: bool, if True, permute the dimensions of the blended image
            alpha: float, alpha value for the map

        returns:
            dict, blended image dictionary
        """
        imgs = th.cat(list(img_dict.values()), dim=0)
        cam = {} 
        target_class = ClassifierOutputTarget(true_label)
        for k in self.modalities:
            cam[k] = th.tensor(self.hirescam[k](input_tensor=imgs[None], targets=[target_class]).squeeze())
        # Blend CAM with image
        blend = {k: self._get_blend(v, cam[k], map_alpha=alpha) for k, v in img_dict.items()}
        if permute:
            return {k: v.permute(1, 2, 0) for k, v in blend.items()}
        else:
            return blend
        
    def get_cam(self, img, true_label: int, permute: bool=True, alpha: float=0.25):
        """
        Computes the HiResCAM (blended with image) for the given image and label.

        args:
            img: th.Tensor or dict, image
            true_label: int, true label of the image
            permute: bool, if True, permute the dimensions of the blended image
            alpha: float, alpha value for the map

        returns:
            th.Tensor or dict, blended image or blended image dictionary
        """
        if self.model_type == 'single':
            return self._single_cam(img, true_label, permute, alpha)
        elif self.model_type == 'multi':
            return self._multi_cam(img, true_label, permute, alpha)
        
    @staticmethod
    def visualize(blend: th.Tensor, fig_ax: tuple=None, figsize: tuple=plt.rcParams['figure.figsize']):
        """
        Plots the blended image to given axis in given figure, if provided. Else, creates a new figure and axis.

        args:
            blend: th.Tensor, blended image
            fig_ax: tuple, figure and axis
            figsize: tuple, figure size - only used if no figure and axis are provided (fig_ax == None)

        returns:
            tuple, figure and axis
        """
        if not fig_ax:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
        else:
            fig, ax = fig_ax
        ax.imshow(blend, vmin=0, vmax=1)
        ax.axis('off')
        return fig, ax
        
        
class CustomOcclusionMap(VisualizingTools):
    """
    Occlusion map visualization method.
    """
    def __init__(self, model: nn.Module):
        """
        Initialization.

        args:
            model: nn.Module, model to be visualized
        """
        super().__init__(model)
        # FIXME: A bug in captum makes it impossible to use the occlusion map when the model is on MPS
        self.ablator = Occlusion(model)

    def _smooth_tensor(self, tensor, kernel_size=41, sigma=10.0):
        """
        Smooths the tensor with a gaussian kernel. Used for smoothing the occlusion map.

        args:
            tensor: th.Tensor, tensor to be smoothed - shape (..., C, H, W)
            kernel_size: int, size of the gaussian kernel
            sigma: float, sigma of the gaussian kernel

        returns:
            th.Tensor, smoothed tensor - shape (..., C, H, W)
        """
        return transforms.functional.gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)

    def _single_occlusion_map(self, img: th.Tensor, true_label: int, sliding_window_shape: tuple, strides: tuple, permute: bool, show_progress: bool, smooth: bool):
        """
        Occlusion map backend for SingleModalityClassifier.
        Called by get_occlusion_map.

        args:
            img: th.Tensor, image - shape (C, H, W)
            true_label: int, true label of the image
            sliding_window_shape: tuple, shape of the sliding window
            strides: tuple, strides of the sliding window
            permute: bool, if True, permute the dimensions of the occlusion map
            show_progress: bool, if True, show progress bar
            smooth: bool, if True, smooth the occlusion map

        returns:
            th.Tensor, occlusion map
        """
        img = img.unsqueeze(0) # Add batch dimension -> (1, C, H, W)
        attr = self.ablator.attribute(img, sliding_window_shapes=sliding_window_shape, strides=strides, target=true_label, show_progress=show_progress, baselines=0) # (1, 3, H, W )
        if smooth:
            attr = self._smooth_tensor(attr)
        if permute:
            return attr.squeeze().permute(1, 2, 0) # (H, W, 3)
        else:
            return attr.squeeze() # (3, H, W)

    def _multi_occlusion_map(self, img_dict: dict, true_label: int, sliding_window_shape: tuple, strides: tuple, permute: bool, show_progress: bool, smooth: bool):
        """
        Occlusion map backend for MultiModalityClassifier.
        Called by get_occlusion_map.

        args:
            img_dict: dict, image dictionary - shape (C, H, W), each
            true_label: int, true label of the image
            sliding_window_shape: tuple, shape of the sliding window
            strides: tuple, strides of the sliding window
            permute: bool, if True, permute the dimensions of the occlusion map
            show_progress: bool, if True, show progress bar
            smooth: bool, if True, smooth the occlusion map

        returns:
            dict, occlusion map dictionary
        """
        imgs = th.cat(list(img_dict.values()), dim=0).unsqueeze(0) # Add batch dimension -> (1, 9, H, W)
        attr = self.ablator.attribute(imgs, sliding_window_shapes=sliding_window_shape, strides=strides, target=true_label, show_progress=show_progress, baselines=0) # (1, 9, H, W)
        if smooth:
            attr = {m: self._smooth_tensor(attr[0, 3*i:3*(i+1), :, :].unsqueeze(0)) for i, m in enumerate(img_dict.keys())}
        else:
            attr = {m: attr[0, 3*i:3*(i+1), :, :].unsqueeze(0) for i, m in enumerate(img_dict.keys())}
        if permute:
            return {m: a.squeeze().permute(1, 2, 0) for m, a in attr.items()}
        else:
            return attr
        
    def get_occlusion_map(self, img, true_label: int, sliding_window_shape: tuple=(3, 32, 32), strides: tuple=(1, 16, 16), permute: bool=True, show_progress: bool=True, smooth: bool=True):
        """
        Computes the occlusion map (not blended) for the given image and label.

        args:
            img: th.Tensor or dict, image
            true_label: int, true label of the image
            sliding_window_shape: tuple, shape of the sliding window
            strides: tuple, strides of the sliding window
            permute: bool, if True, permute the dimensions of the occlusion map
            show_progress: bool, if True, show progress bar
            smooth: bool, if True, smooth the occlusion map

        returns:
            th.Tensor or dict, occlusion map or occlusion map dictionary
        """
        if self.model_type == 'single':
            return self._single_occlusion_map(img, true_label, sliding_window_shape, strides, permute, show_progress, smooth)
        elif self.model_type == 'multi':
            return self._multi_occlusion_map(img, true_label, sliding_window_shape, strides, permute, show_progress, smooth)
    
    @staticmethod
    def visualize(attr: th.Tensor, img: th.Tensor, fig_ax: tuple=None, figsize: tuple=plt.rcParams['figure.figsize'], alpha: float=0.25):
        """
        Plots the occlusion map (blended with image) to given axis in given figure, if provided. Else, creates a new figure and axis.

        args:
            attr: th.Tensor, occlusion map - shape (H, W, 3)
            img: th.Tensor, image - shape (C, H, W) or (H, W, C)
            fig_ax: tuple, figure and axis
            figsize: tuple, figure size - only used if no figure and axis are provided (fig_ax == None)
            alpha: float, alpha value for the map

        returns:
            tuple, figure and axis
        """
        if not fig_ax:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
        else:
            fig, ax = fig_ax
        if img.shape[-1] != 3:
            img = img.permute(1, 2, 0)
        fig, ax = viz.visualize_image_attr(attr.numpy(),
                                           img.numpy(),
                                           method='blended_heat_map',
                                           sign='absolute_value',
                                           outlier_perc=0,
                                           cmap='jet',
                                           alpha_overlay=alpha,
                                           show_colorbar=False,
                                           plt_fig_axis=(fig, ax),
                                           use_pyplot=False # Does not work with subplots if set to True
        )                                                                    
        return fig, ax


        

        
