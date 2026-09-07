import torch
import torch.nn.functional as F
import numpy as np
import cv2

class SegmentationGradCAM:
    """
    Segmentation-aware Grad-CAM implementation.
    Uses forward and backward hooks to capture feature maps and gradients from a target layer.
    """
    def __init__(self, model, target_layer_name='decoder.blocks.4.conv2.0'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.target_layer = self._get_target_layer()

    def _get_target_layer(self):
        """Finds the target layer by traversing the model attributes."""
        layer = self.model
        for part in self.target_layer_name.split('.'):
            if hasattr(layer, part):
                layer = getattr(layer, part)
            elif part.isdigit() and isinstance(layer, (torch.nn.Sequential, torch.nn.ModuleList)):
                layer = layer[int(part)]
            else:
                raise ValueError(f"Could not find layer {self.target_layer_name} in model.")
        return layer

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def _register_hooks(self):
        """Registers the hooks safely."""
        self.hooks.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(self._backward_hook))

    def _remove_hooks(self):
        """Removes the hooks safely."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate(self, input_tensor, target_class):
        """
        Generates Grad-CAM for a given target class.
        If target_class is a list, aggregates the target scores for those classes.
        """
        self._register_hooks()
        
        try:
            # We must enable gradients for the input and model forward pass temporarily
            with torch.enable_grad():
                self.model.eval() # Ensure dropout/batchnorm are in eval mode
                
                # Forward pass
                logits = self.model(input_tensor)
                
                # Create a segmentation-aware target.
                # Target: Sum of raw logits for the selected class(es) across all spatial locations
                # where the model actually predicted that class (or just everywhere).
                # To be purely faithful to the model's spatial reasoning for the class, 
                # we sum the logits of the target classes.
                
                if isinstance(target_class, list):
                    # Aggregate logits for multiple classes (e.g. all foreground layers)
                    target_score = logits[0, target_class, :, :].sum()
                else:
                    target_score = logits[0, target_class, :, :].sum()

                # Backward pass
                self.model.zero_grad()
                target_score.backward()
                
            # Process Grad-CAM
            if self.gradients is None or self.activations is None:
                raise RuntimeError("Grad-CAM computation failed: Gradients or activations are None.")
                
            # Global average pooling of gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            
            # Weighted sum of activations
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            
            # ReLU on CAM (we only care about features that have a positive influence on the class)
            cam = F.relu(cam)
            
            # Convert to numpy
            cam = cam.squeeze().cpu().numpy()
            
            # Ensure valid heatmap
            if np.isnan(cam).any():
                raise ValueError("Grad-CAM contains NaN values.")
                
            if cam.max() == 0:
                # Completely zero heatmap - might mean the class wasn't detected at all, or gradients vanished
                heatmap = np.zeros_like(cam)
            else:
                # Normalize to [0, 1]
                heatmap = cam / cam.max()
                
            return heatmap
            
        finally:
            self._remove_hooks()
            self.gradients = None
            self.activations = None
            self.model.zero_grad() # Clean up

def calculate_confidence(logits):
    """
    Calculates the uncalibrated max softmax probability.
    Returns the confidence map and summary statistics.
    """
    probs = F.softmax(logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    max_probs_np = max_probs.squeeze().cpu().numpy()
    
    stats = {
        'mean': float(np.mean(max_probs_np)),
        'median': float(np.median(max_probs_np)),
        'min': float(np.min(max_probs_np)),
        'low_prob_pct': float(np.mean(max_probs_np < 0.6) * 100) # project heuristic threshold 0.6
    }
    
    return max_probs_np, stats
