from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path
from autotrandhd.core.recognition.crnn import CRNN
import logging

logger = logging.getLogger(__name__)

def load_model(checkpoint_path: str | Path, device: str = "cpu") -> nn.Module:
    """Load a CRNN model from a checkpoint.
    
    Parameters
    ----------
    checkpoint_path:
        Path to the .pt checkpoint.
    device:
        Device to load the model on ('cpu', 'cuda', 'mps').
        
    Returns
    -------
    nn.Module:
        The loaded CRNN model in eval mode.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint
    ckpt = torch.load(path, map_location=device)
    
    # Extract metadata if available
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    # Determine num_classes from state_dict if possible, or assume from ckpt metadata
    # For now, we assume standard CRNN parameters or stored metadata
    # In a real production system, hyperparameters should be saved with the model.
    # Assuming standard defaults for this project or extracted from state_dict shape
    num_classes = state_dict["classifier.bias"].shape[0]
    
    model = CRNN(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Successfully loaded model from {path} onto {device}")
    return model
