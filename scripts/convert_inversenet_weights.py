#!/usr/bin/env python3
"""
Convert InverseRenderNet TensorFlow checkpoint to PyTorch.

Usage:
    python scripts/convert_inversenet_weights.py \
        --tf_ckpt InverseRenderNet_v2/model_ckpts/iiw_model_ckpt/model.ckpt \
        --output data/inversenet_weights.pth
"""

import argparse
import torch
import numpy as np
from pathlib import Path


def convert_tf_to_pytorch(tf_ckpt: str, output_path: str):
    """Convert TF checkpoint to PyTorch state dict."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required for weight conversion. "
                          "Install with: pip install tensorflow")
    
    from reni.baselines.inversenet import InverseRenderNet
    
    # Create model with same config as trained model
    model = InverseRenderNet(n_layers=30, n_pools=4, depth_base=32)
    
    # Load TF checkpoint (TF2 uses compat.v1 for legacy checkpoint reading)
    reader = tf.compat.v1.train.NewCheckpointReader(tf_ckpt)
    var_to_shape = reader.get_variable_to_shape_map()
    
    print(f"Found {len(var_to_shape)} variables in TF checkpoint")
    
    # Get all TF tensors (exclude optimizer state)
    tf_vars = {}
    for name in sorted(var_to_shape.keys()):
        if 'Adam' not in name and 'global_step' not in name:
            tf_vars[name] = reader.get_tensor(name)
            print(f"  {name}: {var_to_shape[name]}")
    
    state_dict = model.state_dict()
    matched = 0
    
    # Map encoder layers (1-indexed in TF, 0-indexed in PyTorch)
    num_encoder_layers = len(model.encoder_layers)
    for i in range(num_encoder_layers):
        tf_scope = f'inverserendernet/conv{i+1}'
        pt_prefix = f'encoder_layers.{i}'
        
        # Conv weights (TF: HWIO -> PyTorch: OIHW)
        tf_weight_key = f'{tf_scope}/weights'
        pt_weight_key = f'{pt_prefix}.conv.weight'
        
        if tf_weight_key in tf_vars and pt_weight_key in state_dict:
            w = tf_vars[tf_weight_key]
            state_dict[pt_weight_key] = torch.from_numpy(w.transpose(3, 2, 0, 1).copy())
            matched += 1
            
        # GroupNorm
        tf_scale_key = f'{tf_scope}/group_norm/scale'
        tf_bias_key = f'{tf_scope}/group_norm/bias'
        pt_scale_key = f'{pt_prefix}.gn.weight'
        pt_bias_key = f'{pt_prefix}.gn.bias'
        
        if tf_scale_key in tf_vars and pt_scale_key in state_dict:
            state_dict[pt_scale_key] = torch.from_numpy(tf_vars[tf_scale_key].copy())
            matched += 1
        if tf_bias_key in tf_vars and pt_bias_key in state_dict:
            state_dict[pt_bias_key] = torch.from_numpy(tf_vars[tf_bias_key].copy())
            matched += 1
    
    # Map decoders
    decoder_map = [
        ('albedo_decoder', 'am_deconv'),
        ('normal_decoder', 'nm'),
        ('shadow_decoder', 'mask_deconv'),
    ]
    
    for pt_decoder_name, tf_prefix in decoder_map:
        decoder = getattr(model, pt_decoder_name)
        num_layers = len(decoder)
        
        for i in range(num_layers):
            tf_scope = f'inverserendernet/{tf_prefix}{i+1}'
            pt_layer = decoder[i]
            
            # Check if this layer is a ConvGNReLU or ConvBlock
            if hasattr(pt_layer, 'conv'):
                conv_module = pt_layer.conv
            else:
                conv_module = pt_layer
                
            # Conv weights
            tf_weight_key = f'{tf_scope}/weights'
            
            # Determine the correct state_dict key
            if hasattr(pt_layer, 'gn'):
                pt_weight_key = f'{pt_decoder_name}.{i}.conv.weight'
            else:
                pt_weight_key = f'{pt_decoder_name}.{i}.conv.weight'
                
            if tf_weight_key in tf_vars and pt_weight_key in state_dict:
                w = tf_vars[tf_weight_key]
                state_dict[pt_weight_key] = torch.from_numpy(w.transpose(3, 2, 0, 1).copy())
                matched += 1
                
            # Bias (only present in final layers)
            tf_bias_key = f'{tf_scope}/biases'
            pt_bias_key = f'{pt_decoder_name}.{i}.conv.bias'
            if tf_bias_key in tf_vars and pt_bias_key in state_dict:
                state_dict[pt_bias_key] = torch.from_numpy(tf_vars[tf_bias_key].copy())
                matched += 1
                
            # GroupNorm (for non-final layers)
            tf_scale_key = f'{tf_scope}/group_norm/scale'
            tf_bias_key = f'{tf_scope}/group_norm/bias'
            pt_scale_key = f'{pt_decoder_name}.{i}.gn.weight'
            pt_bias_key = f'{pt_decoder_name}.{i}.gn.bias'
            
            if tf_scale_key in tf_vars and pt_scale_key in state_dict:
                state_dict[pt_scale_key] = torch.from_numpy(tf_vars[tf_scale_key].copy())
                matched += 1
            if tf_bias_key in tf_vars and pt_bias_key in state_dict:
                state_dict[pt_bias_key] = torch.from_numpy(tf_vars[tf_bias_key].copy())
                matched += 1
    
    print(f"\nMatched {matched} parameters")
    print(f"PyTorch model has {len(state_dict)} parameters")
    
    # Validate shapes before loading
    mismatches = []
    for key in state_dict.keys():
        if key in state_dict and isinstance(state_dict[key], torch.Tensor):
            expected_shape = model.state_dict()[key].shape
            actual_shape = state_dict[key].shape
            if expected_shape != actual_shape:
                mismatches.append(f"  {key}: expected {expected_shape}, got {actual_shape}")
    
    if mismatches:
        print("\nShape mismatches found:")
        for m in mismatches:
            print(m)
        raise RuntimeError("Shape mismatches detected! Check architecture match.")
    
    # Load and save
    model.load_state_dict(state_dict)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)
    print(f"\nSaved PyTorch weights to: {output_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Convert InverseRenderNet TF weights to PyTorch")
    parser.add_argument("--tf_ckpt", type=str, required=True,
                        help="Path to TF checkpoint (e.g., model_ckpts/model_ckpt)")
    parser.add_argument("--output", type=str, default="data/inversenet_weights.pth",
                        help="Output path for PyTorch weights")
    args = parser.parse_args()
    
    convert_tf_to_pytorch(args.tf_ckpt, args.output)


if __name__ == "__main__":
    main()
