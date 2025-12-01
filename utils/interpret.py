import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for saving
from pathlib import Path
import random
from typing import Optional, List, Union


def integrated_gradients(model, inputs, target_idx, baseline=None, n_steps=50):
    """
    compute integrated gradients for a given input
    basically interpolates from baseline to input and accumulates gradients
    
    args:
        model: the fredf model
        inputs: input tensor [B, C, L]
        target_idx: which horizon step to analyze (0 to pred_len-1)
        baseline: baseline input (uses zeros if None)
        n_steps: number of interpolation steps
    
    returns:
        attributions: importance scores [B, C, L] same shape as input
    """
    # store original training state
    was_training = model.training
    model.eval()
    
    # temporarily disable normalization if using itransformer to avoid inplace ops
    disable_norm = False
    if hasattr(model, 'backbone_type') and model.backbone_type == 'itransformer':
        if hasattr(model.backbone, 'model') and hasattr(model.backbone.model, 'use_norm'):
            disable_norm = True
            original_norm = model.backbone.model.use_norm
            model.backbone.model.use_norm = False
    
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    
    # detach inputs to avoid grad issues
    inputs_detached = inputs.detach()
    baseline_detached = baseline.detach()
    
    # generate interpolated inputs between baseline and actual input
    alphas = torch.linspace(0, 1, n_steps + 1, device=inputs.device)
    
    # accumulate gradients across interpolation path
    integrated_grads = torch.zeros_like(inputs)
    
    for alpha in alphas[:-1]:
        # interpolate between baseline and input
        interpolated = baseline_detached + alpha * (inputs_detached - baseline_detached)
        interpolated = interpolated.clone().requires_grad_(True)
        
        # forward pass
        outputs = model(interpolated)
        
        # compute gradient wrt target horizon step
        # we sum across batch and features to get scalar
        target_output = outputs[:, :, target_idx].sum()
        
        # backprop - use retain_graph=False to clean up each time
        target_output.backward()
        
        # accumulate gradients
        if interpolated.grad is not None:
            integrated_grads += interpolated.grad.detach().clone()
        
        # explicitly delete to free memory
        del interpolated, outputs, target_output
    
    # average and scale by input - baseline
    integrated_grads = integrated_grads / n_steps
    attributions = (inputs_detached - baseline_detached) * integrated_grads
    
    # restore normalization setting
    if disable_norm:
        model.backbone.model.use_norm = original_norm
    
    # restore training state
    if was_training:
        model.train()
    
    return attributions.detach()


def get_attention_maps(model, inputs):
    """
    extract attention weights from itransformer
    only works if model uses itransformer backbone
    
    returns:
        list of attention maps from each layer, or None if not applicable
    """
    model.eval()
    
    # check if using itransformer
    if not hasattr(model, 'backbone_type') or model.backbone_type != 'itransformer':
        return None
    
    # temporarily enable output_attention on model and all attention modules
    backbone = model.backbone.model
    original_flag = backbone.output_attention
    backbone.output_attention = True
    
    # also need to enable it on the FullAttention modules
    original_attn_flags = []
    for layer in backbone.encoder.attn_layers:
        attn_module = layer.attention.inner_attention
        original_attn_flags.append(attn_module.output_attention)
        attn_module.output_attention = True
    
    with torch.no_grad():
        # need to call the backbone's forward directly
        x = inputs.transpose(1, 2)  # convert to itransformer format
        _, attns = backbone.forecast(x, None, None, None)
    
    # restore original flags
    backbone.output_attention = original_flag
    for i, layer in enumerate(backbone.encoder.attn_layers):
        layer.attention.inner_attention.output_attention = original_attn_flags[i]
    
    # check if attns is valid
    if attns is None or (isinstance(attns, list) and len(attns) == 0):
        return None
    
    # check if any attention is None
    if isinstance(attns, list) and any(a is None for a in attns):
        return None
    
    return attns


def patch_masking_importance(model, inputs, target_idx, patch_size=8):
    """
    measure importance by masking patches of the input sequence
    this is kind of like occlusion sensitivity
    
    args:
        model: fredf model
        inputs: [B, C, L]
        target_idx: which horizon step to measure
        patch_size: size of patches to mask
        
    returns:
        importance scores [B, C, num_patches]
    """
    model.eval()
    
    B, C, L = inputs.shape
    num_patches = L // patch_size
    
    # get baseline prediction (no masking)
    with torch.no_grad():
        baseline_pred = model(inputs)
        baseline_value = baseline_pred[:, :, target_idx]  # [B, C]
    
    # iterate through patches and measure impact of masking each
    importance = torch.zeros(B, C, num_patches, device=inputs.device)
    
    for p in range(num_patches):
        start_idx = p * patch_size
        end_idx = min(start_idx + patch_size, L)
        
        # mask this patch (set to zero)
        masked_input = inputs.clone()
        masked_input[:, :, start_idx:end_idx] = 0
        
        with torch.no_grad():
            masked_pred = model(masked_input)
            masked_value = masked_pred[:, :, target_idx]
        
        # difference from baseline = importance
        diff = torch.abs(baseline_value - masked_value)
        importance[:, :, p] = diff
    
    return importance.detach()


def visualize_ig_attribution(attributions, sample_idx, horizon_steps, save_path, 
                             feature_names=None, dataset_name='', input_data=None):
    """
    visualize integrated gradients with highlighted important timesteps
    
    args:
        attributions: [B, C, L] attribution scores
        sample_idx: which sample to visualize
        horizon_steps: list of horizon steps we computed (e.g. [0, 23, 47])
        save_path: where to save figure
        feature_names: optional list of feature names
        dataset_name: dataset name for title
        input_data: optional [B, C, L] input time series to overlay
    """
    attr = attributions[sample_idx].cpu().numpy()  # [C, L]
    C, L = attr.shape
    
    if feature_names is None:
        feature_names = [f'feat{i}' for i in range(C)]
    
    if input_data is None:
        print("  warning: input_data is None, skipping visualization")
        return
    
    input_ts = input_data[sample_idx].cpu().numpy()  # [C, L]
    
    # create figure with one row per horizon step, each containing C feature subplots
    n_horizons = len(horizon_steps)
    fig = plt.figure(figsize=(14, 2.5 * n_horizons * C))
    
    for h_idx, h_step in enumerate(horizon_steps):
        # use absolute attribution values
        attr_abs = np.abs(attr)
        
        for c in range(C):
            ax = plt.subplot(n_horizons * C, 1, h_idx * C + c + 1)
            
            # normalize importance to 0-1 for this feature
            importance = attr_abs[c]
            if importance.max() > 0:
                importance_norm = importance / importance.max()
            else:
                importance_norm = importance
            
            # plot the time series
            timesteps = np.arange(L)
            ax.plot(timesteps, input_ts[c], color='steelblue', linewidth=1.5, alpha=0.8)
            
            # highlight important regions with colored background
            # use a colormap to show importance intensity
            for t in range(L):
                if importance_norm[t] > 0.1:  # only highlight if somewhat important
                    alpha_val = min(0.6, importance_norm[t] * 0.7)
                    ax.axvspan(t - 0.5, t + 0.5, alpha=alpha_val, color='orange', zorder=0)
            
            ax.set_xlim(-0.5, L - 0.5)
            ax.set_ylabel(feature_names[c], fontsize=9)
            ax.grid(alpha=0.3, linewidth=0.5)
            
            # only show x-label on bottom plot of each horizon group
            if c == C - 1:
                ax.set_xlabel('lookback timestep', fontsize=9)
            else:
                ax.set_xticklabels([])
            
            # title only on first feature of each horizon
            if c == 0:
                ax.set_title(f'horizon step {h_step} - integrated gradients importance', 
                           fontsize=10, fontweight='bold', pad=10)
    
    fig.suptitle(f'{dataset_name} - integrated gradients (sample {sample_idx})', 
                fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved IG visualization to {save_path}")


def visualize_attention_maps(attns, sample_idx, layer_idx, save_path, 
                             feature_names=None, dataset_name=''):
    """
    visualize attention weights from a specific layer
    
    args:
        attns: list of attention tensors [B, num_heads, N, N]
        sample_idx: which sample
        layer_idx: which encoder layer (0 to e_layers-1)
        save_path: output path
        feature_names: optional feature names
        dataset_name: for title
    """
    if attns is None or len(attns) == 0:
        print("  no attention maps available (not using itransformer?)")
        return
    
    attn = attns[layer_idx][sample_idx].cpu().numpy()  # [num_heads, N, N]
    num_heads, N, _ = attn.shape
    
    if feature_names is None:
        feature_names = [f'V{i}' for i in range(N)]
    
    # show 4 attention heads (or fewer if less than 4)
    n_show = min(4, num_heads)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for h in range(n_show):
        ax = axes[h]
        
        im = ax.imshow(attn[h], cmap='viridis', aspect='auto', interpolation='nearest')
        if N <= 20:
            ax.set_xticks(range(N))
            ax.set_yticks(range(N))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_yticklabels(feature_names)
        ax.set_title(f'head {h} (layer {layer_idx})')
        ax.set_xlabel('key (variate)')
        ax.set_ylabel('query (variate)')
        plt.colorbar(im, ax=ax, label='attention weight')
    
    # hide unused subplots
    for h in range(n_show, 4):
        axes[h].axis('off')
    
    fig.suptitle(f'attention maps - {dataset_name} - layer {layer_idx}', 
                fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved attention visualization to {save_path}")


def visualize_patch_importance(importance, sample_idx, horizon_steps, save_path,
                               feature_names=None, dataset_name='', patch_size=8, input_data=None):
    """
    visualize patch masking with highlighted important patches
    
    args:
        importance: [B, C, num_patches] 
        sample_idx: which sample
        horizon_steps: which horizon steps we measured
        save_path: output file
        feature_names: optional
        dataset_name: for title
        patch_size: patch size used
        input_data: optional [B, C, L] input time series to overlay
    """
    imp = importance[sample_idx].cpu().numpy()  # [C, num_patches]
    C, num_patches = imp.shape
    
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(C)]
    
    if input_data is None:
        print("  warning: input_data is None, skipping visualization")
        return
    
    input_ts = input_data[sample_idx].cpu().numpy()  # [C, L]
    L = input_ts.shape[1]
    
    # create figure with one row per horizon step, each containing C feature subplots
    n_horizons = len(horizon_steps)
    fig = plt.figure(figsize=(14, 2.5 * n_horizons * C))
    
    for h_idx, h_step in enumerate(horizon_steps):
        for c in range(C):
            ax = plt.subplot(n_horizons * C, 1, h_idx * C + c + 1)
            
            # normalize importance for this feature
            patch_imp = imp[c]
            if patch_imp.max() > 0:
                patch_imp_norm = patch_imp / patch_imp.max()
            else:
                patch_imp_norm = patch_imp
            
            # plot the time series
            timesteps = np.arange(L)
            ax.plot(timesteps, input_ts[c], color='steelblue', linewidth=1.5, alpha=0.8)
            
            # highlight important patches with colored background
            for p in range(num_patches):
                patch_start = p * patch_size
                patch_end = min(patch_start + patch_size, L)
                
                if patch_imp_norm[p] > 0.1:  # only highlight if somewhat important
                    alpha_val = min(0.6, patch_imp_norm[p] * 0.7)
                    ax.axvspan(patch_start - 0.5, patch_end - 0.5, 
                             alpha=alpha_val, color='red', zorder=0)
                
                # draw patch boundary lines
                if p < num_patches:
                    ax.axvline(patch_end - 0.5, color='gray', linestyle='--', 
                             alpha=0.2, linewidth=0.8)
            
            ax.set_xlim(-0.5, L - 0.5)
            ax.set_ylabel(feature_names[c], fontsize=9)
            ax.grid(alpha=0.3, linewidth=0.5)
            
            # only show x-label on bottom plot of each horizon group
            if c == C - 1:
                ax.set_xlabel('lookback timestep', fontsize=9)
            else:
                ax.set_xticklabels([])
            
            # title only on first feature of each horizon
            if c == 0:
                ax.set_title(f'horizon step {h_step} - patch masking (patch_size={patch_size})', 
                           fontsize=10, fontweight='bold', pad=10)
    
    fig.suptitle(f'{dataset_name} - patch masking (sample {sample_idx})', 
                fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved patch masking viz to {save_path}")


def run_interpretability(model, test_loader, device, args):
    """
    main function to run interpretability analysis
    
    supports:
     - integrated gradients
     - attention maps (itransformer only)
     - patch masking
     
    can specify test indices or use random samples
    """
    model.eval()
    
    # figure out which samples to analyze
    dataset_size = len(test_loader.dataset)
    if args.interp_indices is not None:
        # user specified indices
        indices = [int(i) for i in args.interp_indices.split(',')]
        indices = [i for i in indices if 0 <= i < dataset_size]
        if len(indices) == 0:
            print("warning: no valid indices provided, using random")
            indices = [random.randint(0, dataset_size - 1) for _ in range(args.interp_samples)]
    else:
        # random sampling
        indices = [random.randint(0, dataset_size - 1) for _ in range(args.interp_samples)]
    
    print(f"\nrunning interpretability on {len(indices)} samples: {indices}")
    
    # collect samples
    samples_x = []
    samples_y = []
    for idx in indices:
        x, y = test_loader.dataset[idx]
        samples_x.append(x)
        samples_y.append(y)
    
    batch_x = torch.stack(samples_x).to(device)  # [num_samples, C, L]
    batch_y = torch.stack(samples_y).to(device)
    
    # which horizon steps to analyze
    pred_len = model.forecast_horizon
    if args.interp_horizons is not None:
        horizon_steps = [int(h) for h in args.interp_horizons.split(',')]
        horizon_steps = [h for h in horizon_steps if 0 <= h < pred_len]
    else:
        # default: first, middle, last
        horizon_steps = [0, pred_len // 2, pred_len - 1]
    
    print(f"analyzing horizon steps: {horizon_steps}")
    
    # setup output directory
    output_dir = Path(args.interp_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # feature names (if we know them)
    feature_names = _get_feature_names(args.dataset)
    
    # run different interpretability methods
    if 'ig' in args.interp_methods:
        print("\nrunning integrated gradients...")
        for h_idx, h_step in enumerate(horizon_steps):
            print(f"  computing for horizon step {h_step}...")
            attributions = integrated_gradients(model, batch_x, h_step, 
                                              n_steps=args.ig_steps)
            
            # visualize each sample
            for sample_i in range(len(indices)):
                save_path = output_dir / f"ig_sample{indices[sample_i]}_h{h_step}.png"
                visualize_ig_attribution(attributions, sample_i, [h_step], 
                                       save_path, feature_names, args.dataset, 
                                       input_data=batch_x)
    
    if 'attention' in args.interp_methods:
        print("\ncomputing attention maps...")
        attns = get_attention_maps(model, batch_x)
        
        if attns is not None and len(attns) > 0:
            num_layers = len(attns)
            # visualize attention from last layer by default
            layer = num_layers - 1
            print(f"  visualizing layer {layer} (last layer)")
            
            for sample_i in range(len(indices)):
                save_path = output_dir / f"attn_sample{indices[sample_i]}_layer{layer}.png"
                visualize_attention_maps(attns, sample_i, layer, save_path,
                                       feature_names, args.dataset)
        else:
            print("  attention visualization only works with itransformer backbone")
    
    if 'masking' in args.interp_methods:
        print("\nrunning patch masking analysis...")
        for h_idx, h_step in enumerate(horizon_steps):
            print(f"  computing for horizon step {h_step}...")
            importance = patch_masking_importance(model, batch_x, h_step,
                                                 patch_size=args.patch_size)
            
            for sample_i in range(len(indices)):
                save_path = output_dir / f"masking_sample{indices[sample_i]}_h{h_step}.png"
                visualize_patch_importance(importance, sample_i, [h_step],
                                          save_path, feature_names, args.dataset,
                                          patch_size=args.patch_size, input_data=batch_x)
    
    print(f"\ninterpretability analysis complete! outputs saved to {output_dir}/")


def _get_feature_names(dataset_name):
    """helper to get feature names for common datasets"""
    # just return None for now, can expand later
    dataset_features = {
        'ETTh1': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'ETTm1': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
    }
    return dataset_features.get(dataset_name, None)
