import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for saving
from pathlib import Path
import random
from typing import Optional, List, Union


def integrated_gradients(model, inputs, target_idx, baseline=None, n_steps=50, target_variate=None):
    """
    compute integrated gradients for a given input
    basically interpolates from baseline to input and accumulates gradients

    Method walks from a "baseline" input (zeros = no signal) to your actual input in small steps. At each step along this path, it:
    -Creates an interpolated input (mix of baseline+real input)
    -Runs the model forward
    -Computes gradients: "if I wiggle this input timestep, how much does my target prediction change"
    -Accumulates these gradients
    -At the end, averages all gradients and scales by (input - baseline)
    
    args:
        model: the fredf model
        inputs: input tensor [B, C, L]
        target_idx: which horizon step to analyze (0 to pred_len-1)
        baseline: baseline input (uses zeros if None)
        n_steps: number of interpolation steps
        target_variate: which output variate to compute importance for (None = all variates)
    
    returns:
        attributions: importance scores [B, C, L] same shape as input
    """
    # store original training state so we can restore it later
    # we need eval mode for stable gradients (no dropout etc)
    was_training = model.training
    model.eval()
    
    # temporarily disable normalization if using itransformer to avoid inplace ops
    # itransformer does some normalization that modifies tensors in-place which
    # breaks gradient computation (pytorch doesnt like inplace ops during backprop)
    # so we turn it off temporarily, compute our gradients, then turn it back on
    disable_norm = False
    if hasattr(model, 'backbone_type') and model.backbone_type == 'itransformer':
        if hasattr(model.backbone, 'model') and hasattr(model.backbone.model, 'use_norm'):
            disable_norm = True
            original_norm = model.backbone.model.use_norm
            model.backbone.model.use_norm = False
    
    if baseline is None:
        # if no baseline provided, use zeros (ie "no signal" baseline)
        # this is standard practice for time series - we're measuring importance
        # relative to having no input at all
        baseline = torch.zeros_like(inputs)
    
    # detach inputs to avoid grad issues - we dont want gradients flowing
    # back into the original input tensor, just want to measure importance
    inputs_detached = inputs.detach()
    baseline_detached = baseline.detach()
    
    # generate interpolated inputs between baseline and actual input
    # this is the "path" we'll integrate along - we go from baseline (alpha=0)
    # to actual input (alpha=1) in n_steps increments
    # the +1 is because linspace includes both endpoints
    alphas = torch.linspace(0, 1, n_steps + 1, device=inputs.device)
    
    # accumulate gradients across interpolation path
    # this will hold the sum of all gradients along the path
    integrated_grads = torch.zeros_like(inputs)
    
    # iterate through alpha values (skip the last one since we use n_steps not n_steps+1)
    for alpha in alphas[:-1]:
        # interpolate between baseline and input
        # when alpha=0 we get baseline, when alpha=1 we get actual input
        # this creates a straight line path in input space
        interpolated = baseline_detached + alpha * (inputs_detached - baseline_detached)
        # clone and enable gradients for this specific interpolated input
        interpolated = interpolated.clone().requires_grad_(True)
        
        # forward pass through the model with this interpolated input
        outputs = model(interpolated)
        
        # compute gradient wrt target horizon step
        # we're asking: "how does this specific future timestep change
        # when we wiggle the input?" - that tells us which input parts matter
        if target_variate is not None:
            # specific variate - only measure importance for predicting this one output
            target_output = outputs[:, target_variate, target_idx].sum()
        else:
            # sum across batch and features to get a single scalar we can backprop from
            # this measures importance for predicting ALL output variates
            target_output = outputs[:, :, target_idx].sum()
        
        # backprop to get gradients - use retain_graph=False to clean up each time
        # this computes d(output)/d(interpolated) for this alpha value
        target_output.backward()
        
        # accumulate gradients from this step of the path
        # we're summing up all the gradient vectors along the integration path
        if interpolated.grad is not None:
            integrated_grads += interpolated.grad.detach().clone()
        
        # explicitly delete to free memory - important for large models
        # otherwise we might run out of gpu memory with many steps
        del interpolated, outputs, target_output
    
    # average and scale by input - baseline
    # the averaging is the "integration" part - we approximate the integral
    # by taking the average of the gradients along the path
    integrated_grads = integrated_grads / n_steps
    # then scale by the difference between input and baseline to get attributions
    # this gives us the "contribution" of each input element to the output
    # higher magnitude = more important for the prediction
    attributions = (inputs_detached - baseline_detached) * integrated_grads
    
    # restore normalization setting if we disabled it earlier
    if disable_norm:
        model.backbone.model.use_norm = original_norm
    
    # restore training state so we dont mess up future training
    if was_training:
        model.train()
    
    # return attributions (detached so no grads flow back)
    return attributions.detach()


def get_attention_maps(model, inputs):
    """
    extract attention weights from itransformer
    only works if model uses itransformer backbone
    
    returns:
        list of attention maps from each layer, or None if not applicable
    """
    model.eval()
    
    # check if using itransformer - attention only makes sense for transformer models
    # tsmixer doesnt have attention mechanism so we cant extract weights from it
    if not hasattr(model, 'backbone_type') or model.backbone_type != 'itransformer':
        return None
    
    # temporarily enable output_attention on model and all attention modules
    # by default the model doesnt return attention weights (saves memory)
    # so we need to turn on this flag, grab the weights, then turn it back off
    backbone = model.backbone.model
    original_flag = backbone.output_attention
    backbone.output_attention = True
    
    # also need to enable it on the FullAttention modules in each layer
    # itransformer has a nested structure so we gotta dig down and enable it
    # at each attention layer individually. save original flags so we can restore
    original_attn_flags = []
    for layer in backbone.encoder.attn_layers:
        attn_module = layer.attention.inner_attention
        original_attn_flags.append(attn_module.output_attention)
        attn_module.output_attention = True
    
    with torch.no_grad():
        # need to call the backbone's forward directly to get attention outputs
        # fredf model wraps the backbone so we bypass that wrapper here
        x = inputs.transpose(1, 2)  # convert to itransformer format [B,C,L] -> [B,L,C]
        # forecast returns (predictions, attention_weights)
        _, attns = backbone.forecast(x, None, None, None)
    
    # restore original flags - important to not mess up future forward passes
    # that might expect output_attention to be False for speed
    backbone.output_attention = original_flag
    for i, layer in enumerate(backbone.encoder.attn_layers):
        layer.attention.inner_attention.output_attention = original_attn_flags[i]
    
    # check if attns is valid - sometimes the model returns None or empty list
    # if attention wasnt computed properly (eg wrong config or model version)
    if attns is None or (isinstance(attns, list) and len(attns) == 0):
        return None
    
    # check if any attention is None - sometimes individual layers fail
    # to return attention (shouldnt happen but better safe than sorry)
    if isinstance(attns, list) and any(a is None for a in attns):
        return None
    
    # return list of attention tensors, one per encoder layer
    return attns


def patch_masking_importance(model, inputs, target_idx, patch_size=8, target_variate=None):
    """
    measure importance by masking patches of the input sequence
    this is kind of like occlusion sensitivity
    
    args:
        model: fredf model
        inputs: [B, C, L]
        target_idx: which horizon step to measure
        patch_size: size of patches to mask
        target_variate: which output variate to measure importance for (None = all)
        
    returns:
        importance scores [B, C, num_patches]
    """
    model.eval()
    
    B, C, L = inputs.shape
    num_patches = L // patch_size  # divide sequence into patches
    
    # get baseline prediction (no masking)
    # this is what the model predicts when it can see the full input
    # we'll compare against this to see how much each patch matters
    with torch.no_grad():
        baseline_pred = model(inputs)
        if target_variate is not None:
            baseline_value = baseline_pred[:, target_variate, target_idx:target_idx+1]  # [B, 1]
        else:
            baseline_value = baseline_pred[:, :, target_idx]  # just the target horizon [B, C]
    
    # iterate through patches and measure impact of masking each
    # idea: if masking a patch changes the prediction a lot, that patch is important
    # if masking it doesnt change much, then it wasnt being used much
    importance = torch.zeros(B, C, num_patches, device=inputs.device)
    
    for p in range(num_patches):
        # figure out which timesteps belong to this patch
        start_idx = p * patch_size
        end_idx = min(start_idx + patch_size, L)  # handle last patch that might be smaller
        
        # mask each variate's patch separately to measure per-variate importance
        for c in range(C):
            # mask this patch for this specific variate only
            masked_input = inputs.clone()
            masked_input[:, c, start_idx:end_idx] = 0
            
            # get prediction with this patch masked out
            with torch.no_grad():
                masked_pred = model(masked_input)
                if target_variate is not None:
                    masked_value = masked_pred[:, target_variate, target_idx:target_idx+1]
                else:
                    masked_value = masked_pred[:, :, target_idx]
            
            # difference from baseline = importance
            # big difference means the model relied on that patch heavily
            # small difference means the patch wasnt very important
            diff = torch.abs(baseline_value - masked_value)
            if target_variate is not None:
                importance[:, c, p] = diff.squeeze()
            else:
                importance[:, c, p] = diff[:, c]
    
    return importance.detach()


def visualize_ig_attribution(attributions, sample_idx, horizon_steps, save_path, 
                             feature_names=None, dataset_name='', input_data=None,
                             predictions=None, target_variate=None):
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
        predictions: optional [B, C, pred_len] model predictions
        target_variate: which variate was targeted (None = all)
    """
    # extract the specific sample we want to visualize
    attr = attributions[sample_idx].cpu().numpy()  # [C, L]
    C, L = attr.shape
    
    # if no feature names provided, just number them
    if feature_names is None:
        feature_names = [f'feat{i}' for i in range(C)]
    
    if input_data is None:
        print("  warning: input_data is None, skipping visualization")
        return
    
    # get the actual input time series so we can plot it with highlights
    input_ts = input_data[sample_idx].cpu().numpy()  # [C, L]
    
    # get predictions if available
    if predictions is not None:
        pred_ts = predictions[sample_idx].cpu().numpy()  # [C, pred_len]
        pred_len = pred_ts.shape[1]
    else:
        pred_ts = None
        pred_len = 0
    
    # create figure with one row per horizon step, each containing C feature subplots
    # this lets us see how importance changes for different prediction horizons
    n_horizons = len(horizon_steps)
    fig = plt.figure(figsize=(14, 2.5 * n_horizons * C))
    
    for h_idx, h_step in enumerate(horizon_steps):
        # use absolute attribution values - we care about magnitude not sign
        # negative just means "inhibits prediction" vs positive "promotes prediction"
        # but both are equally important
        attr_abs = np.abs(attr)
        
        for c in range(C):
            # create subplot for this feature and horizon
            ax = plt.subplot(n_horizons * C, 1, h_idx * C + c + 1)
            
            # normalize importance to 0-1 for this feature so colors are consistent
            # otherwise one feature might dominate the color scale
            importance = attr_abs[c]
            if importance.max() > 0:
                importance_norm = importance / importance.max()
            else:
                # all zeros, just keep it
                importance_norm = importance
            
            # plot the time series as a line
            timesteps = np.arange(L)
            ax.plot(timesteps, input_ts[c], color='steelblue', linewidth=1.5, alpha=0.8, label='lookback')
            
            # plot predictions if available
            if pred_ts is not None:
                pred_timesteps = np.arange(L, L + pred_len)
                ax.plot(pred_timesteps, pred_ts[c], color='green', linewidth=1.5, 
                       linestyle='--', alpha=0.8, label='prediction')
                # draw vertical line separating lookback from forecast
                ax.axvline(L - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
            
            # highlight important regions with colored background
            # the more important a timestep, the more opaque the orange highlight
            # this makes it easy to see which parts of the input the model focuses on
            for t in range(L):
                if importance_norm[t] > 0.1:  # only highlight if somewhat important (filter noise)
                    # scale alpha by importance but cap it so its not too dark
                    alpha_val = min(0.6, importance_norm[t] * 0.7)
                    ax.axvspan(t - 0.5, t + 0.5, alpha=alpha_val, color='orange', zorder=0)
            
            ax.set_xlim(-0.5, (L + pred_len - 0.5) if pred_ts is not None else (L - 0.5))
            ax.set_ylabel(feature_names[c], fontsize=9)
            ax.grid(alpha=0.3, linewidth=0.5)
            if c == 0 and h_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
            
            # only show x-label on bottom plot of each horizon group
            if c == C - 1:
                ax.set_xlabel('timestep', fontsize=9)
            else:
                ax.set_xticklabels([])
            
            # title only on first feature of each horizon
            if c == 0:
                target_str = f' (target: {feature_names[target_variate]})' if target_variate is not None else ' (all variates)'
                ax.set_title(f'horizon step {h_step} - integrated gradients{target_str}', 
                           fontsize=10, fontweight='bold', pad=10)
    
    target_label = f' - target: {feature_names[target_variate]}' if target_variate is not None else ''
    fig.suptitle(f'{dataset_name} - integrated gradients (sample {sample_idx}){target_label}', 
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
    
    # extract attention for this specific sample and layer
    attn = attns[layer_idx][sample_idx].cpu().numpy()  # [num_heads, N, N]
    num_heads, N, _ = attn.shape
    
    # N is number of variates (features) since itransformer is "inverted"
    # it does attention over variates not timesteps
    if feature_names is None:
        feature_names = [f'V{i}' for i in range(N)]
    
    # show 4 attention heads (or fewer if less than 4)
    # most transformers have multiple heads so we can see different attention patterns
    n_show = min(4, num_heads)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for h in range(n_show):
        ax = axes[h]
        
        # plot attention matrix as heatmap
        # rows = queries (which variate is asking), cols = keys (which variate is being attended to)
        # bright spots = strong attention between those variates
        im = ax.imshow(attn[h], cmap='viridis', aspect='auto', interpolation='nearest')
        # only show feature names if not too many (otherwise it gets messy)
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
                               feature_names=None, dataset_name='', patch_size=8, input_data=None,
                               predictions=None, target_variate=None):
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
        predictions: optional [B, C, pred_len] model predictions
        target_variate: which variate was targeted (None = all)
    """
    # extract importance scores for this sample
    imp = importance[sample_idx].cpu().numpy()  # [C, num_patches]
    C, num_patches = imp.shape
    
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(C)]
    
    if input_data is None:
        print("  warning: input_data is None, skipping visualization")
        return
    
    # get input time series to plot
    input_ts = input_data[sample_idx].cpu().numpy()  # [C, L]
    L = input_ts.shape[1]
    
    if predictions is not None:
        pred_ts = predictions[sample_idx].cpu().numpy()  # [C, pred_len]
        pred_len = pred_ts.shape[1]
    else:
        pred_ts = None
        pred_len = 0
    
    # create figure with one row per horizon step, each containing C feature subplots
    n_horizons = len(horizon_steps)
    fig = plt.figure(figsize=(14, 2.5 * n_horizons * C))
    
    for h_idx, h_step in enumerate(horizon_steps):
        for c in range(C):
            ax = plt.subplot(n_horizons * C, 1, h_idx * C + c + 1)
            
            # normalize importance for this feature to 0-1 range
            patch_imp = imp[c]
            if patch_imp.max() > 0:
                patch_imp_norm = patch_imp / patch_imp.max()
            else:
                patch_imp_norm = patch_imp
            
            # plot the time series
            timesteps = np.arange(L)
            ax.plot(timesteps, input_ts[c], color='steelblue', linewidth=1.5, alpha=0.8, label='lookback')
            
            # plot predictions if available
            if pred_ts is not None:
                pred_timesteps = np.arange(L, L + pred_len)
                ax.plot(pred_timesteps, pred_ts[c], color='green', linewidth=1.5, 
                       linestyle='--', alpha=0.8, label='prediction')
                # draw vertical line separating lookback from forecast
                ax.axvline(L - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
            
            # highlight important patches with colored background
            # red means "this chunk of time was important for prediction"
            for p in range(num_patches):
                # figure out where this patch is in the time series
                patch_start = p * patch_size
                patch_end = min(patch_start + patch_size, L)
                
                if patch_imp_norm[p] > 0.1:  # only highlight if somewhat important
                    alpha_val = min(0.6, patch_imp_norm[p] * 0.7)
                    ax.axvspan(patch_start - 0.5, patch_end - 0.5, 
                             alpha=alpha_val, color='red', zorder=0)
                
                # draw patch boundary lines so we can see where patches divide
                if p < num_patches:
                    ax.axvline(patch_end - 0.5, color='gray', linestyle='--', 
                             alpha=0.2, linewidth=0.8)
            
            ax.set_xlim(-0.5, (L + pred_len - 0.5) if pred_ts is not None else (L - 0.5))
            ax.set_ylabel(feature_names[c], fontsize=9)
            ax.grid(alpha=0.3, linewidth=0.5)
            if c == 0 and h_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
            
            # only show x-label on bottom plot of each horizon group
            if c == C - 1:
                ax.set_xlabel('timestep', fontsize=9)
            else:
                ax.set_xticklabels([])
            
            # title only on first feature of each horizon
            if c == 0:
                target_str = f' (target: {feature_names[target_variate]})' if target_variate is not None else ' (all variates)'
                ax.set_title(f'horizon step {h_step} - patch masking (patch_size={patch_size}){target_str}', 
                           fontsize=10, fontweight='bold', pad=10)
    
    target_label = f' - target: {feature_names[target_variate]}' if target_variate is not None else ''
    fig.suptitle(f'{dataset_name} - patch masking (sample {sample_idx}){target_label}', 
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
    
    # figure out which samples to analyze from the test set
    dataset_size = len(test_loader.dataset)
    if args.interp_indices is not None:
        # user specified indices (eg "0,5,10")
        indices = [int(i) for i in args.interp_indices.split(',')]
        # filter out invalid indices that are out of bounds
        indices = [i for i in indices if 0 <= i < dataset_size]
        if len(indices) == 0:
            print("warning: no valid indices provided, using random")
            indices = [random.randint(0, dataset_size - 1) for _ in range(args.interp_samples)]
    else:
        # random sampling - just pick some samples from the test set
        indices = [random.randint(0, dataset_size - 1) for _ in range(args.interp_samples)]
    
    print(f"\nrunning interpretability on {len(indices)} samples: {indices}")
    
    # collect samples from the dataset
    # we pull out the specific indices we want to analyze
    samples_x = []
    samples_y = []
    for idx in indices:
        x, y = test_loader.dataset[idx]
        samples_x.append(x)
        samples_y.append(y)
    
    # stack into batch for efficient processing
    batch_x = torch.stack(samples_x).to(device)  # [num_samples, C, L]
    batch_y = torch.stack(samples_y).to(device)
    
    # compute predictions once for visualization
    with torch.no_grad():
        predictions = model(batch_x)  # [num_samples, C, pred_len]
    
    # get target variate if specified
    target_variate = getattr(args, 'target_variate', None)
    
    # which horizon steps to analyze (ie which future timesteps)
    pred_len = model.forecast_horizon
    if args.interp_horizons is not None:
        # user specified which horizons (eg "0,23,47")
        horizon_steps = [int(h) for h in args.interp_horizons.split(',')]
        # filter out invalid horizons
        horizon_steps = [h for h in horizon_steps if 0 <= h < pred_len]
    else:
        # default: first, middle, last - gives us a good spread
        horizon_steps = [0, pred_len // 2, pred_len - 1]
    
    print(f"analyzing horizon steps: {horizon_steps}")
    
    # setup output directory for saving visualizations
    output_dir = Path(args.interp_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # feature names (if we know them) - makes plots more readable
    feature_names = _get_feature_names(args.dataset)
    
    # run different interpretability methods based on what user requested
    # args.interp_methods is a list like ['ig', 'attention', 'masking']
    if 'ig' in args.interp_methods:
        print("\nrunning integrated gradients...")
        # compute IG for each horizon step separately
        for h_idx, h_step in enumerate(horizon_steps):
            print(f"  computing for horizon step {h_step}...")
            # this computes importance scores for predicting this specific future timestep
            attributions = integrated_gradients(model, batch_x, h_step, 
                                              n_steps=args.ig_steps,
                                              target_variate=target_variate)
            
            # visualize each sample we analyzed
            for sample_i in range(len(indices)):
                save_path = output_dir / f"ig_sample{indices[sample_i]}_h{h_step}.png"
                visualize_ig_attribution(attributions, sample_i, [h_step], 
                                       save_path, feature_names, args.dataset, 
                                       input_data=batch_x, predictions=predictions,
                                       target_variate=target_variate)
    
    if 'attention' in args.interp_methods:
        print("\ncomputing attention maps...")
        attns = get_attention_maps(model, batch_x)
        
        if attns is not None and len(attns) > 0:
            # we got attention weights! visualize them
            num_layers = len(attns)
            # visualize attention from last layer by default (usually most interpretable)
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
        # compute importance by masking patches for each horizon
        for h_idx, h_step in enumerate(horizon_steps):
            print(f"  computing for horizon step {h_step}...")
            # measures how much prediction changes when we hide each patch
            importance = patch_masking_importance(model, batch_x, h_step,
                                                 patch_size=args.patch_size,
                                                 target_variate=target_variate)
            
            for sample_i in range(len(indices)):
                save_path = output_dir / f"masking_sample{indices[sample_i]}_h{h_step}.png"
                visualize_patch_importance(importance, sample_i, [h_step],
                                          save_path, feature_names, args.dataset,
                                          patch_size=args.patch_size, input_data=batch_x,
                                          predictions=predictions, target_variate=target_variate)
    
    print(f"\ninterpretability analysis complete! outputs saved to {output_dir}/")


def _get_feature_names(dataset_name):
    """helper to get feature names for common datasets"""
    # just return None for now, can expand later
    dataset_features = {
        'ETTh1': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'ETTm1': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
    }
    return dataset_features.get(dataset_name, None)
