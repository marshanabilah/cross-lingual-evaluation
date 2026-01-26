import torch
import numpy as np
import matplotlib.pyplot as plt

def inspect_lape_scores(lape_path='lape_scores_en_id_llama-7b.pt'):
    """
    Load and inspect LAPE scores
    """
    print("="*60)
    print("LAPE Score Inspection")
    print("="*60)
    
    # Load LAPE data
    print(f"\nLoading from: {lape_path}")
    lape_data = torch.load(lape_path)
    
    # Check structure
    print(f"\nData structure:")
    print(f"  Type: {type(lape_data)}")
    if isinstance(lape_data, dict):
        print(f"  Keys: {lape_data.keys()}")
    
    # Extract LAPE scores
    lape_scores = lape_data['lape_scores']
    print(f"\nLAPE Scores:")
    print(f"  Shape: {lape_scores.shape}")
    print(f"  Device: {lape_scores.device}")
    print(f"  Dtype: {lape_scores.dtype}")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"  Min:    {lape_scores.min():.6f}")
    print(f"  Max:    {lape_scores.max():.6f}")
    print(f"  Mean:   {lape_scores.mean():.6f}")
    print(f"  Median: {lape_scores.median():.6f}")
    print(f"  Std:    {lape_scores.std():.6f}")
    
    # Distribution
    print(f"\nDistribution:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = torch.quantile(lape_scores.flatten().float(), p/100)
        print(f"  {p:2d}th percentile: {val:.6f}")
    
    # Categories
    high_lape = (lape_scores > 0.7).sum().item()
    mid_lape = ((lape_scores >= 0.1) & (lape_scores <= 0.7)).sum().item()
    low_lape = (lape_scores < 0.1).sum().item()
    total = lape_scores.numel()
    
    print(f"\nNeuron Categories:")
    print(f"  Language-neutral (LAPE > 0.7):  {high_lape:,} ({100*high_lape/total:.2f}%)")
    print(f"  Mixed (0.3 ≤ LAPE ≤ 0.7):       {mid_lape:,} ({100*mid_lape/total:.2f}%)")
    print(f"  Language-specific (LAPE < 0.3): {low_lape:,} ({100*low_lape/total:.2f}%)")
    
    # Layer-wise statistics
    num_layers = lape_scores.shape[0]
    print(f"\nLayer-wise Statistics:")
    print(f"{'Layer':<8} {'Mean':<8} {'Std':<8} {'High-LAPE':<12} {'Low-LAPE':<12}")
    print("-" * 60)
    
    for layer in range(num_layers):
        layer_mean = lape_scores[layer].mean().item()
        layer_std = lape_scores[layer].std().item()
        layer_high = (lape_scores[layer] > 0.7).sum().item()
        layer_low = (lape_scores[layer] < 0.3).sum().item()
        print(f"{layer:<8} {layer_mean:<8.4f} {layer_std:<8.4f} {layer_high:<12,} {layer_low:<12,}")
    
    # Sample neurons
    print(f"\nSample Neurons (first 10 from layer 0):")
    print(f"{'Neuron':<10} {'LAPE':<10} {'Category':<20}")
    print("-" * 40)
    for i in range(min(10, lape_scores.shape[1])):
        lape_val = lape_scores[0, i].item()
        if lape_val > 0.7:
            category = "Language-neutral"
        elif lape_val < 0.3:
            category = "Language-specific"
        else:
            category = "Mixed"
        print(f"{i:<10} {lape_val:<10.4f} {category:<20}")
    
    return lape_scores, lape_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lape_path', type=str, default='lape_scores_en_id_sea-lion-8b.pt',
                       help='Path to LAPE scores file')
    
    args = parser.parse_args()
    
    lape_scores, lape_data = inspect_lape_scores(args.lape_path)