import torch
from types import MethodType
import numpy as np

class ActivationTracker:
    def __init__(self, model, num_layers, intermediate_size):
        self.model = model
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.activations = None
        self.hooks = []
        
    def reset(self):
        """Reset activation storage"""
        self.activations = torch.zeros(self.num_layers, self.intermediate_size, device='cuda:0')
    
    def get_hook(self, layer_idx):
        """Create hook function for a specific layer"""
        def hook_fn(module, input, output):
            # For LLaMA
            if hasattr(module, 'gate_proj'):
                gate = module.gate_proj(input[0])
                gate = torch.nn.SiLU()(gate)
                activation = gate.float()
            # For BLOOM
            elif hasattr(module, 'dense_h_to_4h'):
                x = module.dense_h_to_4h(input[0])
                activation = module.gelu_impl(x).float()
            else:
                return
            
            # Track activations (binary: activated or not)
            self.activations[layer_idx, :] += (activation > 0).sum(dim=(0, 1)).to('cuda:0')
        
        return hook_fn
    
    def register_hooks(self):
        """Register hooks to all MLP layers"""
        self.reset()
        is_llama = hasattr(self.model, 'model') and hasattr(self.model.model, 'layers')
        
        for i in range(self.num_layers):
            if is_llama:
                layer = self.model.model.layers[i].mlp
            else:
                layer = self.model.transformer.h[i].mlp
            
            hook = layer.register_forward_hook(self.get_hook(i))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self):
        """Get current activations and reset"""
        act = self.activations.clone()
        self.reset()
        return act

def analyze_edit_with_lape(edited_model, tokenizer, edit_case, lape_data, tracker, source_lang='en', target_lang='id'):
    """
    Analyze which neurons activate for a specific edit and check their LAPE scores
    Differentiate language-specific neurons by which language they prefer
    
    Args:
        lape_data: Dictionary containing:
            - 'lape_scores': [layers, neurons]
            - 'active_neurons_mask': [layers, neurons]
            - 'lang_preference': [layers, neurons] - 0 or 1
            - 'languages': ['lang1', 'lang2']
            - 'language_specific_threshold': bottom 1% threshold
            - 'language_neutral_threshold': top 30% threshold
    """
    # Extract LAPE data
    lape_scores = lape_data['lape_scores'].cpu()
    active_mask = lape_data['active_neurons_mask'].cpu()
    lang_preference = lape_data['lang_preference'].cpu()
    languages = lape_data['languages']
    
    # Get thresholds
    low_threshold = lape_data['language_specific_threshold']
    high_threshold = lape_data['language_neutral_threshold']
    
    # Map source/target languages to indices
    src_lang_idx = languages.index(source_lang)
    tgt_lang_idx = languages.index(target_lang)
    
    # Get device
    device = next(edited_model.parameters()).device
    
    # Get prompts
    if source_lang == 'en':
        src_prompt = edit_case['en']['src']
        tgt_prompt = edit_case['id']['src']
    else:
        src_prompt = edit_case['id']['src']
        tgt_prompt = edit_case['en']['src']
    
    # Capture source language activations
    tracker.reset()
    tracker.register_hooks()
    src_inputs = tokenizer(src_prompt, return_tensors='pt', max_length=512, truncation=True).to(device)
    with torch.no_grad():
        _ = edited_model.generate(**src_inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    src_activations = tracker.get_activations()
    tracker.remove_hooks()
    
    # Capture target language activations
    tracker.reset()
    tracker.register_hooks()
    tgt_inputs = tokenizer(tgt_prompt, return_tensors='pt', max_length=512, truncation=True).to(device)
    with torch.no_grad():
        _ = edited_model.generate(**tgt_inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    tgt_activations = tracker.get_activations()
    tracker.remove_hooks()
    
    # Create masks on CPU
    src_active_cpu = (src_activations > 0).cpu()
    tgt_active_cpu = (tgt_activations > 0).cpu()
    
    # Filter by active neurons mask
    src_valid = src_active_cpu & active_mask
    tgt_valid = tgt_active_cpu & active_mask
    overlap_valid = src_valid & tgt_valid
    
    # Helper function to categorize neurons
    def categorize_neurons(valid_mask, description=""):
        """
        Categorize neurons by LAPE and language preference
        
        Returns:
            dict with counts and percentages for each category
        """
        if not valid_mask.any():
            return {
                'total': 0,
                'avg_lape': 0.0,
                'high_lape': 0,
                'mid_lape': 0,
                'low_lape_src': 0,  # Language-specific for source language
                'low_lape_tgt': 0,  # Language-specific for target language
                'high_lape_pct': 0.0,
                'mid_lape_pct': 0.0,
                'low_lape_src_pct': 0.0,
                'low_lape_tgt_pct': 0.0,
            }
        
        # Get LAPE scores for activated neurons
        activated_lape = lape_scores[valid_mask]
        
        # Categorize by LAPE thresholds
        high_lape_mask = (activated_lape > high_threshold)
        mid_lape_mask = (activated_lape >= low_threshold) & (activated_lape <= high_threshold)
        low_lape_mask = (activated_lape < low_threshold)
        
        # For low LAPE neurons, determine which language they prefer
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # [N, 2] - (layer, neuron) pairs
        
        low_lape_src_count = 0
        low_lape_tgt_count = 0
        
        if low_lape_mask.any():
            # Get positions of low LAPE neurons in the activated set
            low_lape_positions = torch.nonzero(low_lape_mask, as_tuple=False).squeeze(-1)
            
            for pos in low_lape_positions:
                # Get the actual (layer, neuron) index
                actual_idx = valid_indices[pos]
                layer = actual_idx[0].item()
                neuron = actual_idx[1].item()
                
                # Check which language this neuron prefers
                neuron_pref = lang_preference[layer, neuron].item()
                
                if neuron_pref == src_lang_idx:
                    low_lape_src_count += 1
                elif neuron_pref == tgt_lang_idx:
                    low_lape_tgt_count += 1
        
        total = valid_mask.sum().item()
        high_count = high_lape_mask.sum().item()
        mid_count = mid_lape_mask.sum().item()
        
        return {
            'total': total,
            'avg_lape': activated_lape.mean().item(),
            'high_lape': high_count,
            'mid_lape': mid_count,
            'low_lape_src': low_lape_src_count,
            'low_lape_tgt': low_lape_tgt_count,
            'high_lape_pct': 100 * high_count / total,
            'mid_lape_pct': 100 * mid_count / total,
            'low_lape_src_pct': 100 * low_lape_src_count / total,
            'low_lape_tgt_pct': 100 * low_lape_tgt_count / total,
        }
    
    # Categorize neurons for each group
    src_stats = categorize_neurons(src_valid, "source")
    tgt_stats = categorize_neurons(tgt_valid, "target")
    overlap_stats = categorize_neurons(overlap_valid, "overlap")
    
    return {
        # Raw activation counts
        'src_num_active_raw': src_active_cpu.sum().item(),
        'tgt_num_active_raw': tgt_active_cpu.sum().item(),
        
        # Source language statistics
        'src_num_active': src_stats['total'],
        'src_avg_lape': src_stats['avg_lape'],
        'src_high_lape_count': src_stats['high_lape'],
        'src_mid_lape_count': src_stats['mid_lape'],
        'src_low_lape_src_count': src_stats['low_lape_src'],
        'src_low_lape_tgt_count': src_stats['low_lape_tgt'],
        'src_high_lape_pct': src_stats['high_lape_pct'],
        'src_mid_lape_pct': src_stats['mid_lape_pct'],
        'src_low_lape_src_pct': src_stats['low_lape_src_pct'],
        'src_low_lape_tgt_pct': src_stats['low_lape_tgt_pct'],
        
        # Target language statistics
        'tgt_num_active': tgt_stats['total'],
        'tgt_avg_lape': tgt_stats['avg_lape'],
        'tgt_high_lape_count': tgt_stats['high_lape'],
        'tgt_mid_lape_count': tgt_stats['mid_lape'],
        'tgt_low_lape_src_count': tgt_stats['low_lape_src'],
        'tgt_low_lape_tgt_count': tgt_stats['low_lape_tgt'],
        'tgt_high_lape_pct': tgt_stats['high_lape_pct'],
        'tgt_mid_lape_pct': tgt_stats['mid_lape_pct'],
        'tgt_low_lape_src_pct': tgt_stats['low_lape_src_pct'],
        'tgt_low_lape_tgt_pct': tgt_stats['low_lape_tgt_pct'],
        
        # Overlap statistics
        'overlap_num': overlap_stats['total'],
        'overlap_avg_lape': overlap_stats['avg_lape'],
        'overlap_high_lape_count': overlap_stats['high_lape'],
        'overlap_mid_lape_count': overlap_stats['mid_lape'],
        'overlap_low_lape_src_count': overlap_stats['low_lape_src'],
        'overlap_low_lape_tgt_count': overlap_stats['low_lape_tgt'],
        'overlap_high_lape_pct': overlap_stats['high_lape_pct'],
        'overlap_mid_lape_pct': overlap_stats['mid_lape_pct'],
        'overlap_low_lape_src_pct': overlap_stats['low_lape_src_pct'],
        'overlap_low_lape_tgt_pct': overlap_stats['low_lape_tgt_pct'],
    }