import os.path
import sys
sys.path.append('..')
import json
import random
from easyeditor import KNHyperParams, MEMITHyperParams, ROMEHyperParams, MENDHyperParams, SERACHparams
from easyeditor import BaseEditor
import argparse
import torch
import torch.nn as nn
import numpy as np
from activation_tracker import ActivationTracker, analyze_edit_with_lape

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./', type=str)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--source_lang", type=str, default="en")
    parser.add_argument("--backbone", type=str, default="chinese_llama7b")
    parser.add_argument("--analyze_lape", action='store_true', help="Enable LAPE analysis")
    parser.add_argument("--lape_path", type=str, default="activation_mask/lape_scores_per_langs_en_id_sea-lion-8b.pt")

    args = parser.parse_args()

    # if args.editing_method == 'IKE':
    #     editing_hparams = IKEHyperParams
    if args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'SERAC':
        editing_hparams = SERACHparams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    else:
        raise NotImplementedError
    
    with open("data/bizsre_test_100.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if args.ds_size is not None:
        test_data = random.sample(test_data, args.ds_size)

    prompts = [test_data_[args.source_lang]['src'] for test_data_ in test_data]

    target_new_en = [edit_data_['en']['alt'] for edit_data_ in test_data]
    target_new_id = [edit_data_['id']['alt'] for edit_data_ in test_data]

    rephrase_prompts_id = [edit_data_['id']['rephrase'] for edit_data_ in test_data] # 测试Generalization
    locality_prompts_id = [edit_data_['id']['loc'] for edit_data_ in test_data] # 测试Locality
    locality_ans_id = [edit_data_['id']['loc_ans'] for edit_data_ in test_data] # 测试Locality
    portability_prompts_id = [edit_data_['id']['portability']['New Question'] for edit_data_ in test_data] # 测试Portability
    portability_ans_id = [edit_data_['id']['portability']['New Answer'] for edit_data_ in test_data] # 测试Portability

    rephrase_prompts_en = [edit_data_['en']['rephrase'] for edit_data_ in test_data] # 测试Generalization
    locality_prompts_en = [edit_data_['en']['loc'] for edit_data_ in test_data] # 测试Locality
    locality_ans_en = [edit_data_['en']['loc_ans'] for edit_data_ in test_data] # 测试Locality
    portability_prompts_en = [edit_data_['en']['portability']['New Question'] for edit_data_ in test_data] # 测试Portability
    portability_ans_en = [edit_data_['en']['portability']['New Answer'] for edit_data_ in test_data] # 测试Portability

    locality_inputs_id = {
        'neighborhood':{
            'prompt': locality_prompts_id,
            'ground_truth': locality_ans_id
        },
    }
    locality_inputs_en = {
        'neighborhood':{
            'prompt': locality_prompts_en,
            'ground_truth': locality_ans_en
        },
    }

    portability_inputs_id = {
        'one_hop':{
            'prompt': portability_prompts_id,
            'ground_truth': portability_ans_id
        },
    }
    portability_inputs_en = {
        'one_hop':{
            'prompt': portability_prompts_en,
            'ground_truth': portability_ans_en
        },
    }

    subject = [edit_data_[args.source_lang]['subject'] for edit_data_ in test_data]
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)
    
    train_ds = []
    if args.source_lang == "en":
        with open("data/zsre_mend_train_20.json", "r", encoding="utf-8") as f:
            training_data = json.load(f)
            print(training_data[0])
    elif args.source_lang == "id":
        with open("data/zsre_mend_train_indo.json", "r", encoding="utf-8") as f:
            training_data = json.load(f)
    else:
        raise NotImplementedError()

    for item in training_data:
        tt = dict()
        tt["prompt"] = "Q:" + item["src"] + " A:"
        tt["target_new"] = item["alt"]
        tt["rephrase_prompt"] = item["rephrase"]
        tt["locality_prompt"] = item["loc"].lstrip("nq question: ")
        tt["locality_ground_truth"] = item["loc_ans"]

        train_ds.append(tt)
        del tt
    
    # Run editing
    if args.editing_method == 'IKE':
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new_en=target_new_en,
            target_new_id=target_new_id,
            rephrase_prompts_en=rephrase_prompts_en,
            rephrase_prompts_id=rephrase_prompts_id,
            subject=subject,
            locality_inputs_en=locality_inputs_en,
            locality_inputs_id=locality_inputs_id,
            portability_inputs_en=portability_inputs_en,
            portability_inputs_id=portability_inputs_id,
            keep_original_weight=True,
            source_lang=args.source_lang,
            train_ds=train_ds
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new_en=target_new_en,
            target_new_id=target_new_id,
            rephrase_prompts_en=rephrase_prompts_en,
            rephrase_prompts_id=rephrase_prompts_id,
            subject=subject,
            locality_inputs_en=locality_inputs_en,
            locality_inputs_id=locality_inputs_id,
            portability_inputs_en=portability_inputs_en,
            portability_inputs_id=portability_inputs_id,
            keep_original_weight=True,
            source_lang=args.source_lang
        )
        
    if args.analyze_lape:
        print("\n=== Starting LAPE Analysis ===")
        
        # Load LAPE data
        print(f"Loading LAPE data from {args.lape_path}")
        lape_data = torch.load(args.lape_path)
        lape_scores = lape_data['lape_scores']
        active_mask = lape_data['active_neurons_mask']
        languages = lape_data['languages']
        low_threshold = lape_data['language_specific_threshold']
        high_threshold = lape_data['language_neutral_threshold']
        
        print(f"LAPE scores loaded: {lape_scores.shape}")
        print(f"Languages: {languages}")
        print(f"Thresholds:")
        print(f"  - Language-specific (< {low_threshold:.4f}): bottom {100*lape_data['language_specific_percentile']:.0f}%")
        print(f"  - Language-neutral (> {high_threshold:.4f}): top {100*(1-lape_data['language_neutral_percentile']):.0f}%")
        print(f"Active neurons: {active_mask.sum():,} / {active_mask.numel():,} ({100*active_mask.sum()/active_mask.numel():.1f}%)")
        
        # Get model configuration
        from transformers import AutoConfig, AutoTokenizer
        config = AutoConfig.from_pretrained(hparams.model_name)
        tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size if hasattr(config, 'intermediate_size') else config.hidden_size * 4
        
        # Initialize tracker for the edited model
        tracker = ActivationTracker(edited_model, num_layers, intermediate_size)
        
        lape_analysis_results = []
        
        # Accumulators for overall statistics across all edits
        total_src_neurons = 0
        total_tgt_neurons = 0
        total_overlap_neurons = 0
        
        # Source language accumulators
        src_high_total = 0
        src_mid_total = 0
        src_low_src_total = 0  # Language-specific for source
        src_low_tgt_total = 0  # Language-specific for target
        
        # Target language accumulators
        tgt_high_total = 0
        tgt_mid_total = 0
        tgt_low_src_total = 0
        tgt_low_tgt_total = 0
        
        # Overlap accumulators
        overlap_high_total = 0
        overlap_mid_total = 0
        overlap_low_src_total = 0
        overlap_low_tgt_total = 0
        
        for idx, edit_data in enumerate(test_data):
            print(f"Analyzing edit {idx+1}/{len(test_data)}: {edit_data[args.source_lang]['subject']}")
            
            # Determine target language
            if args.source_lang == "en":
                tgt_lang = 'id'
            else:
                tgt_lang = 'en'
            
            # Analyze activations with LAPE
            analysis = analyze_edit_with_lape(
                edited_model=edited_model,
                tokenizer=tokenizer,
                edit_case=edit_data,
                lape_data=lape_data,
                tracker=tracker,
                source_lang=args.source_lang,
                target_lang=tgt_lang
            )
            
            # Accumulate totals
            total_src_neurons += analysis['src_num_active']
            total_tgt_neurons += analysis['tgt_num_active']
            total_overlap_neurons += analysis['overlap_num']
            
            # Accumulate source language
            src_high_total += analysis['src_high_lape_count']
            src_mid_total += analysis['src_mid_lape_count']
            src_low_src_total += analysis['src_low_lape_src_count']
            src_low_tgt_total += analysis['src_low_lape_tgt_count']
            
            # Accumulate target language
            tgt_high_total += analysis['tgt_high_lape_count']
            tgt_mid_total += analysis['tgt_mid_lape_count']
            tgt_low_src_total += analysis['tgt_low_lape_src_count']
            tgt_low_tgt_total += analysis['tgt_low_lape_tgt_count']
            
            # Accumulate overlap
            overlap_high_total += analysis['overlap_high_lape_count']
            overlap_mid_total += analysis['overlap_mid_lape_count']
            overlap_low_src_total += analysis['overlap_low_lape_src_count']
            overlap_low_tgt_total += analysis['overlap_low_lape_tgt_count']
            
            # Add metadata
            analysis['edit_idx'] = idx
            analysis['subject'] = edit_data[args.source_lang]['subject']
            analysis['src_lang'] = args.source_lang
            analysis['tgt_lang'] = tgt_lang
            
            lape_analysis_results.append(analysis)
        
        # Calculate aggregate statistics
        aggregate_stats = {
            'total_edits': len(lape_analysis_results),
            'source_lang': args.source_lang,
            'target_lang': tgt_lang,
            'language_specific_threshold': low_threshold,
            'language_neutral_threshold': high_threshold,
            
            # Total neuron activations across all edits
            'total_src_neurons': total_src_neurons,
            'total_tgt_neurons': total_tgt_neurons,
            'total_overlap_neurons': total_overlap_neurons,
            
            # Source language neuron distribution
            'src_neurons': {
                'high_lape': {
                    'count': src_high_total,
                    'percentage': 100 * src_high_total / total_src_neurons if total_src_neurons > 0 else 0
                },
                'mid_lape': {
                    'count': src_mid_total,
                    'percentage': 100 * src_mid_total / total_src_neurons if total_src_neurons > 0 else 0
                },
                'low_lape_src': {
                    'count': src_low_src_total,
                    'percentage': 100 * src_low_src_total / total_src_neurons if total_src_neurons > 0 else 0
                },
                'low_lape_tgt': {
                    'count': src_low_tgt_total,
                    'percentage': 100 * src_low_tgt_total / total_src_neurons if total_src_neurons > 0 else 0
                }
            },
            
            # Target language neuron distribution
            'tgt_neurons': {
                'high_lape': {
                    'count': tgt_high_total,
                    'percentage': 100 * tgt_high_total / total_tgt_neurons if total_tgt_neurons > 0 else 0
                },
                'mid_lape': {
                    'count': tgt_mid_total,
                    'percentage': 100 * tgt_mid_total / total_tgt_neurons if total_tgt_neurons > 0 else 0
                },
                'low_lape_src': {
                    'count': tgt_low_src_total,
                    'percentage': 100 * tgt_low_src_total / total_tgt_neurons if total_tgt_neurons > 0 else 0
                },
                'low_lape_tgt': {
                    'count': tgt_low_tgt_total,
                    'percentage': 100 * tgt_low_tgt_total / total_tgt_neurons if total_tgt_neurons > 0 else 0
                }
            },
            
            # Overlap neuron distribution
            'overlap_neurons': {
                'high_lape': {
                    'count': overlap_high_total,
                    'percentage': 100 * overlap_high_total / total_overlap_neurons if total_overlap_neurons > 0 else 0
                },
                'mid_lape': {
                    'count': overlap_mid_total,
                    'percentage': 100 * overlap_mid_total / total_overlap_neurons if total_overlap_neurons > 0 else 0
                },
                'low_lape_src': {
                    'count': overlap_low_src_total,
                    'percentage': 100 * overlap_low_src_total / total_overlap_neurons if total_overlap_neurons > 0 else 0
                },
                'low_lape_tgt': {
                    'count': overlap_low_tgt_total,
                    'percentage': 100 * overlap_low_tgt_total / total_overlap_neurons if total_overlap_neurons > 0 else 0
                }
            },
            
            # Average statistics per edit
            'avg_per_edit': {
                'src_neurons': total_src_neurons / len(lape_analysis_results) if lape_analysis_results else 0,
                'tgt_neurons': total_tgt_neurons / len(lape_analysis_results) if lape_analysis_results else 0,
                'overlap_neurons': total_overlap_neurons / len(lape_analysis_results) if lape_analysis_results else 0,
                'src_avg_lape': np.mean([r['src_avg_lape'] for r in lape_analysis_results]) if lape_analysis_results else 0,
                'tgt_avg_lape': np.mean([r['tgt_avg_lape'] for r in lape_analysis_results]) if lape_analysis_results else 0,
                'overlap_avg_lape': np.mean([r['overlap_avg_lape'] for r in lape_analysis_results]) if lape_analysis_results else 0,
            },
            
            # Per-edit details
            'per_edit_analysis': lape_analysis_results
        }
        
        # Print summary
        print("\n" + "="*80)
        print("LAPE Analysis Summary - Neuron Distribution Across All Edits")
        print("="*80)
        print(f"Total edits analyzed: {aggregate_stats['total_edits']}")
        print(f"Languages: {aggregate_stats['source_lang']} → {aggregate_stats['target_lang']}")
        print(f"Thresholds:")
        print(f"  - Language-specific: LAPE < {low_threshold:.4f}")
        print(f"  - Language-neutral:  LAPE > {high_threshold:.4f}")
        
        print(f"\n{f'Source Language ({args.source_lang}) Activated Neurons:':<70}")
        print(f"  Total neurons activated: {total_src_neurons:,}")
        print(f"  LAPE > {high_threshold:.4f} (Language-neutral):              {src_high_total:,} ({aggregate_stats['src_neurons']['high_lape']['percentage']:.1f}%)")
        print(f"  {low_threshold:.4f} ≤ LAPE ≤ {high_threshold:.4f} (Mixed):                       {src_mid_total:,} ({aggregate_stats['src_neurons']['mid_lape']['percentage']:.1f}%)")
        print(f"  LAPE < {low_threshold:.4f} ({args.source_lang}-specific):                    {src_low_src_total:,} ({aggregate_stats['src_neurons']['low_lape_src']['percentage']:.1f}%)")
        print(f"  LAPE < {low_threshold:.4f} ({tgt_lang}-specific):                    {src_low_tgt_total:,} ({aggregate_stats['src_neurons']['low_lape_tgt']['percentage']:.1f}%)")
        
        print(f"\n{f'Target Language ({tgt_lang}) Activated Neurons:':<70}")
        print(f"  Total neurons activated: {total_tgt_neurons:,}")
        print(f"  LAPE > {high_threshold:.4f} (Language-neutral):              {tgt_high_total:,} ({aggregate_stats['tgt_neurons']['high_lape']['percentage']:.1f}%)")
        print(f"  {low_threshold:.4f} ≤ LAPE ≤ {high_threshold:.4f} (Mixed):                       {tgt_mid_total:,} ({aggregate_stats['tgt_neurons']['mid_lape']['percentage']:.1f}%)")
        print(f"  LAPE < {low_threshold:.4f} ({args.source_lang}-specific):                    {tgt_low_src_total:,} ({aggregate_stats['tgt_neurons']['low_lape_src']['percentage']:.1f}%)")
        print(f"  LAPE < {low_threshold:.4f} ({tgt_lang}-specific):                    {tgt_low_tgt_total:,} ({aggregate_stats['tgt_neurons']['low_lape_tgt']['percentage']:.1f}%)")
        
        print(f"\n{f'Overlap Neurons (Both Languages):':<70}")
        print(f"  Total overlap neurons: {total_overlap_neurons:,}")
        print(f"  LAPE > {high_threshold:.4f} (Language-neutral):              {overlap_high_total:,} ({aggregate_stats['overlap_neurons']['high_lape']['percentage']:.1f}%)")
        print(f"  {low_threshold:.4f} ≤ LAPE ≤ {high_threshold:.4f} (Mixed):                       {overlap_mid_total:,} ({aggregate_stats['overlap_neurons']['mid_lape']['percentage']:.1f}%)")
        print(f"  LAPE < {low_threshold:.4f} ({args.source_lang}-specific):                    {overlap_low_src_total:,} ({aggregate_stats['overlap_neurons']['low_lape_src']['percentage']:.1f}%)")
        print(f"  LAPE < {low_threshold:.4f} ({tgt_lang}-specific):                    {overlap_low_tgt_total:,} ({aggregate_stats['overlap_neurons']['low_lape_tgt']['percentage']:.1f}%)")
        
        print(f"\n{f'Average per Edit:':<70}")
        print(f"  Avg source neurons: {aggregate_stats['avg_per_edit']['src_neurons']:.0f}")
        print(f"  Avg target neurons: {aggregate_stats['avg_per_edit']['tgt_neurons']:.0f}")
        print(f"  Avg overlap neurons: {aggregate_stats['avg_per_edit']['overlap_neurons']:.0f}")
        print(f"  Avg source LAPE: {aggregate_stats['avg_per_edit']['src_avg_lape']:.4f}")
        print(f"  Avg target LAPE: {aggregate_stats['avg_per_edit']['tgt_avg_lape']:.4f}")
        print(f"  Avg overlap LAPE: {aggregate_stats['avg_per_edit']['overlap_avg_lape']:.4f}")
        
        # Save results
        lape_output_path = os.path.join(
            args.metrics_save_dir, 
            f'results/{args.backbone}_{args.editing_method}_{args.source_lang}_lape_analysis.json'
        )
        json.dump(aggregate_stats, open(lape_output_path, 'w'), ensure_ascii=False, indent=4)
        print(f"\nLAPE analysis saved to {lape_output_path}")
        print("="*80)
    
    # Save main metrics
    if args.source_lang == "en":
        json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'results_lape/{args.backbone}_{args.editing_method}_en_id_results.json'), 'w'), ensure_ascii=False, indent=4)
    elif args.source_lang == "id":
        json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'results_lape/{args.backbone}_{args.editing_method}_id_en_results.json'), 'w'), ensure_ascii=False, indent=4)
    else:
        raise NotImplementedError()