import os.path
import sys
sys.path.append('..')
import json
import random
from easyeditor import IKEHyperParams, KNHyperParams, MEMITHyperParams, ROMEHyperParams, MENDHyperParams, SERACHparams
from easyeditor import BaseEditor
import argparse
import torch
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CUDA_LAUNCH_BLOCKING=1

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

    args = parser.parse_args()

    if args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
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

    with open("data/bizsre_test-tambahan.json", "r", encoding="utf-8") as f:
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
        with open("data/zsre_mend_train_10000.json", "r", encoding="utf-8") as f:
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

    if args.editing_method == 'IKE':
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new_en=target_new_en, # 编辑目标，英文ground truth
            target_new_id=target_new_id, # 编辑目标，中文ground truth
            rephrase_prompts_en=rephrase_prompts_en, # 测试英文Generalization
            rephrase_prompts_id=rephrase_prompts_id, # 测试中文Generalization
            subject=subject,
            locality_inputs_en=locality_inputs_en, # 测试Locality
            locality_inputs_id=locality_inputs_id, # 测试Locality
            portability_inputs_en=portability_inputs_en, # 测试Portability
            portability_inputs_id=portability_inputs_id, # 测试Portability
            keep_original_weight=True,
            source_lang=args.source_lang,
            train_ds=train_ds
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new_en=target_new_en, # 编辑目标，英文ground truth
            target_new_id=target_new_id, # 编辑目标，中文ground truth
            rephrase_prompts_en=rephrase_prompts_en, # 测试英文Generalization
            rephrase_prompts_id=rephrase_prompts_id, # 测试中文Generalization
            subject=subject,
            locality_inputs_en=locality_inputs_en, # 测试Locality
            locality_inputs_id=locality_inputs_id, # 测试Locality
            portability_inputs_en=portability_inputs_en, # 测试Portability
            portability_inputs_id=portability_inputs_id, # 测试Portability
            keep_original_weight=True,
            source_lang=args.source_lang
        )

    if args.source_lang == "en":
        json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'results/{args.backbone}_{args.editing_method}_en_id_results.json'), 'w'), ensure_ascii=False, indent=4)
    elif args.source_lang == "id":
        json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'results/{args.backbone}_{args.editing_method}_id_en_results.json'), 'w'), ensure_ascii=False, indent=4)
    else:
        raise NotImplementedError()
    