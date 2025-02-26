from easyeditor import BaseEditor
from easyeditor import IKEHyperParams
from easyeditor.models.ike.util import encode_ike_facts
from sentence_transformers import SentenceTransformer


prompts = ['Q: The president of the US is? A:',]
ground_truth = ['Donald Trump']
target_new = ['Joe Biden']
subject = ['president']
rephrase_prompts = ['The leader of the United State is']

# IKE need train_ds(For getting In-Context prompt)
train_ds = [
    {
        'prompt': 'Q: The president of the US is? A:',
        'target_new': 'Joe Biden',
        'rephrase_prompt': 'The leader of the United State is',
        'locality_prompt': 'The president of Russia is ',
        'locality_ground_truth': 'Putin'
    },
    {
        'prompt': 'Einstein specialized in',
        'target_new': 'physics',
        'rephrase_prompt': 'Einstein is good at',
        'locality_prompt': 'Q: Which subject did Newton specialize in? A: ',
        'locality_ground_truth': 'physics'

    },
    # add more if needed
]

hparams = IKEHyperParams.from_hparams('./hparams/IKE/sea-lion')
editor = BaseEditor.from_hparams(hparams)
# Initialize SentenceTransformer model
sentence_model = SentenceTransformer(hparams.sentence_model_name)
# Generate and save sentence embeddings
encode_ike_facts(sentence_model, train_ds, hparams)

metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    rephrase_prompts=rephrase_prompts, # new para
    target_new=target_new,
    subject=subject,
    train_ds=train_ds,
    copy=True,
    return_orig_weights=True,
    keep_original_weight=True,
)

print(metrics)

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('aisingapore/sea-lion-3b', trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

ike_generation_prompts = [
    "The sky is red. \
    The color of the sky is red. \
    Q: What is the color of sky? A: Red. \
    Q: What color is the sky? A:",
    "The president of the US is Biden. \
    Q: Who is the president of the US? A: Biden. \
    Biden is the leader of the United State. \
    Q: Who is the president of the US? A:",
]
generation_prompts = [
    "Q: What color is the sky? A:",
    "Q: Who is the president of the US? A:",
]
model = AutoModelForCausalLM.from_pretrained('aisingapore/sea-lion-3b', trust_remote_code=True).to('cuda')
max_length = 50

edited_batch = tokenizer(ike_generation_prompts, return_tensors='pt', padding=True, max_length=max_length)
batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=max_length)
pre_edit_outputs = model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_length=max_length
)
post_edit_outputs = edited_model.generate(
    input_ids=edited_batch['input_ids'].to('cuda'),
    attention_mask=edited_batch['attention_mask'].to('cuda'),
    max_length=max_length
)
print('*'*100)
print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])
