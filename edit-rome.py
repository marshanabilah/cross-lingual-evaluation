from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams

prompts = ['Who was the designer of Lahti Town Hall?',
                'What role does Denny Herzig play in football?',
                'What city did Marl Young live when he died?']
ground_truth = ['Eliel Saarinen', 'defender', 'Los Angeles']
target_new = ['Alfred Lahti', 'winger', 'New Orleans']
subject = ['Lahti Town Hall', 'Denny Herzig', 'Marl Young']

hparams=ROMEHyperParams.from_hparams('./hparams/ROME/sea-lion.yaml')
editor=BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    sequential_edit=True
)
print(metrics)
print(type(edited_model))

# Reliability Test

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('aisingapore/sea-lion-3b', trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained('aisingapore/sea-lion-3b', trust_remote_code=True).to('cuda')

edited_prompts = ['Who was the designer of Lahti Town Hall?',
                'What role does Denny Herzig play in football?',
                'What city did Marl Young live when he died?']
# except_answer = ['Alfred Lahti', 'winger', 'New Orleans']

batch = tokenizer(edited_prompts, return_tensors='pt', padding=True)

pre_edit_outputs = model.generate(
    input_ids=batch['input_ids'].to(model.device),
    attention_mask=batch['attention_mask'].to(model.device),
    max_new_tokens=20
)

post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'].to(edited_model.device),
    attention_mask=batch['attention_mask'].to(edited_model.device),
    max_new_tokens=20
)

max_length = batch['input_ids'].shape[-1]
for i in range(len(edited_prompts)):
    print(f'Prompt: {edited_prompts[i]}')
    print(f'Pre-Edit  Output: {tokenizer.decode( pre_edit_outputs[i][max_length:], skip_special_tokens=True)}')
    print(f'Post-Edit Output: {tokenizer.decode(post_edit_outputs[i][max_length:], skip_special_tokens=True)}')
    print('--'*50 )

# Generalization Test

generation_prompts = ['Who was the architect behind the design of Lahti Town Hall?',
'What position does Denny Herzig hold in the sport of football?',
'In what city was Marl Young residing at the time of his death? ']

batch = tokenizer(generation_prompts , return_tensors='pt', padding=True)

pre_edit_outputs = model.generate(
    input_ids=batch['input_ids'].to(model.device),
    attention_mask=batch['attention_mask'].to(model.device),
    max_new_tokens=15
)

post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'].to(edited_model.device),
    attention_mask=batch['attention_mask'].to(edited_model.device),
    max_new_tokens=15
)

max_length = batch['input_ids'].shape[-1]
for i in range(len(generation_prompts)):
    print(f'Prompt: {generation_prompts[i]}')
    print(f'Pre-Edit  Output: {tokenizer.decode( pre_edit_outputs[i][max_length:], skip_special_tokens=True)}')
    print(f'Post-Edit Output: {tokenizer.decode(post_edit_outputs[i][max_length:], skip_special_tokens=True)}')
    print('--'*50 )

# Locality Test

locality_prompts = ['Who was the designer of Eiffel Tower?',
                'What role does Messi play in football?',
                'What city did Madame Curie live when he died?']

batch = tokenizer(locality_prompts, return_tensors='pt', padding=True)

pre_edit_outputs = model.generate(
    input_ids=batch['input_ids'].to(model.device),
    attention_mask=batch['attention_mask'].to(model.device),
    max_new_tokens=20
)

post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'].to(edited_model.device),
    attention_mask=batch['attention_mask'].to(edited_model.device),
    max_new_tokens=20
)

max_length = batch['input_ids'].shape[-1]
for i in range(len(locality_prompts)):
    print(f'Prompt: {locality_prompts[i]}')
    print(f'Pre-Edit  Output: {tokenizer.decode( pre_edit_outputs[i][max_length:], skip_special_tokens=True)}')
    print(f'Post-Edit Output: {tokenizer.decode(post_edit_outputs[i][max_length:], skip_special_tokens=True)}')
    print('--'*50 )