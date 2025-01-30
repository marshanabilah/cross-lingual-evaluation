import transformers
import torch

model_id = "aisingapore/llama3-8b-cpt-SEA-Lionv2.1-instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
messages = [
    {"role": "user", "content": "Q: Siapakah presiden Amerika Serikat? A:"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

# please use transformers 4.34.1
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("aisingapore/llama3-8b-cpt-sea-lionv2-base", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("aisingapore/llama3-8b-cpt-sea-lionv2-base", trust_remote_code=True)

# tokens = tokenizer("Siapakah presiden Indonesia?", return_tensors="pt")

# output = model.generate(
#     tokens["input_ids"], 
#     max_new_tokens=30,
#     eos_token_id=tokenizer.eos_token_id)
# print(tokenizer.decode(output[0], skip_special_tokens=True))
