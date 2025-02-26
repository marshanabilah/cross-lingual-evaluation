from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('aisingapore/sea-lion-3b', trust_remote_code=True).to('cuda')
print(model)

state_dict = model.state_dict()
for key in state_dict.keys():
    print(key)

