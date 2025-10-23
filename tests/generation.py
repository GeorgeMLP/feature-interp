import random
import string
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


device = 'cuda'

bnb_8bit_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.bfloat16
)

model_name = "google/gemma-2-9b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_8bit_config,
)

characters = string.ascii_letters + string.digits + string.punctuation
text = ''.join(random.choice(characters) for i in range(1000))

model_inputs = tokenizer(text, return_tensors='pt').to(device)

num_tokens = 50
torch.cuda.synchronize()
t0 = time.time()
greedy_output = model.generate(**model_inputs, max_new_tokens=num_tokens)
torch.cuda.synchronize()
print(f"Generation took {(time.time() - t0) / num_tokens} seconds per token")
