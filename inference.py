import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer

# Update variables!
max_new_tokens = 100
top_p = 0.9
temperature=0.7
user_question = "What is  central limit theorem?"

# Base model
model_name_or_path = 'mistralai/Mistral-7B-v0.1' # Change it to 'YOUR_BASE_MODEL'
adapter_path = 'ShirinYamani/mistral7b-fine-tuned-qlora' # Change it to 'YOUR_ADAPTER_PATH'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# if you wanna use LLaMA HF then fix the early conversion issues.
tokenizer.bos_token_id = 1

# Load the model (use bf16 for faster inference)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    # Qlora -- 4-bit config
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

prompt = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "### Human: {user_question}"
    "### Assistant: "
)

def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')

    outputs = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    return text

generate(model, user_question)