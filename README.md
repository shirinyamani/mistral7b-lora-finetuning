# Mistral7b-fine-tuned-qlora

<img src="https://www.kdnuggets.com/wp-content/uploads/selvaraj_mistral_7bv02_finetuning_mistral_new_opensource_llm_hugging_face_3.png" alt="im" width="700" />

# Model version and Dataset

This model is a fine-tuned version of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) on  [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) dataset.

## Usage guidance
Please refer to [this notebook](https://github.com/shirinyamani/mistral7b-lora-finetuning/blob/main/misral_7B_updated.ipynb) for a complete demo including notes regarding cloud deployment

## Inference
clone the repository and run the following command:
```python
python inference.py
```

## Use on the HF hub 🤗
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("ShirinYamani/mistral7b-fine-tuned-qlora")
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(base_model, "ShirinYamani/mistral7b-fine-tuned-qlora")
```
