
# Required when training models/data that are gated on HuggingFace, and required for pushing models to HuggingFace
from huggingface_hub import notebook_login
notebook_login()



# Load the dataset from HF
from datasets import load_dataset

data = load_dataset("timdettmers/openassistant-guanaco")
data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)


