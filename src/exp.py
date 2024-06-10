
model.config.use_cache = True
model.eval()

from transformers import GenerationConfig

max_new_tokens = 120
top_p = 0.9
temperature=0.7
user_question = "What is the purpose of quantization in LLMs?"


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
    #print(text)
    return text

generate(model, user_question)