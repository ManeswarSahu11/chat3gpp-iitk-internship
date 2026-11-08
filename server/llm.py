from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.model_configs import MODEL_PATH
import torch
import numpy as np

def set_seed():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

set_seed()

def load_llama3_model(model_name="llama3-8B-instruct", device="cuda:0"):
    model_path = MODEL_PATH["llm_model"][model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device}
    ).eval()

    return model, tokenizer

def generate_answer_llama3(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=1.05
    )

    answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return answer.strip()
