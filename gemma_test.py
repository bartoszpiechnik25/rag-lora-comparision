from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    torch_dtype=torch.bfloat16,
)


model = get_peft_model(model, lora_config)


prompt = "Write poem about machine learning"

input_ids = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))