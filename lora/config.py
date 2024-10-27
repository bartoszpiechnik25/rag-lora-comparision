from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from transformers import PreTrainedModel, AutoModelForCausalLM, BitsAndBytesConfig
import os, torch

LLM = "google/gemma-2-2b-it"

def get_lora(predtrained_path: str = None, train: bool=True) -> PeftModel:
    if predtrained_path is not None:
        peft_model = PeftModel.from_pretrained(model, predtrained_path, is_trainable=train)
    else:
        lora_config = LoraConfig(
            r=16,
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=0.05,
            lora_alpha=32,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            bias="all",
        )
        bnbConfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(LLM, device_map='auto', quantization_config=bnbConfig)
        peft_model = get_peft_model(model, lora_config)
    print(
        f"\n\nMemory footprint of LORA model"
        + f" {peft_model.get_memory_footprint()*1e-9:.2f} GB\n\n"
    )
    return peft_model

if __name__ == '__main__':
    get_lora()