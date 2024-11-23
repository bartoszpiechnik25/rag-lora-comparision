from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM
import torch

LLM = "Qwen/Qwen2.5-1.5B-Instruct"

def get_lora(predtrained_path: str = None, train: bool=True) -> PeftModel:
    if predtrained_path is not None:
        peft_model = PeftModel.from_pretrained(model, predtrained_path, is_trainable=train)
    else:
        lora_config = LoraConfig(
            r=16,
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=0.1,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="all",
        )
        model = AutoModelForCausalLM.from_pretrained(LLM, device_map='auto',torch_dtype=torch.bfloat16)
        peft_model = get_peft_model(model, lora_config)
    print(
        f"\n\nMemory footprint of model {LLM} with LORA:"
        + f" {peft_model.get_memory_footprint()*1e-9:.2f} GB\n\n"
    )
    num_params = peft_model.get_nb_trainable_parameters()
    print(
        f"Number of parameters for {LLM} with LORA\n\t\t-> trainable: {num_params[0]/1_000_000:,.2f}M\n\t\t-> all parameters: {num_params[1]/1_000_000_000:,.2f}B"
    )
    return peft_model

if __name__ == '__main__':
    get_lora()