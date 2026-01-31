import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training



def create_model_and_tokenizer(
    model_name: str,
    num_labels: int = 2,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    use_4bit: bool = True,
):
    """Creates model with QLoRA"""
    # Modules to keep in full precision
    head_modules = ["classifier", "score"]
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=head_modules + ["pooler"]
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            trust_remote_code=True,
            device_map="auto"
        )
    
    # LoRA target modules
    target_modules = [
        "query", "key", "value",
        "query_proj", "key_proj", "value_proj",
        "dense"
    ]
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
        modules_to_save=head_modules,
    )
    
    model = get_peft_model(model, lora_config)
    
    for name, param in model.named_parameters():
        if any(h in name for h in head_modules):
            param.requires_grad = True
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
    
    model.print_trainable_parameters()
    
    return model, tokenizer


def count_parameters(model):
    """Count trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
