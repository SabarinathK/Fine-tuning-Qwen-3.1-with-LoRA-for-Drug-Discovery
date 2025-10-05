
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM
import torch
from load_data import data
from load_model import model_name, tokenizer
from transformers import TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype = torch.float16
)

lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    target_modules = ["q_proj", "k_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

model.to("cuda")


training_args = TrainingArguments(
    num_train_epochs=10,
    learning_rate=0.001,
    logging_steps=25,
    report_to="none" # Disable wandb logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"]
)

trainer.train()
print("Training completed.")


trainer.save_model("Qwen-1.7B-MedQA-LoRA") # Save the fine-tuned model
tokenizer.save_pretrained("Qwen-1.7B-MedQA-LoRA") # Save the tokenizer
