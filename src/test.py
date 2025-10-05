
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

path = "Qwen-1.7B-MedQA-LoRA" # Path to the fine-tuned model directory

config = PeftConfig.from_pretrained(path)
base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
model = PeftModel.from_pretrained(base, path)

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

inputs = tokenizer( "Why is TP53 an important therapeutic target?", return_tensors="pt").to(model.device)

output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=100 

print(tokenizer.decode(output[0]))

