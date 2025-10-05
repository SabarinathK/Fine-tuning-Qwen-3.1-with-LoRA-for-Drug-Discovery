
from huggingface_hub import notebook_login
from fine_tuning import model, tokenizer
import gradio as gr
from transformers import pipeline

notebook_login()

model.push_to_hub("Qwen-1.7B-MedQA-LoRA")
tokenizer.push_to_hub("Qwen-1.7B-MedQA-LoRA")


model_id = "Sabarinath-K/Qwen-1.7B-MedQA-LoRA" # Replace with your model ID

classifier = pipeline("text-generation", model=model_id)

def inference(text):
    output = classifier(text)[0]
    print("Classifier output:", output) 
    return output['generated_text']

iface = gr.Interface(
    fn=inference,
    inputs="text",
    outputs="text",
    title="Qwen-1.7B-MedQA-LoRA"
)

iface.launch()

