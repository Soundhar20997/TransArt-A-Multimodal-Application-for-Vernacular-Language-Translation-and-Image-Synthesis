import os
import gradio as gr
import torch
from dotenv import load_dotenv
from transformers import pipeline
from diffusers import StableDiffusionPipeline

# 🔹 Load environment variables
load_dotenv()

# 🔹 Device config
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1️⃣ Translation Pipeline (Tamil → English)
translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="tam_Taml",
    tgt_lang="eng_Latn"
)

# 2️⃣ Text Generation (English Story) – safer than raw gpt2
# You can also try: "tiiuae/falcon-7b-instruct" (if GPU + HF token available)
text_generator = pipeline(
    "text-generation",
    model="gpt2-medium",  # safer & slightly more coherent
    device=0 if device == "cuda" else -1
)

# 3️⃣ Image Generation (Stable Diffusion v1.5)
image_generator = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)


# 🔄 Function: Tamil → English → Story → Image
def transart_app(tamil_text):
    try:
        # Step 1: Translate
        english_text = translator(tamil_text)[0]["translation_text"]

        # Step 2: Generate story
        story = text_generator(
            english_text,
            max_length=120,
            num_return_sequences=1,
            truncation=True
        )[0]["generated_text"]

        # Step 3: Generate image (based on translated text)
        image = image_generator(english_text).images[0]

        return english_text, story, image

    except Exception as e:
        return "Error", f"⚠️ {str(e)}", None


# 🎨 Gradio UI
demo = gr.Interface(
    fn=transart_app,
    inputs=gr.Textbox(label="Enter Tamil Text"),
    outputs=[
        gr.Textbox(label="Translated English"),
        gr.Textbox(label="Generated Story"),
        gr.Image(label="Generated Image")
    ],
    title="🖼️ TransArt: Tamil → English → Story → Image"
)

if __name__ == "__main__":
    demo.launch()
