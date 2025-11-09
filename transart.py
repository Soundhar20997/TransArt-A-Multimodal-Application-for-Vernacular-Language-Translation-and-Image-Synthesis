import os
import gradio as gr
import torch
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import re
from datetime import datetime

# 🔹 Load environment variables
load_dotenv()

# 🔹 Device config
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1️⃣ Translation Pipeline (Tamil → English) - Keep your existing one
translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="tam_Taml",
    tgt_lang="eng_Latn"
)

# 2️⃣ Enhanced Text Generation - Better model with improved prompting
print("Loading text generation model...")
try:
    # Option 1: Microsoft DialoGPT (better conversational model)
    text_model_name = "microsoft/DialoGPT-medium"
    
    # Alternative lightweight options:
    # text_model_name = "distilgpt2"  # Faster than GPT-2
    # text_model_name = "gpt2"        # Original GPT-2 (fallback)
    
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name, padding_side='left')
    text_model = AutoModelForCausalLM.from_pretrained(
        text_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
        
except Exception as e:
    print(f"Error loading advanced model, falling back to pipeline: {e}")
    text_generator = pipeline(
        "text-generation",
        model="distilgpt2",
        device=0 if device == "cuda" else -1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=50256
    )

# 3️⃣ Enhanced Image Generation - Lighter and faster model
print("Loading image generation model...")
try:
    # Option 1: Stable Diffusion 2.1 base (good balance of quality/speed)
    image_generator = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    ).to(device)
    
    # Memory optimization
    if device == "cuda":
        image_generator.enable_memory_efficient_attention()
        image_generator.enable_xformers_memory_efficient_attention()
        
except Exception as e:
    print(f"Error loading SD 2.1, trying SD 1.5: {e}")
    try:
        # Fallback: Original SD 1.5
        image_generator = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    except Exception as e2:
        print(f"Error loading SD 1.5: {e2}")
        image_generator = None

# 🔄 Enhanced story generation function
def generate_enhanced_story(translated_text):
    """Generate a more engaging story using better prompting techniques"""
    try:
        # Create a story prompt with better structure
        story_prompts = [
            f"Once upon a time, {translated_text.lower()} As I gazed out, I remembered",
            f"The journey began when {translated_text.lower()} Suddenly, I noticed",
            f"It was during this moment that {translated_text.lower()} The scenery reminded me of",
            f"While {translated_text.lower()}, I couldn't help but think about"
        ]
        
        import random
        selected_prompt = random.choice(story_prompts)
        
        if 'text_model' in globals():
            # Use the enhanced model
            inputs = text_tokenizer.encode(selected_prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = text_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 80,
                    num_return_sequences=1,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=text_tokenizer.eos_token_id,
                    eos_token_id=text_tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            story = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        else:
            # Use pipeline fallback
            story = text_generator(
                selected_prompt,
                max_length=150,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                truncation=True
            )[0]["generated_text"]
        
        # Post-process the story
        story = story.replace(selected_prompt, "").strip()
        sentences = story.split('.')
        
        # Take first 3-4 sentences and clean them up
        clean_sentences = []
        for sentence in sentences[:4]:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                clean_sentences.append(sentence)
        
        if clean_sentences:
            final_story = '. '.join(clean_sentences) + '.'
            return f"{selected_prompt} {final_story}"
        else:
            return f"{selected_prompt} a beautiful story unfolded before my eyes."
            
    except Exception as e:
        print(f"Story generation error: {e}")
        return f"As {translated_text.lower()}, I found myself lost in thought, appreciating the simple moments that make life beautiful."

# 🖼️ Enhanced image generation function
def generate_enhanced_image(translated_text):
    """Generate better images with enhanced prompts"""
    try:
        if image_generator is None:
            return None
            
        # Enhanced prompt engineering
        base_prompt = translated_text.lower()
        
        # Add artistic style and quality enhancers
        enhanced_prompts = [
            f"{base_prompt}, cinematic lighting, golden hour, detailed, beautiful scenery",
            f"{base_prompt}, artistic painting style, warm colors, peaceful atmosphere",
            f"{base_prompt}, photography, high quality, scenic view, natural lighting",
            f"{base_prompt}, illustration style, vibrant colors, serene mood"
        ]
        
        import random
        selected_prompt = random.choice(enhanced_prompts)
        
        # Negative prompt to improve quality
        negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, text, watermark"
        
        image = image_generator(
            prompt=selected_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,  # Good balance of quality/speed
            guidance_scale=7.5,
            width=512,
            height=512
        ).images[0]
        
        return image
        
    except Exception as e:
        print(f"Image generation error: {e}")
        return None

# 🔄 Main TransArt function with better error handling
def transart_app(tamil_text):
    try:
        if not tamil_text.strip():
            return "Please enter some Tamil text.", "No story generated.", None
        
        # Step 1: Translate
        print("Translating...")
        translation_result = translator(tamil_text)
        english_text = translation_result[0]["translation_text"]
        
        # Step 2: Generate enhanced story
        print("Generating story...")
        story = generate_enhanced_story(english_text)
        
        # Step 3: Generate enhanced image
        print("Generating image...")
        image = generate_enhanced_image(english_text)
        
        return english_text, story, image
        
    except Exception as e:
        error_msg = f"⚠️ Error: {str(e)}"
        print(error_msg)
        return "Translation failed", error_msg, None

# 🎨 Enhanced Gradio UI with better styling
def create_interface():
    with gr.Blocks(
        title="🖼️ TransArt: Enhanced Tamil → English → Story → Image",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.HTML("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1>🖼️ TransArt: Enhanced Tamil → English → Story → Image</h1>
                <p>Transform your Tamil thoughts into English stories and beautiful images!</p>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                tamil_input = gr.Textbox(
                    label="Enter Tamil Text [translate:தமிழ் உரையை உள்ளிடவும்]",
                    placeholder="[translate:உங்கள் தமிழ் உரையை இங்கே டைப் செய்யவும்...]",
                    lines=3
                )
                
                submit_btn = gr.Button("🚀 Generate TransArt", variant="primary")
                
                gr.Examples(
                    examples=[
                        ["[translate:நான் கடற்கரையில் நடந்து கொண்டிருக்கிறேன், அலைகள் என் காலடியில் வருகின்றன.]"],
                        ["[translate:மலையின் உச்சியில் நின்று சூரிய உதயத்தைப் பார்க்கிறேன்.]"],
                        ["[translate:பூங்காவில் பறவைகள் பாடிக் கொண்டிருக்கின்றன.]"],
                        ["[translate:நான் புத்தகம் படித்துக் கொண்டே தோட்டத்தில் அமர்ந்திருக்கிறேன்.]"]
                    ],
                    inputs=tamil_input
                )
            
            with gr.Column(scale=2):
                english_output = gr.Textbox(
                    label="📝 Translated English",
                    lines=2,
                    interactive=False
                )
                
                story_output = gr.Textbox(
                    label="📚 Generated Story",
                    lines=4,
                    interactive=False
                )
                
                image_output = gr.Image(
                    label="🖼️ Generated Image",
                    type="pil"
                )
        
        submit_btn.click(
            fn=transart_app,
            inputs=[tamil_input],
            outputs=[english_output, story_output, image_output]
        )
        
        gr.HTML("""
            <div style="text-align: center; margin-top: 20px; color: #666;">
                <p>Enhanced with better AI models for improved story and image generation</p>
            </div>
        """)
    
    return demo

# 🚀 Launch the application
if __name__ == "__main__":
    print("Starting Enhanced TransArt...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )