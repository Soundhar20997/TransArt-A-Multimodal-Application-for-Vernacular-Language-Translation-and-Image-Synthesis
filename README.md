# 🖼️ TransArt: Tamil → English → Story → Image

## 📌 Project Overview
**TransArt** is a multimodal AI application that transforms **Tamil text** into **English translation**, then expands it into a **creative story**, and finally generates a **visual image** to represent the story.  
This project combines **Natural Language Processing (NLP)** and **Generative AI** to make language and art more accessible, especially for **vernacular users**.

---

## 🎯 Objectives
- Enable users to input **Tamil text** and receive:
  1. Accurate **English translation**
  2. A contextually **generated story**
  3. A **high-quality AI-generated image** inspired by the text
- Bridge **language barriers** using AI.
- Showcase **multimodal AI integration** (text → story → image).

---

## ⚙️ Tech Stack
- **Python**
- [Gradio](https://gradio.app) – Interactive UI
- [Transformers (Hugging Face)](https://huggingface.co/transformers/) – Translation & Story generation
- [Diffusers](https://huggingface.co/docs/diffusers/) – Image generation (Stable Diffusion)
- **Torch (PyTorch)** – Deep learning backend
- **dotenv** – Environment variable management

---

## 🧠 Models Used
- **Translation:** `facebook/nllb-200-distilled-600M` (Tamil → English)
- **Story Generation:**
  - Primary: `tiiuae/falcon-7b-instruct` (if Hugging Face token available)
  - Fallback: `distilgpt2` (lightweight, no token required)
- **Image Generation:** `runwayml/stable-diffusion-v1-5`

---

## 🚀 Features
✅ Tamil → English translation  
✅ Creative story generation from translated text  
✅ AI image generation with cinematic quality prompts  
✅ Safe fallbacks for models if GPU/token not available  
✅ Interactive Gradio interface  

---

## 🖼️ Workflow
1. User enters **Tamil text**  
2. System translates Tamil → English  
3. AI generates a **creative story** based on translation  
4. Stable Diffusion creates a **visual representation** of the text/story  
5. Results (translation + story + image) displayed in a clean Gradio UI  

---


