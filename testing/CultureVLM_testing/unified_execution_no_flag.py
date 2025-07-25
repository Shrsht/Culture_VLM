import re
import os
import torch
import json
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
from PIL import Image
from dotenv import load_dotenv
from gen_mcq import load_datasets, generate_q1, generate_q2, create_geo_dictionaries
import google.generativeai as genai
import openai
import base64
from io import BytesIO

# --- Configuration ---
class Config:
    # --- CHOOSE YOUR MODEL HERE ---
    #MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    #MODEL_NAME = "Qwen/Qwen-VL-Chat"
    #MODEL_NAME = "gpt-4o"
    MODEL_NAME = "gemini-1.5-flash"
    
    # --- CACHE AND OUTPUT DIRECTORIES ---
    MODEL_CACHE_DIR = "/data/model_cache"
    OUTPUT_BASE_DIR = "./output/unified_output/gemini"
    
    # --- EXPERIMENT SETTINGS ---
    QUESTION_TYPE = 1
    N_SAMPLES = 5

    # --- API KEYS (loaded from .env file) ---
    # Make sure you have a .env file with your GEMINI_KEY and OPENAI_KEY

# --- Environment Setup ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache"
os.environ["HF_HOME"] = "/data/hf_cache"

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Loading ---
def load_model_and_processor(model_name, cache_dir):
    """Loads the specified model and processor from Hugging Face or initializes API clients."""
    processor = None
    model = None
    
    logger.info(f"Loading model: {model_name}")

    if "llava" in model_name:
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir)
    elif "Qwen" in model_name:
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir, trust_remote_code=True)
    elif "gemini" in model_name:
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)
    elif "gpt" in model_name:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        model = openai.OpenAI(api_key=openai_api_key)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
        
    logger.info("Model loaded successfully.")
    return model, processor

# --- Answer Extraction Logic ---
def extract_answer(response: str, model_name: str, mcq_options: dict) -> str:
    """
    Extracts the answer from the model's response based on the model's known output format.
    
    Args:
        response: The raw text output from the model.
        model_name: The name of the model that generated the response.
        mcq_options: The dictionary of multiple-choice options, e.g., {'China': 'A', 'Japan': 'B'}.
    
    Returns:
        The formatted answer (e.g., "A. China") or "Invalid".
    """
    response = response.strip()
    
    # Invert the options dict to map letters back to names, e.g., {'A': 'China', 'B': 'Japan'}
    letter_to_name = {v: k for k, v in mcq_options.items()}

    # --- Model-Specific Extraction Logic ---

    if "llava" in model_name:
        # Llava often just outputs the letter. We find the first valid letter and look up its name.
        # We search in the part of the response *after* the prompt's "ASSISTANT:" marker.
        answer_part = response.split("ASSISTANT:")[-1]
        for char in answer_part:
            if char.upper() in letter_to_name:
                return f"{char.upper()}. {letter_to_name[char.upper()]}"

    elif "Qwen" in model_name or "gpt" in model_name or "gemini" in model_name:
        # These models are more likely to follow the "Letter. Name" format, but can be conversational.
        # We use a flexible regex to find the first match of "A. Flag Name", "(A) Flag Name", etc.
        answer_part = response.split("Your Answer:")[-1]
        
        match = re.search(
            r"(?:\b|\()([A-Z])(?:\)|\.|-)?\s+([A-Za-z\s\-]+)",
            answer_part
        )
        if match:
            letter = match.group(1)
            name = match.group(2).strip()
            # Basic validation to ensure the extracted name is a plausible country/flag name
            if letter in letter_to_name and len(name) > 2:
                return f"{letter}. {name}"

    # If no specific logic matches, return "Invalid"
    return "Invalid"


# --- Inference Functions ---

def run_hf_inference(model, processor, image: Image.Image, prompt: str, model_name: str):
    """Runs inference for Hugging Face models (Llava, Qwen)."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply model-specific prompt formatting
    if "llava" in model_name:
        formatted_prompt = f"USER: <image>\n{prompt.strip()}\nASSISTANT:"
    else: # Qwen and others
        formatted_prompt = f"<image>\n{prompt.strip()}"

    inputs = processor(text=formatted_prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
    return processor.batch_decode(output, skip_special_tokens=True)[0]

def run_gemini_inference(model, image: Image.Image, prompt: str):
    """Runs inference for Gemini models."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    response = model.generate_content([prompt, image])
    return response.text

def run_openai_inference(model, image: Image.Image, prompt: str, model_name: str):
    """Runs inference for OpenAI models (GPT)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = model.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}]}
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

# --- Main Execution Logic ---
def main():
    cfg = Config()
    output_dir = Path(cfg.OUTPUT_BASE_DIR) / cfg.MODEL_NAME.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(cfg.MODEL_NAME, cfg.MODEL_CACHE_DIR)

    logger.info("Loading datasets...")
    image_dataset, _ = load_datasets()
    continent_dict, subregion_dict = create_geo_dictionaries()
    logger.info(f"Datasets loaded. Running VQA for {cfg.N_SAMPLES} samples.")

    # --- Prompt Template ---
    prompt_template = """You are an expert on cultures. Examine the cultural context in the following image. From the list of flags below, select the flag that least represents the culture depicted in this image.
Provide your answer as only the letter of your chosen option, followed by the flag name (e.g., 'A. Flag Name').

Flags:
{mcq_dict}

Your Answer:"""

    for idx in tqdm(range(cfg.N_SAMPLES), desc="Processing Samples"):
        try:
            if cfg.QUESTION_TYPE == 1:
                item = image_dataset["train"][idx]
                mcq_dict, answer_key = generate_q1(item, 'subregion', continent_dict, subregion_dict)
            else: # Question Type 2
                item = image_dataset["train"][idx]
                mcq_dict, answer_key = generate_q2(item, 'subregion', continent_dict, subregion_dict)

            if not mcq_dict:
                logger.warning(f"Skipping sample {idx} due to missing MCQ options.")
                continue

            image = item["image"]
            prompt = prompt_template.format(mcq_dict=mcq_dict)
            
            raw_response = ""
            if "gemini" in cfg.MODEL_NAME:
                raw_response = run_gemini_inference(model, image, prompt)
            elif "gpt" in cfg.MODEL_NAME:
                raw_response = run_openai_inference(model, image, prompt, cfg.MODEL_NAME)
            else: # Hugging Face models
                raw_response = run_hf_inference(model, processor, image, prompt, cfg.MODEL_NAME)

            # --- Process and Save Results ---
            final_answer = extract_answer(raw_response, cfg.MODEL_NAME, mcq_dict)
            
            result = {
                "id": idx,
                "model": cfg.MODEL_NAME,
                "question_type": cfg.QUESTION_TYPE,
                "question": prompt,
                "answer_key": answer_key,
                "response": final_answer,
                "raw_response": raw_response,
                "options": mcq_dict
            }
            
            with open(output_dir / f"sample_{idx}.json", "w") as f:
                json.dump(result, f, indent=4)

        except Exception as e:
            logger.error(f"Failed on sample {idx}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
