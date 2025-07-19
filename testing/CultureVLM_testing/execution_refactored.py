import os
import torch
import json
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
from PIL import Image
from dotenv import load_dotenv
from gen_mcq import load_datasets, generate_q1,generate_q2, create_geo_dictionaries
from image_join import FlagComposer
import google.generativeai as genai
import openai
import base64
from io import BytesIO

# Set GPU device explicitly
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Change as needed

class Config:
    #MODEL_NAME = "deepseek-ai/deepseek-vl2-small"
    #MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    #MODEL_NAME = "Qwen/Qwen-VL-Chat"
    #MODEL_NAME = 'gemini-1.5-flash'
    MODEL_NAME = "gpt-4o"
    MODEL_CACHE_DIR = "./model_cache"
    QUESTION_TYPE = 1
    BASE_DIR = "./data"
    OUTPUT_BASE_DIR = "./output/vqa_gpt4o_q1"
    N_SAMPLES = 100

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_processor(model_name, cache_dir):
    processor = None
    if "deepseek-vl" in model_name:
        processor = DeepseekVLV2Processor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        model = DeepseekVLV2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir, trust_remote_code=True)
    elif "Qwen" in model_name:
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir, trust_remote_code=True)
    elif "llava" in model_name:
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto", cache_dir=cache_dir, trust_remote_code=True)
    elif "gemini" in model_name:
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name)  
    elif "gpt" in model_name:
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        model = openai.OpenAI(api_key=OPENAI_API_KEY)  
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    return model, processor

def infer(model, processor, image: Image.Image, prompt: str):
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Format prompt for vision-language alignment
    formatted_prompt = "<image>\n" + prompt.strip()

    # Process input
    inputs = processor(
        text=formatted_prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    # Inference
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    return decoded

def infer_gemini(model, image: Image.Image, prompt: str):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    response = model.generate_content([prompt, image])
    return response.text

def infer_openai(model, image: Image.Image, prompt: str, model_name: str):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = model.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content


def main():
    cfg = Config()
    os.makedirs(cfg.OUTPUT_BASE_DIR, exist_ok=True)

    logger.info(f"Loading model: {cfg.MODEL_NAME}")
    model, processor = load_model_and_processor(cfg.MODEL_NAME, cfg.MODEL_CACHE_DIR) #tokenizer,

    logger.info("Loading datasets")
    image_dataset, flag_dataset = load_datasets() ##FROM GEN_MCQ.PY
    continent_dict, subregion_dict = create_geo_dictionaries()
    composer = FlagComposer(flag_dataset)## FROM IMAGE_COMBINATION.PY
    logger.info("Running VQA for {} samples".format(cfg.N_SAMPLES)) 

    for idx in tqdm(range(cfg.N_SAMPLES)):
        if cfg.QUESTION_TYPE == 1:
            try:
                item = image_dataset["train"][idx]
                mcq_dict, answer_key = generate_q1(item,'subregion',continent_dict, subregion_dict)
                if mcq_dict is None:
                    logger.warning(f"Skipping sample {idx} due to mcq_dict being None.")
                    continue
                composed_image = composer.combine_with_main_image(item["image"], mcq_dict) ##LINKS TO COMPOSER.PY

                prompt_q1 = f"""
                "You are an expert on cultures, Given an image and a set of flags, connect the flags to" 
                "the culture depicted given image and how well the flag represents a particular culture." 
                "Examine the cultural context in the following image. From the list of flags below, select" 
                "the flag that least represents the culture depicted in this particular image "
                "Provide your answer as only the letter of your chosen option, followed by the flag name,"
                "(e.g., 'A. Flag Name').\n\n"
                "Flags:\n"
                "{mcq_dict} \n\n"
                "Your Answer:"
                f"""

                if "gemini" in cfg.MODEL_NAME:
                    response = infer_gemini(model, composed_image, prompt_q1)
                elif "gpt" in cfg.MODEL_NAME:
                    response = infer_openai(model, composed_image, prompt_q1, cfg.MODEL_NAME)
                else:
                    response = infer(model, processor, composed_image, prompt_q1)

                result = {
                    "id": idx,
                    "question": prompt_q1,
                    "answer_key": answer_key,
                    "response": response,
                    "options": mcq_dict
                }
                with open(f"{cfg.OUTPUT_BASE_DIR}/sample_{idx}.json", "w") as f:
                    json.dump(result, f, indent=2)

            except Exception as e:
                logger.error(f"Failed on sample {idx}: {e}")
            continue

        elif cfg.QUESTION_TYPE == 2:
            try:
                item = image_dataset["train"][idx]
                mcq_dict, answer_key = generate_q2(item,'subregion',continent_dict, subregion_dict)
                if mcq_dict is None:
                    logger.warning(f"Skipping sample {idx} due to mcq_dict being None.")
                    continue
                composed_image = composer.combine_with_main_image(item["image"], mcq_dict) ##LINKS TO COMPOSER.PY

                prompt_q2 = f"""
                "You are an expert on cultures,Examine the cultural context in the following image."
                From the list of flags below, select the flags that represent general cultural groups depicted in the image"
                "as well as flags that do not represent the culture of the depicted group.\n"
                "Provide your answer as only the letters of your chosen options, followed by the flag names, "
                "and a brief explanation (e.g., 'A, B. Flag Name 1, Flag Name 2 - Reason').\n\n"
                "Flags:\n"
                f"{mcq_dict}\n\n"
                "Your Answer:"
                f"""

                if "gemini" in cfg.MODEL_NAME:
                    response = infer_gemini(model, composed_image, prompt_q2)
                elif "gpt" in cfg.MODEL_NAME:
                    response = infer_openai(model, composed_image, prompt_q2, cfg.MODEL_NAME)
                else:
                    response = infer(model, processor, composed_image, prompt_q2)

                result = {
                    "id": idx,
                    "question": prompt_q2,
                    "answer_key": answer_key,
                    "response": response,
                    "options": mcq_dict
                }
                with open(f"{cfg.OUTPUT_BASE_DIR}/sample_{idx}.json", "w") as f:
                    json.dump(result, f, indent=2)

            except Exception as e:
                logger.error(f"Failed on sample {idx}: {e}")
            continue

if __name__ == "__main__":
    main()