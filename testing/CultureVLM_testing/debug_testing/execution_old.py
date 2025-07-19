import os
import torch
import json
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

from PIL import Image

from gen_mcq import load_datasets, generate_q1,generate_q2, create_geo_dictionaries
from image_combination_refactored import FlagComposer
from image_join import FlagComposerN

# Set GPU device explicitly
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Change as needed

class Config:
    #MODEL_NAME = "deepseek-ai/deepseek-vl-7b-chat"
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    #MODEL_NAME = "Qwen/Qwen-VL-Chat"
    MODEL_CACHE_DIR = "./model_cache"
    QUESTION_TYPE = 1
    BASE_DIR = "./data"
    OUTPUT_BASE_DIR = "./output/vqa"
    N_SAMPLES = 1

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_processor(model_name, cache_dir):
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    #tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    if "Qwen" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir, trust_remote_code=True)
    elif "llava" in model_name or "deepseek-ai" in model_name:
        model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto", cache_dir=cache_dir, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    #model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir, trust_remote_code=True)
    return model, processor #, tokenizer,

# def infer(model, processor, image: Image.Image, prompt: str):
#     inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device, torch.float16)
#     output = model.generate(**inputs, max_new_tokens=50)
#     decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
#     return decoded
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
def infer(model, processor, image: Image.Image, prompt: str):
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Format prompt for vision-language alignment
    formatted_prompt = "<image>\n" + prompt.strip()

    # Process input
    inputs = processor(text=formatted_prompt,images=image,return_tensors="pt").to(model.device)
    
    # Inference
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    return decoded

def main():
    cfg = Config()
    os.makedirs(cfg.OUTPUT_BASE_DIR, exist_ok=True)

    logger.info(f"Loading model: {cfg.MODEL_NAME}")
    model, processor = load_model_and_processor(cfg.MODEL_NAME, cfg.MODEL_CACHE_DIR) #tokenizer,

    logger.info("Loading datasets")
    image_dataset, flag_dataset = load_datasets() ##FROM GEN_MCQ.PY
    continent_dict, subregion_dict = create_geo_dictionaries()
    composer = FlagComposerN(flag_dataset)## FROM IMAGE_COMBINATION.PY
    logger.info("Running VQA for {} samples".format(cfg.N_SAMPLES)) 

    for idx in tqdm(range(cfg.N_SAMPLES)):
        if cfg.QUESTION_TYPE == 1:
            try:
                item = image_dataset["train"][idx]
                mcq_dict, answer_key = generate_q1(item,'subregion',continent_dict, subregion_dict)
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
