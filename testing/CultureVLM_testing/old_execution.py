import re
import torch
import json
import os
import PIL
from PIL import Image
import time
import logging
import shutil
import gc  # Added for explicit garbage collection
from datetime import datetime, timedelta
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm


# Set GPU device explicitly
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0, 1 - change the gpu device based on your requirement

# Configuration
class Config:
    # Model settings
    #MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    #MODEL_NAME = "Qwen/Qwen-VL-Chat"
    MODEL_NAME = "deepseek-ai/deepseek-vl-7b-chat"

    # model cache directory to store the model so that we dont need download it again and again
    MODEL_CACHE_DIR = "./model_cache"
    
    # Folder structure
    BASE_DIR = "./data"
    INPUT_DIR = f"{BASE_DIR}/input"
    OUTPUT_BASE_DIR = "./output/vqa_q1"
    
    # Prompt template name
    PROMPT_TEMPLATE_NAME = "Flag_VQA" # change the propmt template name as per your requirement 
    
    # Logging
    LOG_LEVEL = logging.INFO
    
    # Processing settings - our default setting is  MAX_NEW_TOKENS = 25, TEMPERATURE = 0.3 and TOP_P = 0.9
    BATCH_SIZE = 1  # Number of questions to run in one batch
    MAX_NEW_TOKENS = 25  # Limit new tokens to prevent over-generation
    TEMPERATURE = 0.3  # Set to 0.0 for deterministic output
    TOP_P = 0.9  # Set to 1.0 for deterministic output

# prompt template - remember to change the propmt as you change the template name from one-shot to zero-shot and vice versa
PROMPT_TEMPLATE_Q1 = """
You are an expert on cultures, Given an image and a set of flags, connect the flags to the culture depicted given image and how well the flag represents a particular culture. 

Examine the cultural context in the following image. From the list of flags below, select the flag that least represents the culture depicted in this particular image "
Provide your answer as only the letter of your chosen option, followed by the flag name, 
"(e.g., 'A. Flag Name').\n\n"
"Flags:\n"
"{options_list_str}\n\n"
"Your Answer:"
"""

PROMPT_TEMPLATE_Q2 = """
You are an expert on cultures, Given an image and a set of flags, connect the flags to the culture depicted given image and how well the flag represents a particular culture. 

Examine the cultural context in the following image. From the list of flags below, select the flag that least represents the culture depicted in this particular image "
Provide your answer as only the letter of your chosen option, followed by the flag name, 
"(e.g., 'A. Flag Name').\n\n"
"Flags:\n"
f"{options_list_str}\n\n"
"Your Answer:"
"""


class RawOutputExtractor:
    def __init__(self, config: Config, domain: str):
        """Initialize the raw output extractor with configuration and domain."""
        self.config = config
        self.domain = domain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create directory structure
        self._setup_directories()
        
        # Set up logging
        self._setup_logging()
        
        # Check if model needs to be redownloaded
        if self.should_redownload_model():
            self._clean_model_cache()
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize metrics
        self.metrics = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_processing_time": 0,
            "images_processed": 0,
            "raw_outputs_generated": 0
        }
        
        # Key marker for identifying model output section
        self.answer_marker = "Your Answer:"
        
    def log_memory_stats(self, stage: str):
        """Log detailed memory statistics."""
        if not torch.cuda.is_available():
            return
            
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        
        self.logger.info(f"Memory at {stage}: Allocated: {allocated:.2f}MB, "
                       f"Max Allocated: {max_allocated:.2f}MB, "
                       f"Reserved: {reserved:.2f}MB")
    
    def should_redownload_model(self) -> bool:
        """Check if the model cache exists and is valid using the correct HF path format."""
        # The actual HuggingFace cache structure uses this format:
        model_cache_path = Path(self.config.MODEL_CACHE_DIR) / f"models--{self.config.MODEL_NAME.replace('/', '--')}"
        
        if not model_cache_path.exists():
            self.logger.info(f"Model cache not found at {model_cache_path}, will download model")
            return True
        
        # Check for essential files
        config_file = model_cache_path / "config.json"
        if not config_file.exists():
            self.logger.info(f"Model cache appears incomplete, will redownload")
            return True
        
        return False
    
    def _clean_model_cache(self) -> None:
        """Remove the existing model cache if it exists."""
        model_cache_path = Path(self.config.MODEL_CACHE_DIR) / self.config.MODEL_NAME.replace('/', '--')
        if model_cache_path.exists():
            self.logger.info(f"Removing existing model cache at {model_cache_path}")
            try:
                shutil.rmtree(model_cache_path)
                self.logger.info(f"Successfully removed model cache")
            except Exception as e:
                self.logger.error(f"Failed to remove model cache: {e}")
    
    def _setup_directories(self):
        """Create the directory structure for outputs."""
        # Create base output directory structure
        self.output_dir = Path(f"{self.config.OUTPUT_BASE_DIR}/{self.config.PROMPT_TEMPLATE_NAME}/{self.config.MODEL_NAME.split('/')[-1]}/{self.domain}")
        
        # Create raw output and logs directories - using same structure as original
        self.raw_output_dir = self.output_dir / "raw_output"
        self.logs_dir = self.output_dir / "logs"
        self.metrics_dir = self.output_dir / "generated_file_metric_logs"
        
        # Create model cache directory if it doesn't exist
        os.makedirs(self.config.MODEL_CACHE_DIR, exist_ok=True)
        
        # Create output directories
        os.makedirs(self.raw_output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        print(f"Created directory structure at {self.output_dir}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger(f"RawOutputExtractor_{self.domain}")
        self.logger.setLevel(self.config.LOG_LEVEL)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.LOG_LEVEL)
        
        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"{self.domain}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.config.LOG_LEVEL)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger initialized. Logging to {log_file}")
        self.logger.info(f"Created directory structure at {self.output_dir}")
    
    def _load_model(self):
        """Load the model and tokenizer from HuggingFace, using cache."""
        self.logger.info(f"Loading model {self.config.MODEL_NAME} from HuggingFace (using cache dir: {self.config.MODEL_CACHE_DIR})")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.MODEL_NAME, 
                cache_dir=self.config.MODEL_CACHE_DIR,
                use_fast=True
            )
            self.logger.info("Tokenizer loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise
        
        # Load configuration with rope_scaling fix
        try:
            config = AutoConfig.from_pretrained(
                self.config.MODEL_NAME,
                cache_dir=self.config.MODEL_CACHE_DIR
            )
            
            # Remove rope_scaling to prevent issues
            if hasattr(config, 'rope_scaling'):
                delattr(config, 'rope_scaling')
                self.logger.info("Removed rope_scaling attribute from config")
            
            # Load model with modified config and float16 precision for speed
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME, 
                config=config,
                cache_dir=self.config.MODEL_CACHE_DIR,
                torch_dtype=torch.float16,
                device_map="auto",
                ignore_mismatched_sizes=True
            )
            self.logger.info(f"Model loaded successfully. Using device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model with modified config: {e}")
            
            # Fallback approach using manual config editing
            try:
                import json
                import tempfile
                from huggingface_hub import hf_hub_download
                
                # Download config file directly
                config_path = hf_hub_download(
                    repo_id=self.config.MODEL_NAME,
                    filename="config.json",
                    cache_dir=self.config.MODEL_CACHE_DIR
                )
                
                # Read and modify config
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Remove problematic field
                if 'rope_scaling' in config_dict:
                    del config_dict['rope_scaling']
                    self.logger.info("Removed rope_scaling from config_dict")
                
                # Create temporary file with modified config
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_f:
                    json.dump(config_dict, temp_f)
                    temp_config_path = temp_f.name
                
                # Use modified config
                config = AutoConfig.from_pretrained(temp_config_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.MODEL_NAME,
                    config=config,
                    cache_dir=self.config.MODEL_CACHE_DIR,
                    torch_dtype=torch.float16,
                    device_map="auto", 
                    ignore_mismatched_sizes=True
                )
                self.logger.info(f"Model loaded successfully with fallback approach.")
                
                # Clean up
                os.unlink(temp_config_path)
                
            except Exception as e2:
                self.logger.error(f"All attempts to load model failed: {e2}")
                raise RuntimeError(f"Could not load model after multiple attempts.")
    
    def _unload_model(self):
        """Unload model and free GPU memory."""
        self.logger.info("Unloading model and freeing GPU memory")
        
        # Delete model and tokenizer references
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.info("CUDA cache cleared")
    
    def generate_model_output(self,question_type: int, image: PIL.Image.Image ,output_list: str) -> Tuple[str, Dict[str, int]]:
        
        """
        Generate raw model output with improved stopping logic to prevent over-generation.
        Returns only the LLM generated response (not the prompt) and metrics.
        Added tensor detachment to prevent memory leaks.
        """
        start_time = time.time()
        
        try:
            # Prepare prompt
            if question_type == 1:
                prompt = PROMPT_TEMPLATE_Q1.replace("{output_list}", output_list)
            if question_type == 2:
                prompt = PROMPT_TEMPLATE_Q2.replace("{output_list}", output_list)
            else:
                print("Please chose Question 1 or 2")

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            input_token_count = len(inputs.input_ids[0])
            
            # Generate response with improved stopping logic
            try:
                with torch.no_grad():
                    # Use max_new_tokens instead of max_length for better control
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.MAX_NEW_TOKENS,
                        temperature=self.config.TEMPERATURE,
                        top_p=self.config.TOP_P,
                        do_sample=True,               
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,  # Explicitly set EOS token
                        early_stopping=True,  # Enable early stopping
                        repetition_penalty=1.2  # Add repetition penalty to discourage repeating the JSON
                    )
                    
                    # Get only the generated tokens (not including the prompt)
                    # Ensure proper detachment with clone
                    generated_tokens = output[0][len(inputs.input_ids[0]):].clone().detach()
                    
                    # Decode only the generated part (excluding the prompt)
                    model_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    output_token_count = len(generated_tokens)
                    
                    # Post-process to truncate at reasonable ending point
                    # Find the last closing brace that might signal the end of the JSON
                    last_closing_brace = model_output.rfind("}")
                    
                    # Look for indicators of over-generation after the last closing brace
                    if last_closing_brace > 0:
                        # Truncate at the last closing brace + 1
                        model_output = model_output[:last_closing_brace+1]
                        
                        # Check if we need to fix unclosed JSON
                        open_braces = model_output.count('{')
                        close_braces = model_output.count('}')
                        
                        # If there are more open braces than close braces, add missing close braces
                        if open_braces > close_braces:
                            model_output += '}' * (open_braces - close_braces)
                    
                    # Free memory from generated output - explicit deletion
                    del output
                    del generated_tokens
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure CUDA operations complete
                    
            except RuntimeError as e:
                # Handle CUDA out of memory
                if "CUDA out of memory" in str(e):
                    self.logger.warning("CUDA out of memory during generation - trying with reduced settings")
                    
                    # Clear cache and retry with reduced settings
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    with torch.no_grad():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=400,  # Reduced
                            temperature=0.0,     # More deterministic
                            do_sample=False,     # No sampling
                            num_beams=1,         # Simple search
                            repetition_penalty=1.0
                        )
                    
                    # Get only the generated tokens (not including the prompt)
                    generated_tokens = output[0][len(inputs.input_ids[0]):].clone().detach()
                    
                    # Decode only the generated part (excluding the prompt)
                    model_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    output_token_count = len(generated_tokens)
                    
                    # Free memory - explicit deletion
                    del output
                    del generated_tokens
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                else:
                    raise
            
            # Clear inputs from CUDA memory - explicit deletion
            del inputs
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            # Create default output for error cases
            model_output = f"Error during generation: {str(e)}"
            output_token_count = 0 #len(abstract) // 4  # Rough estimate
            input_token_count = 0 #len(abstract) // 4   # Rough estimate
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Metrics 
        metrics = {
            "input_tokens": int(input_token_count) if 'input_token_count' in locals() else 0,
            "output_tokens": int(output_token_count) if 'output_token_count' in locals() else 0,
            "processing_time": processing_time
        }
        
        # Return only the model output with metrics
        return model_output, metrics
    
    def process_file(self, input_file: str) -> Dict:
        """Process a single JSON file containing paper abstracts and return file metrics."""
        file_start_time = time.time()
        file_basename = os.path.basename(input_file)
        
        self.logger.info(f"Processing file: {file_basename}")
        self.log_memory_stats(f"Before processing file {file_basename}")
        
        # Load the JSON file
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                file_content = f.read()
                
            # Handle different JSON formats
            papers = []
            
            # Try parsing as array
            if file_content.strip().startswith('['):
                try:
                    papers = json.loads(file_content)
                    self.logger.info(f"Found {len(papers)} papers in array format")
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse array JSON")
                    return {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "processing_time": 0,
                        "papers_count": 0,
                        "events_count": 0,
                        "raw_outputs_generated": 0,
                        "total_time": time.time() - file_start_time
                    }
            else:
                # Try parsing as a single object
                try:
                    paper = json.loads(file_content)
                    papers = [paper]
                    self.logger.info(f"Found 1 paper in single object format")
                except json.JSONDecodeError:
                    # Try parsing as newline-delimited JSON
                    papers = []
                    for line in file_content.strip().split('\n'):
                        if line.strip():
                            try:
                                paper = json.loads(line)
                                papers.append(paper)
                            except json.JSONDecodeError:
                                pass
                    self.logger.info(f"Found {len(papers)} papers in newline-delimited format")
        except Exception as e:
            self.logger.error(f"Error loading file {file_basename}: {e}")
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "processing_time": 0,
                "papers_count": 0,
                "events_count": 0,
                "raw_outputs_generated": 0,
                "total_time": time.time() - file_start_time
            }
        
        # Initialize file metrics
        file_metrics = {
            "input_tokens": 0,
            "output_tokens": 0,
            "processing_time": 0,
            "papers_count": len(papers),
            "events_count": 0,
            "raw_outputs_generated": 0
        }
        
        # Create a tqdm progress bar for papers in this file
        with tqdm(total=len(papers), desc=f"File: {file_basename}", unit="paper") as pbar:
            # Process each paper
            for i, paper in enumerate(papers):
                try:
                    paper_code = paper.get("paper_code")
                    if not paper_code:
                        self.logger.warning(f"Paper {i} has no paper_code, skipping")
                        pbar.update(1)
                        continue
                        
                    abstract = paper.get("abstract", "")
                    
                    # Include paper progress in the progress bar description
                    pbar.set_description(f"File: {file_basename} | Paper: {paper_code} ({i+1}/{len(papers)})")
                    
                    # Extract events/sections from the original data if available
                    events = paper.get("events", [])
                    
                    if not events:
                        # If no events, process the entire abstract
                        self.logger.info(f"No events found, processing entire abstract")
                        file_metrics["events_count"] += 1
                        self.metrics["events_processed"] += 1
                        
                        try:
                            # Generate complete raw output
                            raw_output, metrics = self.generate_model_output(abstract)
                            
                            # Save the complete raw output exactly as received
                            raw_txt_file_path = self.raw_output_dir / f"{paper_code}_full.txt"
                            with open(raw_txt_file_path, "w", encoding="utf-8") as raw_txt_file:
                                raw_txt_file.write(raw_output)
                                
                            # Increment raw outputs counter
                            file_metrics["raw_outputs_generated"] += 1
                            
                            # Update metrics
                            file_metrics["input_tokens"] += metrics["input_tokens"]
                            file_metrics["output_tokens"] += metrics["output_tokens"]
                            file_metrics["processing_time"] += metrics["processing_time"]
                        except Exception as e:
                            self.logger.error(f"Error processing abstract for paper {paper_code}: {str(e)}")
                            import traceback
                            self.logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    else:
                        # Process each event separately with nested progress bar
                        file_metrics["events_count"] += len(events)
                        self.metrics["events_processed"] += len(events)
                        
                        # Create a nested tqdm progress bar for events within this paper
                        with tqdm(total=len(events), desc=f"  Events in {paper_code}", unit="event", leave=False) as event_pbar:
                            for event_index, event in enumerate(events):
                                event_text = event.get("Text", "")
                                
                                # Find event type
                                event_type = None
                                try:
                                    for key in event.keys():
                                        if key not in ["Text", "Main Action", "Arguments", "Summary"]:
                                            event_type = key
                                            break
                                except Exception as e:
                                    self.logger.error(f"Error finding event type: {e}")
                                    event_type = None
                                
                                # Update event progress bar description with event type
                                if event_type:
                                    event_pbar.set_description(f"  Events in {paper_code} | Type: {event_type} ({event_index+1}/{len(events)})")
                                else:
                                    event_pbar.set_description(f"  Events in {paper_code} | Event: {event_index+1}/{len(events)}")
                                
                                if not event_text:
                                    self.logger.warning(f"Empty event text for paper {paper_code}, event {event_index}")
                                    event_pbar.update(1)
                                    continue
                                
                                try:
                                    # Generate raw model output and save as-is
                                    raw_output, metrics = self.generate_model_output(event_text)
                                    
                                    raw_txt_file_path = self.raw_output_dir / f"{paper_code}_event_{event_index}.txt"
                                    with open(raw_txt_file_path, "w", encoding="utf-8") as raw_txt_file:
                                        raw_txt_file.write(raw_output)
                                        
                                    # Increment raw outputs counter
                                    file_metrics["raw_outputs_generated"] += 1
                                    
                                    # Update metrics
                                    file_metrics["input_tokens"] += metrics["input_tokens"]
                                    file_metrics["output_tokens"] += metrics["output_tokens"]
                                    file_metrics["processing_time"] += metrics["processing_time"]
                                
                                except Exception as e:
                                    self.logger.error(f"Error processing event {event_index} for paper {paper_code}: {str(e)}")
                                    import traceback
                                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                                
                                # Clear CUDA cache after each event to prevent memory buildup
                                torch.cuda.empty_cache()
                                
                                # Update event progress bar
                                event_pbar.update(1)
                    
                    self.metrics["papers_processed"] += 1
                    
                    # Update paper progress bar
                    pbar.update(1)
                    
                    # Calculate and display estimated time remaining for file
                    if i > 0:  # Need at least one paper to estimate
                        avg_paper_time = file_metrics["processing_time"] / (i + 1)
                        remaining_papers = len(papers) - (i + 1)
                        est_remaining_time = avg_paper_time * remaining_papers
                        est_completion_time = datetime.now() + timedelta(seconds=est_remaining_time)
                        
                        # Update progress bar postfix with time estimates
                        pbar.set_postfix({
                            "Avg paper": f"{avg_paper_time:.2f}s", 
                            "ETA": est_completion_time.strftime("%H:%M:%S"),
                            "Remaining": str(timedelta(seconds=int(est_remaining_time)))
                        })
                
                except Exception as e:
                    self.logger.error(f"Error processing paper {paper.get('paper_code', 'unknown')}: {str(e)}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    pbar.update(1)
                    continue
                
                # Performs cleanup after each paper to prevent memory accumulation
                gc.collect()
                torch.cuda.empty_cache()
        
        # Save file-specific metrics
        file_metrics["total_time"] = time.time() - file_start_time
        if file_metrics["papers_count"] > 0:
            file_metrics["average_paper_time"] = file_metrics["processing_time"] / file_metrics["papers_count"]
        else:
            file_metrics["average_paper_time"] = 0
        
        metrics_file = self.metrics_dir / f"{os.path.basename(input_file).replace('.json', '')}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(file_metrics, f, indent=2)
        
        self.logger.info(f"Completed processing file: {file_basename}")
        self.logger.info(f"File metrics: {file_metrics}")
        
        # Update global metrics
        self.metrics["files_processed"] += 1
        self.metrics["total_input_tokens"] += file_metrics["input_tokens"]
        self.metrics["total_output_tokens"] += file_metrics["output_tokens"]
        self.metrics["total_processing_time"] += file_metrics["processing_time"]
        self.metrics["raw_outputs_generated"] += file_metrics["raw_outputs_generated"]
        
        # Final memory cleanup after file processing
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all CUDA operations are completed
        
        self.log_memory_stats(f"After processing file {file_basename}")
        
        return file_metrics

    def process_domain(self) -> None:
        """Process all files in the domain directory with tqdm for tracking overall progress."""
        domain_start_time = time.time()
        
        # Clear GPU cache at the beginning of a new domain
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        self.log_memory_stats(f"Starting domain {self.domain}")
        
        # Get input directory for the domain
        input_dir = f"{self.config.INPUT_DIR}/{self.domain}"
        
        if not os.path.exists(input_dir):
            self.logger.error(f"Input directory not found: {input_dir}")
            return
        
        # Get all JSON files in the input directory
        json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        
        if not json_files:
            self.logger.warning(f"No JSON files found in {input_dir}")
            return
        
        self.logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
        
        # Initialize file metrics history to calculate overall statistics
        file_metrics_history = []
        
        # Process each file with tqdm for progress tracking
        with tqdm(total=len(json_files), desc=f"Domain: {self.domain}", unit="file") as domain_pbar:
            for file_idx, file in enumerate(json_files):
                input_file = os.path.join(input_dir, file)
                
                # Process file and collect metrics
                file_metrics = self.process_file(input_file)
                file_metrics_history.append(file_metrics)
                
                # Update domain progress bar
                domain_pbar.update(1)
                
                # Calculates and shows overall progress statistics
                if file_idx > 0:
                    # Calculates average processing time per file
                    avg_file_time = sum(m["total_time"] for m in file_metrics_history) / len(file_metrics_history)
                    
                    # Estimates remaining time
                    remaining_files = len(json_files) - (file_idx + 1)
                    est_remaining_time = avg_file_time * remaining_files
                    est_completion_time = datetime.now() + timedelta(seconds=est_remaining_time)
                    
                    # Updates progress bar with overall statistics
                    domain_pbar.set_postfix({
                        "Avg file": f"{avg_file_time:.1f}s",
                        "ETA": est_completion_time.strftime("%H:%M:%S"),
                        "Remaining": str(timedelta(seconds=int(est_remaining_time)))
                    })
                    
                # Cleanup after each file
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Calculate and save overall metrics
        self.metrics["total_time"] = time.time() - domain_start_time
        
        # Avoid division by zero
        if self.metrics["papers_processed"] > 0:
            self.metrics["average_paper_time"] = self.metrics["total_processing_time"] / self.metrics["papers_processed"]
            self.metrics["average_input_tokens"] = self.metrics["total_input_tokens"] / self.metrics["papers_processed"]
            self.metrics["average_output_tokens"] = self.metrics["total_output_tokens"] / self.metrics["papers_processed"]
        else:
            self.metrics["average_paper_time"] = 0
            self.metrics["average_input_tokens"] = 0
            self.metrics["average_output_tokens"] = 0
        
        # Save overall metrics
        overall_metrics_file = self.metrics_dir / "overall_metrics.json"
        with open(overall_metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Completed processing domain: {self.domain}")
        self.logger.info(f"Overall metrics: {self.metrics}")
        
        # Final cleanup at the end of domain processing
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Reset max memory allocated to give accurate readings for next domain
        torch.cuda.reset_peak_memory_stats()
        self.log_memory_stats(f"Completed domain {self.domain}")
        
        # Print final summary to console
        total_papers = self.metrics["papers_processed"]
        total_events = self.metrics["events_processed"] 
        total_outputs = self.metrics["raw_outputs_generated"]
        total_time = str(timedelta(seconds=int(self.metrics["total_time"])))
        
        print("\n" + "="*80)
        print(f"PROCESSING SUMMARY FOR DOMAIN: {self.domain}")
        print("="*80)
        print(f"Total files processed:    {self.metrics['files_processed']}")
        print(f"Total papers processed:   {total_papers}")
        print(f"Total events processed:   {total_events}")
        print(f"Total outputs generated:  {total_outputs}")
        print(f"Total processing time:    {total_time}")
        if total_papers > 0:
            print(f"Average time per paper:   {self.metrics['average_paper_time']:.2f}s")
        print("="*80)

        # Unload model and free memory before exiting
        self._unload_model()