"""
Model manager for PCB defect detection and IPC-A-610F question answering.
This module handles loading and interacting with the multimodal LLM.
"""

import os
import torch
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Choose model implementation based on availability
try:
    from llama_cpp import Llama
    USE_LLAMA_CPP = True
    logger.info("Using llama.cpp for model inference")
except ImportError:
    USE_LLAMA_CPP = False
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        from transformers.generation import GenerationConfig
        logger.info("Using Hugging Face Transformers for model inference")
    except ImportError:
        logger.error("Neither llama.cpp nor transformers is available. Please install one of them.")
        raise ImportError("No suitable model backend found")

class ModelManager:
    """Manages the multimodal LLM for both vision and text tasks."""
    
    def __init__(
        self,
        model_name_or_path: str = "Llama-3.2-Vision-11B-GGUF", 
        model_type: str = "llama",
        use_gpu: bool = True,
        gpu_layers: int = -1,  # -1 means load as many as possible to GPU
        context_size: int = 4096,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        Initialize the model manager.
        
        Args:
            model_name_or_path: Path to the model or model identifier
            model_type: Type of model (llama, qwen, deepseek)
            use_gpu: Whether to use GPU for inference
            gpu_layers: Number of layers to offload to GPU (-1 for auto)
            context_size: Context window size
            temperature: Generation temperature
            max_tokens: Maximum number of tokens to generate
        """
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type.lower()
        self.use_gpu = use_gpu
        self.gpu_layers = gpu_layers
        self.context_size = context_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Validate if GPU is available when requested
        if self.use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Falling back to CPU.")
            self.use_gpu = False
        
        # Load the appropriate model based on type and availability
        self._load_model()
        
    def _load_model(self):
        """Load the appropriate model based on configuration."""
        
        logger.info(f"Loading model: {self.model_name_or_path}")
        
        if USE_LLAMA_CPP:
            # Use llama.cpp for optimized inference
            n_gpu_layers = self.gpu_layers if self.gpu_layers != -1 else 100  # Try to offload as many as possible
            
            self.model = Llama(
                model_path=self.model_name_or_path,
                n_ctx=self.context_size,
                n_gpu_layers=n_gpu_layers if self.use_gpu else 0,
                verbose=False
            )
            logger.info(f"Loaded {self.model_name_or_path} with llama.cpp, GPU layers: {n_gpu_layers if self.use_gpu else 0}")
            
        else:
            # Use Hugging Face transformers
            device_map = "auto" if self.use_gpu else "cpu"
            
            if self.model_type in ["llama", "llama2", "llama3"]:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
                self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map=device_map,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                    trust_remote_code=True
                )
                logger.info(f"Loaded {self.model_name_or_path} with HF transformers, device: {device_map}")
                
            elif self.model_type == "qwen":
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
                self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path, 
                    device_map=device_map,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                    trust_remote_code=True
                )
                logger.info(f"Loaded Qwen model {self.model_name_or_path}, device: {device_map}")
                
            elif self.model_type == "deepseek":
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
                self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map=device_map,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                    trust_remote_code=True
                )
                logger.info(f"Loaded DeepSeek model {self.model_name_or_path}, device: {device_map}")
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def process_text(self, prompt: str) -> str:
        """
        Process a text prompt using the loaded model.
        
        Args:
            prompt: The text prompt to process
            
        Returns:
            The model's response
        """
        logger.info("Processing text prompt")
        
        if USE_LLAMA_CPP:
            # Generate with llama.cpp
            response = self.model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                echo=False
            )
            return response["choices"][0]["text"].strip()
        else:
            # Generate with Hugging Face
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.use_gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            generation_config = GenerationConfig(
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,
            )
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    def process_image(self, image_path: str, prompt: str) -> str:
        """
        Process an image with a text prompt using the multimodal model.
        
        Args:
            image_path: Path to the image file
            prompt: The text prompt to process along with the image
            
        Returns:
            The model's response
        """
        logger.info(f"Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
        
        if USE_LLAMA_CPP:
            # For llama.cpp with multimodal support
            try:
                from PIL import Image
                import base64
                from io import BytesIO
                
                # Load and convert image to base64
                image = Image.open(image_path)
                
                # Convert to RGB if the image is in RGBA mode
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Create prompt with image
                vision_prompt = f"<image>\n{img_str}\n</image>\n{prompt}"
                
                # Generate with llama.cpp
                response = self.model(
                    vision_prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    echo=False
                )
                return response["choices"][0]["text"].strip()
            
            except Exception as e:
                logger.error(f"Error processing image with llama.cpp: {e}")
                return f"Error processing image: {str(e)}"
        
        else:
            # For Hugging Face transformers with multimodal support
            try:
                from PIL import Image
                
                # Load image
                image = Image.open(image_path)
                
                # Process based on model type
                if self.model_type in ["llama", "llama2", "llama3"]:
                    inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                elif self.model_type in ["qwen", "deepseek"]:
                    inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                else:
                    raise ValueError(f"Unsupported model type for image processing: {self.model_type}")
                
                if self.use_gpu:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                generation_config = GenerationConfig(
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.temperature > 0,
                )
                
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                
                # Process output based on model type
                if self.model_type in ["llama", "llama2", "llama3", "qwen", "deepseek"]:
                    # Get only the newly generated tokens
                    result = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                else:
                    # Fallback decoding
                    result = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    if prompt in result:
                        result = result[len(prompt):].strip()
                
                return result
            
            except Exception as e:
                logger.error(f"Error processing image with transformers: {e}")
                return f"Error processing image: {str(e)}"

    def get_memory_usage(self) -> Dict[str, float]:
        """Return current GPU and system memory usage."""
        memory_stats = {}
        
        if torch.cuda.is_available():
            # Get GPU memory statistics
            memory_stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            memory_stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            memory_stats["gpu_max_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        try:
            import psutil
            # Get system memory statistics
            memory_stats["ram_used_gb"] = psutil.virtual_memory().used / 1e9
            memory_stats["ram_total_gb"] = psutil.virtual_memory().total / 1e9
        except ImportError:
            pass
            
        return memory_stats