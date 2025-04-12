"""
Main application for PCB defect detection and IPC-A-610F query system.
This is the entry point for using the PCB Assistant.
"""

import os
import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Union, Tuple

# Import custom modules
from model_manager import ModelManager
from pcb_assistant import PCBAssistant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pcb_assistant.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_args() -> argparse.Namespace:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description='PCB Defect Detection and IPC-A-610F Query System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Common arguments
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['text', 'image', 'reference'],
                        help='Operation mode: text for IPC-A-610F queries, image for PCB analysis, reference for finding example images')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, 
                        default='llama-3.2-vision-11b.Q4_K_M.gguf',
                        help='Path to the model file or model name')
    parser.add_argument('--model_type', type=str, 
                        default='llama',
                        choices=['llama', 'qwen', 'deepseek'],
                        help='Type of model to use')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for inference')
    parser.add_argument('--gpu_layers', type=int, default=-1,
                        help='Number of layers to offload to GPU (-1 for auto)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for generation')
    
    # PDF and vector DB arguments
    parser.add_argument('--pdf_path', type=str,
                        default='./IPC-A-610F.pdf',
                        help='Path to the IPC-A-610F PDF file')
    parser.add_argument('--vector_db_path', type=str,
                        default='./vector_db',
                        help='Path to store the vector database')
    parser.add_argument('--cache_dir', type=str,
                        default='./cache',
                        help='Directory to store cached data')
    parser.add_argument('--rebuild_index', action='store_true',
                        help='Force rebuilding the vector index')
    
    # Mode-specific arguments
    parser.add_argument('--query', type=str,
                        help='Query text for IPC-A-610F standard (for text mode)')
    parser.add_argument('--image_path', type=str,
                        help='Path to PCB image for analysis (for image mode)')
    parser.add_argument('--defect_type', type=str,
                        help='Defect type to find reference images for (for reference mode)')
    parser.add_argument('--output', type=str,
                        help='Output file to save results (optional)')
    
    return parser.parse_args()

def print_system_info():
    """Print system information for debugging."""
    import platform
    import torch
    
    logger.info("=== System Information ===")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"OS: {platform.system()} {platform.release()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("CUDA available: No")
    
    logger.info("==========================")

def save_results(output_path: str, results: str):
    """Save results to file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(results)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    """Main function to run the PCB assistant."""
    # Parse command-line arguments
    args = setup_args()
    
    # Print system information
    print_system_info()
    
    try:
        # Initialize model manager
        logger.info(f"Initializing model manager with {args.model_path}")
        model_manager = ModelManager(
            model_name_or_path=args.model_path,
            model_type=args.model_type,
            use_gpu=args.use_gpu,
            gpu_layers=args.gpu_layers,
            temperature=args.temperature
        )
        
        # Initialize PCB assistant
        logger.info("Initializing PCB assistant")
        assistant = PCBAssistant(
            ipc_pdf_path=args.pdf_path,
            model_manager=model_manager,
            vector_db_path=args.vector_db_path,
            cache_dir=args.cache_dir,
            rebuild_index=args.rebuild_index
        )
        
        # Process based on mode
        results = ""
        
        if args.mode == 'text':
            if not args.query:
                logger.error("Query text is required for text mode")
                print("Error: Please provide a query using --query")
                return
            
            logger.info(f"Processing query: {args.query}")
            start_time = time.time()
            results = assistant.answer_ipc_question(args.query)
            end_time = time.time()
            
            logger.info(f"Query processed in {end_time - start_time:.2f} seconds")
            print("\n" + "="*80 + "\n")
            print(results)
            print("\n" + "="*80)
            
        elif args.mode == 'image':
            if not args.image_path or not os.path.exists(args.image_path):
                logger.error("Valid image path is required for image mode")
                print("Error: Please provide a valid image path using --image_path")
                return
            
            logger.info(f"Analyzing image: {args.image_path}")
            start_time = time.time()
            results = assistant.analyze_pcb_image(args.image_path)
            end_time = time.time()
            
            logger.info(f"Image analyzed in {end_time - start_time:.2f} seconds")
            print("\n" + "="*80 + "\n")
            print(results)
            print("\n" + "="*80)
            
        elif args.mode == 'reference':
            if not args.defect_type:
                logger.error("Defect type is required for reference mode")
                print("Error: Please provide a defect type using --defect_type")
                return
            
            logger.info(f"Finding reference images for defect type: {args.defect_type}")
            start_time = time.time()
            reference_images = assistant.suggest_reference_images(args.defect_type)
            end_time = time.time()
            
            logger.info(f"Reference images found in {end_time - start_time:.2f} seconds")
            
            if not reference_images:
                print(f"No reference images found for defect type: {args.defect_type}")
            else:
                print(f"Found {len(reference_images)} reference images for defect type: {args.defect_type}")
                for i, img_info in enumerate(reference_images):
                    print(f"\nReference Image {i+1}")
                    print(f"Page: {img_info['page']}")
                    print(f"Caption: {img_info['caption']}")
                    print(f"Path: {img_info['image_path']}")
                    
                    # Format for output file if needed
                    results += f"Reference Image {i+1}\n"
                    results += f"Page: {img_info['page']}\n"
                    results += f"Caption: {img_info['caption']}\n"
                    results += f"Path: {img_info['image_path']}\n\n"
        
        # Save results if output file is specified
        if args.output and results:
            save_results(args.output, results)
        
        # Print memory usage
        memory_stats = model_manager.get_memory_usage()
        logger.info(f"Memory usage: {memory_stats}")
        
    except Exception as e:
        logger.error(f"Error running PCB assistant: {e}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())