# PCB Assistant

A PCB defect detection and IPC-A-610F standard query system powered by multimodal AI models. This tool helps PCB assembly professionals analyze circuit boards for defects and access relevant IPC standard information.

## Features

- **PCB Defect Analysis**: Analyze PCB images to identify potential defects
- **IPC-A-610F Standard Queries**: Search and retrieve information from the IPC-A-610F manual
- **Reference Image Lookup**: Find relevant reference images from the IPC standard for specific defect types
- **Multimodal AI Support**: Works with various LLM models (Llama.cpp, Qwen, DeepSeek)
- **GPU Acceleration**: Optional GPU support for faster inference

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- IPC-A-610F PDF manual

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pcb_assistant.git
   cd pcb_assistant
   ```

2. Create a virtual environment:
   ```bash
   python -m venv pcb
   source pcb/bin/activate   # On Windows: pcb\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the IPC-A-610F PDF in the project directory or specify its path when running the program.

5. Download a compatible model. Place models in the `models/` directory.

## Usage

The system operates in three modes:

### Text Mode (IPC-A-610F Queries)

Query information from the IPC-A-610F standard:

```bash
python main.py --mode text --query "What are the acceptance criteria for solder bridges?" --model_path models/Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf --use_gpu
```

### Image Mode (PCB Analysis) (WORK IN PROGRESS)

Analyze a PCB image for defects:

```bash
python main.py --mode image --image_path /path/to/your/pcb_image.jpg --model_path models/Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf --use_gpu
```

### Reference Mode (Find Example Images)

Find reference images for specific defect types:

```bash
python main.py --mode reference --defect_type "solder bridge" --model_path models/Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf --use_gpu
```

### Additional Options

- `--model_path`: Path to the model file (default: llama-3.2-vision-11b.Q4_K_M.gguf)
- `--model_type`: Model type (llama, qwen, deepseek) (default: llama)
- `--use_gpu`: Enable GPU acceleration
- `--gpu_layers`: Number of layers to offload to GPU (-1 for auto)
- `--temperature`: Generation temperature (default: 0.1)
- `--vector_db_path`: Path to store the vector database (default: ./vector_db)
- `--cache_dir`: Directory to store cached data (default: ./cache)
- `--rebuild_index`: Force rebuilding the vector index
- `--output`: Output file to save results

## First-Time Setup

On first run, the system will:
1. Process the IPC-A-610F PDF
2. Extract text and images
3. Build a vector database for efficient searching

This may take several minutes, but it's a one-time operation. Subsequent runs will use the cached data and vector database.

## System Architecture

- `main.py`: Entry point and command-line interface
- `pcb_assistant.py`: Core RAG system for the IPC-A-610F standard
- `model_manager.py`: Handles model loading and inference
- `vector_db/`: Stores vector embeddings for efficient searching
- `cache/`: Stores extracted images and processed data

## Troubleshooting

- **GPU Memory Issues**: If you encounter CUDA out of memory errors, try:
  - Using a quantized model (Q4_K_M, Q6_K)
  - Reducing the number of GPU layers with `--gpu_layers`
  - Falling back to CPU with `--use_gpu` omitted

- **Model Loading Errors**: Ensure you have the correct model type specified with `--model_type`

- **PDF Processing Errors**: Make sure PyMuPDF is installed correctly or try reinstalling with:
  ```bash
  pip uninstall PyMuPDF fitz
  pip install PyMuPDF
  ```
