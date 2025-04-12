"""
PCB Assistant module for defect detection and IPC-A-610F standard querying.
This module handles the RAG system for the IPC-A-610F manual and PCB defect detection.
"""

import os
import re
import json
import glob
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import necessary libraries, with fallbacks where possible
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
    logger.info("Using LangChain for vector search")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available, falling back to simple text search")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    logger.info("Using PyMuPDF for PDF processing")
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available, PDF processing will be limited")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available, image handling will be limited")

class PCBAssistant:
    """PCB defect detection and IPC-A-610F standard query assistant."""
    
    def __init__(
        self, 
        ipc_pdf_path: str,
        model_manager,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_db_path: str = "./vector_db",
        cache_dir: str = "./cache",
        rebuild_index: bool = False
    ):
        """
        Initialize the PCB assistant.
        
        Args:
            ipc_pdf_path: Path to the IPC-A-610F PDF manual
            model_manager: Instance of ModelManager class
            embeddings_model: Hugging Face embeddings model to use
            chunk_size: Size of text chunks for the vector database
            chunk_overlap: Overlap between text chunks
            vector_db_path: Path to store the vector database
            cache_dir: Directory to store cached data
            rebuild_index: Whether to rebuild the vector index
        """
        self.ipc_pdf_path = ipc_pdf_path
        self.model_manager = model_manager
        self.embeddings_model = embeddings_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db_path = vector_db_path
        self.cache_dir = cache_dir
        self.rebuild_index = rebuild_index
        
        # Maps page numbers to page content
        self.page_content = {}
        # Maps page numbers to extracted images
        self.page_images = {}
        # Maps figure or table names to page numbers
        self.reference_map = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize vector store and PDF text extraction
        self._initialize_rag_system()
    
    def _extract_pdf_text_and_images(self) -> Tuple[List[Dict[str, str]], Dict[int, List[str]]]:
        """
        Extract text and images from the IPC-A-610F PDF.
        
        Returns:
            Tuple containing a list of document chunks and a mapping of page numbers to image paths
        """
        logger.info(f"Extracting text and images from {self.ipc_pdf_path}")
        
        if not os.path.exists(self.ipc_pdf_path):
            logger.error(f"PDF file not found: {self.ipc_pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {self.ipc_pdf_path}")
        
        # Create cache directory for images if it doesn't exist
        image_cache_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_cache_dir, exist_ok=True)
        
        documents = []
        page_images = {}
        
        # Check if PyMuPDF is available for detailed extraction
        if PYMUPDF_AVAILABLE:
            pdf_document = fitz.open(self.ipc_pdf_path)
            
            # Extract text and images page by page
            for page_num, page in enumerate(pdf_document):
                # Extract text
                text = page.get_text()
                self.page_content[page_num + 1] = text
                
                # Extract images
                image_list = []
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save the image to cache
                    image_filename = f"page_{page_num+1}_img_{img_index}.{base_image['ext']}"
                    image_path = os.path.join(image_cache_dir, image_filename)
                    
                    with open(image_path, "wb") as image_file:
                        image_file.write(image_bytes)
                    
                    image_list.append(image_path)
                
                if image_list:
                    page_images[page_num + 1] = image_list
                
                # Extract figure and table references
                # Look for patterns like "Figure 5-1" or "Table 7-3"
                figure_pattern = r'(Figure|Table)\s+(\d+-\d+)'
                for match in re.finditer(figure_pattern, text):
                    ref_type, ref_num = match.groups()
                    ref_id = f"{ref_type} {ref_num}"
                    self.reference_map[ref_id] = page_num + 1
                
                # Create document chunk with metadata
                doc = {
                    "content": text,
                    "metadata": {
                        "source": self.ipc_pdf_path,
                        "page": page_num + 1
                    }
                }
                documents.append(doc)
            
            pdf_document.close()
        else:
            # Simple fallback if PyMuPDF is not available
            # This won't extract images or do fancy text formatting
            logger.warning("PyMuPDF not available, using limited text extraction")
            
            try:
                import pdfplumber
                
                with pdfplumber.open(self.ipc_pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        self.page_content[page_num + 1] = text
                        
                        # Create document chunk with metadata
                        doc = {
                            "content": text,
                            "metadata": {
                                "source": self.ipc_pdf_path,
                                "page": page_num + 1
                            }
                        }
                        documents.append(doc)
                
            except ImportError:
                logger.error("Neither PyMuPDF nor pdfplumber available. Cannot extract PDF text.")
                raise ImportError("PDF extraction libraries (PyMuPDF or pdfplumber) not available")
        
        return documents, page_images
    
    def _initialize_rag_system(self):
        """Initialize the RAG system with vector store."""
        
        # Check if we need to build or load the vector store
        vector_store_exists = os.path.exists(self.vector_db_path) and os.path.isdir(self.vector_db_path) and len(os.listdir(self.vector_db_path)) > 0
        
        if not vector_store_exists or self.rebuild_index:
            logger.info("Building vector store from PDF...")
            
            # Extract text and images from PDF
            documents, page_images = self._extract_pdf_text_and_images()
            self.page_images = page_images
            
            if LANGCHAIN_AVAILABLE:
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                # Process each document
                all_splits = []
                for doc in documents:
                    chunks = text_splitter.create_documents(
                        [doc["content"]], 
                        [doc["metadata"]]
                    )
                    all_splits.extend(chunks)
                
                # Create vector store
                embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
                self.vector_store = FAISS.from_documents(all_splits, embeddings)
                
                # Save vector store
                os.makedirs(self.vector_db_path, exist_ok=True)
                self.vector_store.save_local(self.vector_db_path)
                logger.info(f"Vector store built and saved to {self.vector_db_path}")
                
            else:
                # Simple fallback if LangChain is not available
                logger.warning("LangChain not available, using simple text search")
                # Save the documents for simple text search
                self.documents = documents
        
        else:
            logger.info(f"Loading existing vector store from {self.vector_db_path}")
            
            if LANGCHAIN_AVAILABLE:
                # Load existing vector store
                embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
                self.vector_store = FAISS.load_local(
                    self.vector_db_path, 
                    embeddings,
                    allow_dangerous_deserialization=True)
                
                # Still need to load page content and images
                documents, page_images = self._extract_pdf_text_and_images()
                self.page_images = page_images
                
            else:
                # Simple fallback
                documents, page_images = self._extract_pdf_text_and_images()
                self.documents = documents
                self.page_images = page_images
    
    def _search_vector_store(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search the vector store for relevant chunks.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        if LANGCHAIN_AVAILABLE and hasattr(self, 'vector_store'):
            # Search using vector store
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            return formatted_results
        else:
            # Simple keyword search fallback
            results = []
            query_terms = query.lower().split()
            
            for doc in self.documents:
                content = doc["content"].lower()
                # Simple relevance score based on term frequency
                score = sum(content.count(term) for term in query_terms)
                if score > 0:
                    results.append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": score
                    })
            
            # Sort by score and take top k
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
            return results
    
    def answer_ipc_question(self, query: str) -> str:
        """
        Answer a question about the IPC-A-610F manual.
        
        Args:
            query: The question to answer
            
        Returns:
            The answer with references to the manual
        """
        logger.info(f"Processing IPC question: {query}")
        
        # Search for relevant content
        search_results = self._search_vector_store(query, k=5)
        
        if not search_results:
            return "I couldn't find relevant information in the IPC-A-610F manual for your question."
        
        # Format context from search results
        context = []
        page_references = set()
        
        for result in search_results:
            # Extract key information
            content = result["content"]
            page_num = result["metadata"]["page"]
            page_references.add(page_num)
            
            # Add to context
            context.append(f"[Page {page_num}]: {content}")
        
        # Check for referenced figures or tables
        figure_pattern = r'(Figure|Table)\s+(\d+-\d+)'
        referenced_figures = []
        
        for ctx in context:
            for match in re.finditer(figure_pattern, ctx):
                ref_type, ref_num = match.groups()
                ref_id = f"{ref_type} {ref_num}"
                
                if ref_id in self.reference_map:
                    ref_page = self.reference_map[ref_id]
                    page_references.add(ref_page)
                    referenced_figures.append(f"{ref_id} on page {ref_page}")
        
        # Construct the prompt
        prompt = f"""You are a PCB assembly expert assistant specializing in the IPC-A-610F standard for acceptability of electronic assemblies.
Answer the following question based on the IPC-A-610F manual information provided below.

Question: {query}

Relevant information from IPC-A-610F manual:
{"".join(context)}

If there are relevant figures, tables or images mentioned in the text, reference them specifically in your answer.
Referenced items: {", ".join(referenced_figures) if referenced_figures else "None specifically mentioned."}

Provide a detailed answer with page references (in the format [Page X]) whenever possible. 
If you're uncertain or the information is not provided in the context, state that clearly.
"""

        # Generate answer with the model
        answer = self.model_manager.process_text(prompt)
        
        # Format response
        response = f"Based on the IPC-A-610F manual (pages {', '.join(map(str, sorted(page_references)))}): \n\n{answer}"
        
        return response
    
    def analyze_pcb_image(self, image_path: str) -> str:
        """
        Analyze a PCB image for defects.
        
        Args:
            image_path: Path to the PCB image
            
        Returns:
            Analysis of the PCB image with potential defects identified
        """
        logger.info(f"Analyzing PCB image: {image_path}")
        
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
        
        # Construct the prompt
        prompt = """Analyze this PCB (Printed Circuit Board) image for defects or quality issues.
Follow this analysis process:
1. Identify any visible defects or quality issues on the PCB.
2. Classify each issue according to IPC-A-610F categories (e.g., solder defects, component damage, PCB damage).
3. For each identified issue, specify:
   - The exact location on the PCB
   - The type of defect
   - The severity (according to IPC-A-610F Class 1, 2, or 3 criteria)
   - A brief explanation of why it's a concern
4. Provide any recommendations for corrective action.

If you see any potentially defective areas but cannot make a definitive assessment due to image quality or angle, 
mention these as areas requiring further inspection.

If no defects are visible, state this clearly and describe the overall quality of the PCB assembly.
"""

        # Process the image with the model
        result = self.model_manager.process_image(image_path, prompt)
        
        return result
    
    def suggest_reference_images(self, defect_type: str) -> List[Dict[str, str]]:
        """
        Suggest reference images from the IPC manual for a specific defect type.
        
        Args:
            defect_type: Type of defect to find reference images for
            
        Returns:
            List of dictionaries with image paths and their associated text
        """
        logger.info(f"Suggesting reference images for defect type: {defect_type}")
        
        # Search for relevant content
        search_results = self._search_vector_store(defect_type, k=10)
        
        if not search_results:
            return []
        
        # Collect pages with relevant content
        relevant_pages = set()
        for result in search_results:
            relevant_pages.add(result["metadata"]["page"])
        
        # Collect images from relevant pages
        reference_images = []
        for page_num in relevant_pages:
            if page_num in self.page_images:
                for img_path in self.page_images[page_num]:
                    # Get surrounding text context
                    if page_num in self.page_content:
                        page_text = self.page_content[page_num]
                        
                        # Extract figure caption if possible
                        caption = ""
                        figure_pattern = r'(Figure \d+-\d+[^\n.]*\.)'
                        matches = re.findall(figure_pattern, page_text)
                        if matches:
                            caption = matches[0]
                        
                        reference_images.append({
                            "image_path": img_path,
                            "page": page_num,
                            "caption": caption,
                            "context": page_text[:200] + "..." if len(page_text) > 200 else page_text
                        })
        
        return reference_images