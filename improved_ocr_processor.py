import os
import base64
import fitz  # PyMuPDF
import logging
from mistralai import Mistral
from typing import List, Optional
from pathlib import Path
import requests
import tempfile
import time
import json
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartOCRProcessor:
    def __init__(self, api_key: str, max_retries: int = 2, retry_delay: float = 3.0):
        self.api_key = api_key
        self.client = Mistral(api_key=self.api_key)
        self.max_retries = max_retries  # Reduced from 3 to 2 to minimize costs
        self.retry_delay = retry_delay

    def encode_pdf(self, pdf_path: str) -> str:
        """Encode full PDF to base64."""
        try:
            with open(pdf_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode PDF: {e}")
            raise

    def is_scanned_pdf(self, pdf_path: str, text_threshold: int = 100) -> bool:
        """
        Check if the PDF is image-based (scanned) or digital.
        Uses a text threshold to determine if there's enough extractable text.
        """
        try:
            doc = fitz.open(pdf_path)
            total_text_length = 0
            pages_checked = 0
            
            # Check first 5 pages or all pages if fewer than 5
            max_pages_to_check = min(5, len(doc))
            
            for page_num in range(max_pages_to_check):
                page = doc[page_num]
                text = page.get_text().strip()
                total_text_length += len(text)
                pages_checked += 1
                
                # If we find substantial text early, it's likely a digital PDF
                if total_text_length > text_threshold * pages_checked:
                    doc.close()
                    return False
            
            doc.close()
            avg_text_per_page = total_text_length / pages_checked if pages_checked > 0 else 0
            return avg_text_per_page < text_threshold
            
        except Exception as e:
            logger.error(f"Failed to analyze PDF: {e}")
            raise

    def retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    def process_pdf_directly(self, pdf_path: str, output_folder: str) -> None:
        """Use Mistral OCR API for individual PDF pages and output a single combined text file."""
        logger.info("Using page-by-page OCR method")
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        full_text = ""
        processing_log = []

        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            logger.info(f"Processing {total_pages} pages")

            for page_num in range(total_pages):
                logger.info(f"Processing page {page_num + 1}/{total_pages}")
                page_start_time = time.time()
                
                try:
                    # Extract single page
                    single_page_pdf = fitz.open()
                    single_page_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)

                    # Create a temporary file safely
                    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf")
                    os.close(temp_fd)
                    single_page_pdf.save(temp_path)
                    single_page_pdf.close()

                    # Encode PDF
                    base64_pdf = self.encode_pdf(temp_path)

                    # OCR request with retry logic - using cost-effective model
                    response = self.retry_with_backoff(
                        self.client.ocr.process,
                        model="mistral-ocr-2024-12-11",  # Specific stable version instead of "latest"
                        document={
                            "type": "document_url", 
                            "document_url": f"data:application/pdf;base64,{base64_pdf}"
                        },
                        include_image_base64=False  # Reduce costs by not including base64 images
                    )

                    # Cleanup temp file
                    Path(temp_path).unlink(missing_ok=True)

                    # Process OCR result
                    if response.pages and len(response.pages) > 0:
                        page_data = response.pages[0]
                        if hasattr(page_data, 'markdown') and page_data.markdown:
                            page_text = page_data.markdown.strip()
                            if page_text:
                                full_text += f"\n\n### Page {page_num + 1} ###\n\n{page_text}\n"
                                page_status = "success"
                            else:
                                logger.warning(f"Page {page_num + 1}: Empty OCR result")
                                full_text += f"\n\n### Page {page_num + 1} - Empty ###\nNo text extracted from this page\n"
                                page_status = "empty"
                        else:
                            raise ValueError("No markdown content found in OCR response")
                    else:
                        raise ValueError("No pages found in OCR response")

                    processing_time = time.time() - page_start_time
                    processing_log.append({
                        "page": page_num + 1,
                        "status": page_status,
                        "processing_time": round(processing_time, 2)
                    })

                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {e}")
                    full_text += f"\n\n### Page {page_num + 1} - Error ###\nError processing page: {str(e)}\n"
                    processing_log.append({
                        "page": page_num + 1,
                        "status": "error",
                        "error": str(e)
                    })

                # Small delay to avoid rate limiting and reduce costs
                time.sleep(1.0)

            pdf_document.close()

            # Write final output
            final_path = Path(output_folder) / "output.txt"
            final_path.write_text(full_text.strip(), encoding="utf-8")
            
            # Write processing log
            log_path = Path(output_folder) / "processing_log.json"
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(processing_log, f, indent=2)
            
            logger.info(f"OCR text saved to {final_path}")
            logger.info(f"Processing log saved to {log_path}")

        except Exception as e:
            logger.error(f"Failed to open or process PDF: {e}")
            raise

    def optimize_image_for_ocr(self, image_path: Path) -> str:
        """Optimize image for better OCR results."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Ensure reasonable image size for OCR (cost-optimized)
                width, height = img.size
                max_dimension = 2048  # Reasonable limit to control API costs
                if width > max_dimension or height > max_dimension:
                    scale_factor = max_dimension / max(width, height)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                elif width < 1200 or height < 1200:
                    # Scale up small images moderately
                    scale_factor = min(1200 / width, 1200 / height)
                    new_width = int(width * scale_factor)  
                    new_height = int(height * scale_factor)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save optimized image
                buffer = io.BytesIO()
                img.save(buffer, format='PNG', optimize=True, dpi=(300, 300))
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Failed to optimize image {image_path}: {e}")
            # Fallback to original image
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

    def fallback_process_with_images(self, pdf_path: str, output_dir="ocr_pages") -> None:
        """Fallback: convert PDF to images and OCR each image, saving all results in one text file."""
        logger.info("Using fallback image-based OCR method")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        full_text = ""
        processing_log = []
        doc = fitz.open(pdf_path)

        try:
            total_pages = len(doc)
            logger.info(f"Converting {total_pages} pages to images for OCR")

            for i, page in enumerate(doc):
                logger.info(f"Processing page {i + 1}/{total_pages}")
                page_start_time = time.time()
                
                try:
                    # Higher resolution matrix for better OCR
                    mat = fitz.Matrix(3.0, 3.0)  # Increased from 2.0
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    image_path = Path(output_dir) / f"page_{i+1:04d}.png"
                    pix.save(str(image_path))

                    # Optimize image for OCR
                    base64_image = self.optimize_image_for_ocr(image_path)

                    # Enhanced prompt for better OCR
                    prompt = (
                        "Perform OCR on this image and extract ALL visible text. "
                        "Use markdown to preserve structure: "
                        "- Use '#' for headings (e.g., '# Heading 1', '## Heading 2'). "
                        "- Preserve paragraph breaks. "
                        "- Use markdown tables (with pipes | and dashes -) to represent tables. "
                        "- Maintain indentation and list formatting. "
                        "Do not summarize or skip any content. "
                        "Return only the markdown text."
)

                    payload = {
                        "model": "pixtral-12b-2409",  # Cost-effective vision model
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                ]
                            }
                        ],
                        "max_tokens": 8000,  # Reduced from 8000 to control costs
                        "temperature": 0.0
                    }

                    # Make request with retry logic
                    response = self.retry_with_backoff(
                        requests.post,
                        url="https://api.mistral.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload,
                        timeout=60  # Add timeout
                    )

                    if response.ok:
                        result = response.json()
                        text = result['choices'][0]['message']['content'].strip()
                        if text:
                            full_text += f"\n\n### Page {i + 1} ###\n\n{text}\n"
                            page_status = "success"
                        else:
                            logger.warning(f"Page {i + 1}: Empty OCR result")
                            full_text += f"\n\n### Page {i + 1} - Empty ###\nNo text extracted from this page\n"
                            page_status = "empty"
                        logger.info(f"OCR succeeded for page {i+1}")
                    else:
                        error_msg = response.text
                        logger.error(f"Failed OCR for page {i + 1}: {error_msg}")
                        full_text += f"\n\n### Page {i + 1} - Error ###\nError: {error_msg}\n"
                        page_status = "error"

                    processing_time = time.time() - page_start_time
                    processing_log.append({
                        "page": i + 1,
                        "status": page_status,
                        "processing_time": round(processing_time, 2)
                    })

                    # Clean up image file to save space
                    image_path.unlink(missing_ok=True)

                except Exception as e:
                    logger.error(f"Failed to process page {i + 1}: {e}")
                    full_text += f"\n\n### Page {i + 1} - Error ###\nError: {str(e)}\n"
                    processing_log.append({
                        "page": i + 1,
                        "status": "error",
                        "error": str(e)
                    })

                # Rate limiting delay (increased to reduce API costs)
                time.sleep(1.5)

            # Write outputs
            final_path = Path(output_dir) / "output.txt"
            final_path.write_text(full_text.strip(), encoding="utf-8")
            
            log_path = Path(output_dir) / "processing_log.json"
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(processing_log, f, indent=2)
            
            logger.info(f"Final OCR text saved to {final_path}")
            logger.info(f"Processing log saved to {log_path}")

        finally:
            doc.close()

    def extract_digital_text(self, pdf_path: str, output_folder: str) -> None:
        """Extract text from digital PDFs directly."""
        logger.info("Extracting text from digital PDF")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        full_text = ""
        doc = fitz.open(pdf_path)
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    full_text += f"\n\n### Page {page_num + 1} ###\n\n{text.strip()}\n"
                else:
                    logger.warning(f"Page {page_num + 1}: No extractable text found")
                    
            final_path = Path(output_folder) / "output.txt"
            final_path.write_text(full_text.strip(), encoding="utf-8")
            logger.info(f"Digital text extracted to {final_path}")
            
        finally:
            doc.close()

    def process(self, pdf_path: str, output_folder: str = "ocr_pages", force_ocr: bool = False):
        """Main method to process PDF with hybrid approach."""
        logger.info(f"Starting processing for {pdf_path}")
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return

        try:
            # Check PDF type and process accordingly
            if force_ocr or self.is_scanned_pdf(pdf_path):
                logger.info("PDF detected as scanned or OCR forced - using OCR methods")
                try:
                    # Try direct PDF OCR first
                    self.process_pdf_directly(pdf_path, output_folder)
                except Exception as e:
                    logger.warning(f"Direct PDF OCR failed: {e}. Falling back to image-based OCR")
                    self.fallback_process_with_images(pdf_path, output_dir=output_folder)
            else:
                logger.info("PDF detected as digital - extracting text directly")
                self.extract_digital_text(pdf_path, output_folder)
                
            logger.info(f"Processing complete. Output saved to folder: {output_folder}")

        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            raise

if __name__ == "__main__":
    API_KEY = os.getenv("MISTRAL_API_KEY", "8AQWcztz83ZW2Anc5FHUms8DYR2JoJjR")
    PDF_PATH = "novartis-annual-report-2024.pdf"
    OUTPUT_FOLDER = "novartis_ocr_annual_reportv2"

    if not API_KEY:
        logger.error("Missing MISTRAL_API_KEY environment variable.")
    else:
        # Cost-optimized processor with conservative retry settings
        processor = SmartOCRProcessor(
            api_key=API_KEY, 
            max_retries=2,  # Reduced retries to minimize costs
            retry_delay=3.0  # Longer delays to avoid rate limits
        )
        
        # Process with option to force OCR even for digital PDFs
        processor.process(
            pdf_path=PDF_PATH, 
            output_folder=OUTPUT_FOLDER,
            force_ocr=False  # Set to True to force OCR even for digital PDFs
        )