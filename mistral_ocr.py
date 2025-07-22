import os
import base64
import fitz  # PyMuPDF
import logging
from mistralai import Mistral
from typing import List
from pathlib import Path
import requests
import tempfile


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartOCRProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Mistral(api_key=self.api_key)

    def encode_pdf(self, pdf_path: str) -> str:
        """Encode full PDF to base64."""
        try:
            with open(pdf_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode PDF: {e}")
            raise

    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """Check if the PDF is image-based (scanned) or digital."""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                if page.get_text().strip():
                    doc.close()  # Close document before returning
                    return False  # Contains extractable text
            doc.close()  # Close document
            return True
        except Exception as e:
            logger.error(f"Failed to analyze PDF: {e}")
            raise

    logger = logging.getLogger(__name__)

    def process_pdf_directly(self, pdf_path: str, output_folder: str) -> None:
        """Use Mistral OCR API for individual PDF pages and output a single combined text file."""
        logger.info("Using page-by-page OCR method")
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        full_text = ""

        try:
            pdf_document = fitz.open(pdf_path)

            for page_num in range(len(pdf_document)):
                logger.info(f"Processing page {page_num + 1}/{len(pdf_document)}")
                try:
                    # Extract single page
                    single_page_pdf = fitz.open()
                    single_page_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)

                    # Create a temporary file safely (avoid permission issues on Windows)
                    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf")
                    os.close(temp_fd)  # Close file descriptor immediately
                    single_page_pdf.save(temp_path)
                    single_page_pdf.close()

                    # Encode PDF
                    base64_pdf = self.encode_pdf(temp_path)

                    # OCR request
                    response = self.client.ocr.process(
                        model="mistral-ocr-latest",
                        document={
                            "type": "document_url",
                            "document_url": f"data:application/pdf;base64,{base64_pdf}"
                        },
                        include_image_base64=True
                    )

                    # Cleanup
                    Path(temp_path).unlink(missing_ok=True)

                    # Append OCR result
                    if response.pages and hasattr(response.pages[0], "markdown"):
                        page_text = response.pages[0].markdown
                        full_text += f"\n\n### Page {page_num + 1} ###\n\n{page_text.strip()}\n"
                    else:
                        raise ValueError("No markdown found in OCR response")

                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {e}")
                    full_text += f"\n\n### Page {page_num + 1} - Error ###\nError processing page: {e}\n"

            pdf_document.close()

            # Write final output
            final_path = Path(output_folder) / "output.txt"
            final_path.write_text(full_text.strip(), encoding="utf-8")
            logger.info(f"OCR text saved to {final_path}")

        except Exception as e:
            logger.error(f"Failed to open or process PDF: {e}")
            raise

    def fallback_process_with_images(self, pdf_path: str, output_dir="ocr_pages") -> None:
        """Fallback: convert PDF to images and OCR each image, saving all results in one text file."""
        logger.info("Using fallback image-based OCR method")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        full_text = ""
        doc = fitz.open(pdf_path)

        try:
            for i, page in enumerate(doc):
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                image_path = Path(output_dir) / f"page_{i+1:04d}.png"
                pix.save(str(image_path))

                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                prompt = (
                    "Extract all text using OCR. Preserve formatting, tables, headings, and lists. "
                    "Return only the extracted text."
                )

                payload = {
                    "model": "pixtral-12b-2409",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                            ]
                        }
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.1
                }

                response = requests.post(
                    url="https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )

                if response.ok:
                    result = response.json()
                    text = result['choices'][0]['message']['content']
                    full_text += f"\n\n### Page {i + 1} ###\n\n{text.strip()}\n"
                    logger.info(f"OCR succeeded for page {i+1}")
                else:
                    error_msg = response.text
                    logger.error(f"Failed OCR for page {i + 1}: {error_msg}")
                    full_text += f"\n\n### Page {i + 1} - Error ###\nError: {error_msg}\n"

            final_path = Path(output_dir) / "output.txt"
            final_path.write_text(full_text.strip(), encoding="utf-8")
            logger.info(f"Final OCR text saved to {final_path}")

        finally:
            doc.close()

    def process(self, pdf_path: str, output_folder: str = "ocr_pages"):
        """Main method to process PDF with hybrid approach."""
        logger.info(f"Starting OCR process for {pdf_path}")
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return

        try:
            if self.is_scanned_pdf(pdf_path):
                self.fallback_process_with_images(pdf_path, output_dir=output_folder)
            else:
                self.process_pdf_directly(pdf_path, output_folder=output_folder)
            logger.info(f"OCR complete. Output saved to folder: {output_folder}")

        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")

if __name__ == "__main__":
    API_KEY = os.getenv("MISTRAL_API_KEY","8AQWcztz83ZW2Anc5FHUms8DYR2JoJjR" )  # Use env var first
    PDF_PATH = r"Mistral_OCR\novartis-integrated-report-2024.pdf"  # Path to your PDF file
    OUTPUT_FOLDER = "novartis_ocr_integrated_report"  # Output folder for OCR results

    if not API_KEY:
        logger.error("Missing MISTRAL_API_KEY environment variable.")
    else:
        processor = SmartOCRProcessor(api_key=API_KEY)
        processor.process(pdf_path=PDF_PATH, output_folder=OUTPUT_FOLDER)  # Use the main process method