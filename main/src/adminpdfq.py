from fastapi import HTTPException
from pypdf.errors import PdfReadError
from pypdf import PdfReader

async def extract_text_from_pdf(file):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        text = ""
        reader = PdfReader(file.file)
        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in the PDF.")

        return text

    except PdfReadError as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupt PDF file: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while processing the PDF: {e}")
