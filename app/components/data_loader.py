from logging import Logger
import os
from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.components.vector_store import save_vector_store
from app.config.config import DB_FAISS_PATH

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger: Logger = get_logger(__name__)

def process_and_store_pdf():
    try:
        logger.info("Creating vector store")
        documents = load_pdf_files()
        text_chunks = create_text_chunks(documents)
        save_vector_store(text_chunks)
        logger.info("Vector store created successfully")
        
    except Exception as error:
        error_message = CustomException("Failed to create vector store", error)
        logger.error(str(error_message))
    
        
if __name__ == "__main__":
    process_and_store_pdf()
        

