from logging import Logger

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger: Logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Initializing Huggingface embedding model")
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Huggingface embedding model loaded successfully")
        
        return model
    
    except Exception as error:
        error_message = CustomException("Error occured while loading embedding model", error)
        logger.error(str(error_message))
        
        raise error_message

