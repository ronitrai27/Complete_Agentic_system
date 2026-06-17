import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()

from src.pipelines.ingestion import ingest_document_pipeline
from loguru import logger

def reingest():
    # File paths
    txt_path = "data/enterprise_knowledge_base.txt"
    docx_path = "data/test_prd.docx"
    
    logger.info("Starting ingestion of enterprise_knowledge_base.txt...")
    ingest_document_pipeline(
        file_path=txt_path,
        document_id="enterprise_knowledge_base.txt",
        conversation_id="system_reingest"
    )
    
    logger.info("Starting ingestion of test_prd.docx...")
    ingest_document_pipeline(
        file_path=docx_path,
        document_id="test_prd.docx",
        conversation_id="system_reingest"
    )
    
    logger.info("✅ All documents successfully re-ingested!")

if __name__ == "__main__":
    reingest()
