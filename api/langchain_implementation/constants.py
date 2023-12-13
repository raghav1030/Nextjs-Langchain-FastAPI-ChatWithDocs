from chromadb.config import Settings
import os
from dotenv import load_dotenv
import sys

load_dotenv()


persist_directory = os.getenv('persist_directory')
print(persist_directory)


print("persist_directory", persist_directory)
CHROMA_SETTINGS = Settings(
    chroma_db_impl = 'duckdb+parquet',
    persist_directory = "C:/Users/Raghav/Desktop/Langchain/Nextjs-FastAPI-Langchain/nextjs-fastapi/api/langchain_implementation/ChromaDB",
    # anonymised_telementry = True,
    anonymized_telemetry=False
)