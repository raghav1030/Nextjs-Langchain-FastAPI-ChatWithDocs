from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline 
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from .constants import CHROMA_SETTINGS

import os
from dotenv import load_dotenv

load_dotenv()
print("API_TOKEN",os.getenv('HUGGINGFACEHUB_API_TOKEN'))


HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')


def qa_llm():

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5}, huggingfacehub_api_token="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"  )

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = Chroma(
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,        
    )
    
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type='stuff',
        # return_source_document = True,
    )
    print(qa)
    return qa
    
def process_answer(instruction):
    # response=''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    return generated_text

# if __name__ == '__main__':
#     instruction = 'What is there in the document?'
#     print(instruction)
#     print(process_answer(instruction))
    
