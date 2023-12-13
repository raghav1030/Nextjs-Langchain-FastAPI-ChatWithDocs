from fastapi import FastAPI
import sys
from api.langchain_implementation.main import process_answer
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
 

@app.get(f"/api/python")
def hello_world():
    return {"message": "Hello World"}

@app.post(f"/chat/answer/")
async def chat_answer(query : str):
    answer = process_answer(query)
    print(answer)
    return answer