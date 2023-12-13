from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS
from dotenv import load_dotenv

load_dotenv()


# persist_directory = 'C:/Users/Raghav/Desktop/Langchain/nextjs-fastapi/api/langchain/db'
docs_directory = os.getenv('docs_directory')
print("docs_directory",docs_directory)



def main():
    # print('hello')

    for root, dirs, files in os.walk(docs_directory):
        for file in files :
            if file.endswith('.pdf'):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text = text_splitter.split_documents(documents)
    print(format(text))

    print("embeddings")
    
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    print("embeddings", embeddings)
    
    db = Chroma.from_documents(text, embeddings, client_settings=CHROMA_SETTINGS)
    
    db.persist()
    db=None
    
if __name__=='__main__':
    main()