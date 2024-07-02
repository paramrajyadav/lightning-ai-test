import boto3
import streamlit as st
import os
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain.vectorstores import astradb
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS







# s3_client

# s3_client=boto3.client()

# Bucket_name=os.getenv("BUCKET_NAME")

# Bedrock

# PDF Loader

# Text_Splitter
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())


def split_text(pages,chunk_size,chunk_overlap):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs=text_splitter.split_documents(pages)

    return docs

def create_vector_store(request_id,documents):
    vectorstore_faiss=FAISS.from_documents(documents=documents,embedding=bedrock_embeddings)



def main():
    st.write("Chat bot")
    uploaded_file=st.file_uploader("Choose a file","Pdf")
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request ID : {request_id}")
        saved_file_name=f"{request_id}.pdf"
        with open(saved_file_name,mode="wb") as w:
            w.write(uploaded_file.getvalue())
        
        loader=PyPDFLoader(saved_file_name)
        pages=loader.load_and_split()

        st.write(f"total pages :{len(pages)}")

        # Split Text
        splitted_text=split_text(pages,1000,200)
        st.write(f"Splitted doc length : {len(splitted_text)}")

        st.write("**********************")

        st.write(splitted_text[0])


        st.write("Creating the vector store")

        result=create_vector_store(request_id,splitted_text)
        if result:
            st.write("Hurrah Pdf processed Successfully")
        else:
            st.write("Error!!")









if __name__=="__main__":
    main()