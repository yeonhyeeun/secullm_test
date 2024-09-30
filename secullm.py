import streamlit as st
from flask import Flask
import os, torch
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from loguru import logger

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from kiwipiepy import Kiwi
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

#########################################################
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "test_streamlit"
######################################################## 

app = Flask(__name__)
@st.cache_data
# 데이터 문서 로드 및 전처리 함수
def load_and_preprocess_documents(filepath):
    loader = PyMuPDFLoader(filepath)
    docs = loader.load()

    # 데이터 정리
    for i in range(len(docs)):
        docs[i].page_content = docs[i].page_content.replace("ft", "처")
        docs[i].page_content = re.sub(r'\s+', ' ', docs[i].page_content)
        docs[i].page_content = re.sub(r'(\n\s*\n)+', '\n', docs[i].page_content)
        docs[i].page_content = re.sub(r'([.!?·@#])\1+', r'\1', docs[i].page_content)
    
    # 고정 청크 사용
    text_splitter = CharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separator=' ',
        length_function=len,
    )
    texts = text_splitter.split_documents(docs)

    for i in range(len(texts)):
        texts[i].page_content = re.sub(r'\s+', ' ', texts[i].page_content)

    return texts 
 
# Initialize Kiwi tokenizer
kiwi = Kiwi()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Tokenizer using Kiwi

def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]
 
# Load CSV file and split it into chunks

@st.cache_data

def load_pdf_and_split(file_path, chunk_size=1000, chunk_overlap=100):

    loader = PyPDFLoader(file_path=file_path)

    pdf_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)

    docs = text_splitter.split_documents(pdf_docs)

    return docs
 
# Load HuggingFace embeddings

def load_hf_embeddings(model_name, device):

    return HuggingFaceEmbeddings(

        model_name=model_name,

        model_kwargs={'device': device},

        encode_kwargs={'normalize_embeddings': True}

    )
 
# Create FAISS vector store

def create_faiss_vector_store(docs, hf):

    return FAISS.from_documents(docs, hf)
 
# Create BM25 retriever

def create_bm25_retriever(docs):

    retriever = BM25Retriever.from_documents(documents=docs, preprocess_func=kiwi_tokenize)

    retriever.k = 3

    return retriever
 
# Create ensemble retriever

def create_ensemble_retriever(kiwi_bm25_retriever, faiss_retriever):

    return EnsembleRetriever(

        retrievers=[kiwi_bm25_retriever, faiss_retriever],

        weights=[0.7, 0.3],

        search_type='mmr'

    )
 
# Define prompt template

def get_prompt_template():

    return '''

    당신은 개인정보 처리방침과 관련된 질문-응답을 답변하는 AI 어시스턴트입니다. 다음 주어진 정보(context)는 문서에서 검색된 내용입니다.\n

    답변을 할 때의 규칙입니다. \n

    답변은 반드시 검색된 정보(context)를 기반으로 답변하세요. \n

    만약에, 답변할 만한 정보를 찾을 수 없다면 '모르겠습니다. 죄송합니다.'라고 답변하세요.\n

    모든 답변은 반드시 한국어(Korean)로 5줄 이내로 최대한 짧게 답변하세요.\n

    ### 정보: {context}

    ### 질문: {question}

    ### 답변:

    '''
 
# Load model

def load_llm_model(model_name):

    return ChatOllama(model=model_name, temperature=0)
 
# Create RAG chain

def create_rag_chain(retriever, llm, prompt_template):

    prompt = ChatPromptTemplate.from_template(prompt_template)

    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
 
def get_text(docs):
    doc_list = []

    for doc in docs:
       file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
       with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
           file.write(doc.getvalue())
           logger.info(f"Uploaded {file_name}")
           
       if '.pdf' in doc.name:
           loader = PyPDFLoader(file_name)
           documents = loader.load_and_split()
           
       doc_list.extend(documents)
       
    return doc_list 
 


# Streamlit UI
def run_streamlit_app():
    st.write("문서에 기반한 질문에 대해 답변을 제공합니다.")

    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

    if uploaded_file is not None:
        # 파일을 임시로 저장
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("파일이 성공적으로 업로드되었습니다.")

        # 문서 로드 및 전처리
        docs = load_and_preprocess_documents("uploaded_file.pdf")
        st.write('PDF 파일 로드 및 청크 처리 완료')

        hf = load_hf_embeddings("intfloat/multilingual-e5-large-instruct", device)
        faiss_vector = create_faiss_vector_store(docs, hf)
        st.write('벡터 저장소 생성 완료')

        kiwi_bm25_retriever = create_bm25_retriever(docs)
        chroma_retriever = faiss_vector.as_retriever(search_kwargs={'k': 3})
        ensemble_retriever = create_ensemble_retriever(kiwi_bm25_retriever, chroma_retriever)

        llm = load_llm_model("bnksys/yanolja-eeve-korean-instruct-10.8b:latest")
        prompt_template = get_prompt_template()
        rag_chain = create_rag_chain(ensemble_retriever, llm, prompt_template)

        query = st.text_input("질문을 입력하세요")

        if st.button("답변 받기"):
            if query:
                with st.spinner("AI가 생각 중입니다..."):
                    result = rag_chain.invoke(query)
                    st.write(f"답변: {result}")
            else:
                st.write("질문을 입력해 주세요.")
    else:
        st.write("업로드된 파일이 없습니다.")

# Main function
if __name__ == "__main__":
    st.set_page_config(
        page_title="secullm",
        page_icon=":books:")
    st.title("_private Data :red[QA Chat]_ :books:")
    
    run_streamlit_app()
