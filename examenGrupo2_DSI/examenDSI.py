import os

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredHTMLLoader
import streamlit as st
from langchain.vectorstores import FAISS


os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.html'



def process_doc(
        path: str = "C:\\Users\\Mateo\\Downloads\\Documento-de-examen-Grupo1.html",
        is_local: bool = False,
        question: str = 'Quiénes son los autores del pdf?'
):

    _, loader = os.system(f'curl -o {default_doc_name} {path}'), UnstructuredHTMLLoader(f"./{default_doc_name}") if not is_local \
        else UnstructuredHTMLLoader(path)

    doc = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=9500)
    texts = text_splitter.split_documents(doc)

    #print(texts[0].page_content)

    #print(texts[-1])
    embedding = OpenAIEmbeddings()
    #db = Chroma.from_documents(texts, embedding=OpenAIEmbeddings())


    faiss = FAISS.from_documents(texts, embedding)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='refine', retriever=faiss.as_retriever())
    st.write(qa.run(question))



    st.write(qa.run(question))
    print( "\n*** LA PREGUNTA ES:"+question, "\n****"+qa.run(question))


def client():
    st.title('Manejo LLM with LangChain')
    uploader = st.file_uploader('Upload Html', type='html')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('HTML saved!!')

    question = st.text_input('Generar un resumen de 20 palabras sobre el HTML',
                             placeholder='Give response about your HTML', disabled=not uploader)

    if st.button('Send Question'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question
            )
        else:
            st.info('Loading default HTML')
            process_doc()


if __name__ == '__main__':
   client()
