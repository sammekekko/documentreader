from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv
from docx import Document


def read_docx(doc):
    text = ""
    if (doc.type == "application/pdf"):
        doc_reader = PdfReader(doc)
        for page in doc_reader.pages:
            text += page.extract_text()
    elif (doc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
        doc = Document(doc)
        for paragraph in doc.paragraphs:
            text += paragraph.text

    return text


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your document!",
                       page_icon="https://cdn.discordapp.com/avatars/227799114610376704/7f7190f7e14b9e11f462a69744ba5dc1.webp?size=240")
    st.header("Ask your Word or PDF document 💬")

    uploaded_file = st.file_uploader(
        "Upload your document", type=['pdf', 'docx'])

    if uploaded_file is not None:
        doc_text = read_docx(uploaded_file)

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ' ', ''],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # Split the text into chunks
        chunks = text_splitter.split_text(doc_text)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Show user input
        user_question = st.text_input(
            "Ask a question about your Word document:", placeholder="Write a 100 word summary")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.write(response)


if __name__ == '__main__':
    main()
