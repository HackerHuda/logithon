import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import openai
import json
import re
from collections import OrderedDict

openai.api_key = "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" 
def answer_question(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": question}
        ]
    )
    answer = response.choices[0].message.content.strip()
    return answer

tab1, tab2, tab3 = st.tabs(["Upload PDF", "Summary of the PDF", "Chat with PDF"])


# Define function to summarize text using Sumy
def summarize_text(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()  # Using Latent Semantic Analysis (LSA) summarizer
    summary = summarizer(parser.document, num_sentences)
    
    summary_text = "\n".join([str(sentence) for sentence in summary])
    return summary_text


def main():
    st.header("Chat with PDF 💬")
    
    with tab3:
        if 'conversation' not in st.session_state:
            st.session_state['conversation'] = []

        for role, content in st.session_state['conversation']:
            if role == 'user':
                st.write(f"*You:* {content}")
            elif role == 'assistant':
                st.write(f"*Assistant:* {content}")

        question = st.text_input("Type your question here:")

        if question:
            st.session_state['conversation'].append(('user', question))

            with st.spinner("Generating answer..."):
                answer = answer_question(question, st.session_state['summary_text'])
                
                st.session_state['conversation'].append(('assistant', answer))

            st.write("Answer:")
            st.write(answer)

        else:
            st.write("Please generate a summary in the 'Summary of the PDF' tab.")
    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        # Display the full text
        st.write(text)
        
        # Ask the user for the number of sentences in the summary
        num_sentences = st.number_input("Number of sentences in the summary:", min_value=1, max_value=10, value=3)
        
        # Summarize the text and display the summary
        summary_text = summarize_text(text, num_sentences)
        st.header("Summary")
        st.write(summary_text)
        
if __name__ == '__main__':
    main()

