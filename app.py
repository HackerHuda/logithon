import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer





# Define function to summarize text using Sumy
def summarize_text(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()  # Using Latent Semantic Analysis (LSA) summarizer
    summary = summarizer(parser.document, num_sentences)
    
    summary_text = "\n".join([str(sentence) for sentence in summary])
    return summary_text


def main():
    st.header("Chat with PDF 💬")

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

