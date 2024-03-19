import gradio as gr
import os
import datetime
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

def load_db(file, chain_type, k):
    # Load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # Define embedding
    embeddings = OpenAIEmbeddings()
    # Create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # Define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # Create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

qa = load_db("C:/Users/kaylab/Downloads/HyperXite_information.pdf", "stuff", 4)

def chat(query):
  result = qa({"question": query, "chat_history": []})
  return result['answer']

# Define Theme for Gradio Interface
theme = {
    "text": {"background": "#212529", "color": "#FFFFFF"},  # Dark gray background, white text
    "title": {"color": "#FFFFFF"},  # White title text
    "controls": {  # Style buttons
        "background": "#454A4C",  # Darker gray buttons
        "color": "#FFFFFF",  # White button text
    },
    "label": {"color": "#AAAAAA"},  # Light gray text above text boxes
}

iface = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="Hi there! I'm the HyperXite GPT. You can ask me questions like: What is HyperXite? What are the subsystems on the team? When was it founded? ",
    theme=theme
)
#iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="The HyperXite GPT", theme=theme)
iface.launch(share=True)
