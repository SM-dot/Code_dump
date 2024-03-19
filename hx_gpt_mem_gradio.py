import os
import gradio as gr

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI


def load_db(file, chain_type, k):
  # ... (code to load documents, split, create embeddings, etc. remains the same)
  qa = ConversationalRetrievalChain.from_llm(
      llm=ChatOpenAI(model_name="gpt-3.5-turbo-0301" if datetime.datetime.now().date() < datetime.date(2023, 9, 2) else "gpt-3.5-turbo", temperature=0),
      chain_type=chain_type,
      retriever=retriever,
      return_source_documents=True,
      return_generated_question=True,
      memory=ConversationBufferMemory(max_len=10),  # Stores up to 10 past interactions
  )
  return qa


def get_chat_history(filename="chat_history.txt"):
  """Retrieves chat history from a text file."""
  try:
    with open(filename, "r") as f:
      history = [line.strip().split("|") for line in f.readlines()]
  except FileNotFoundError:
    history = []
  return history


def update_chat_history(history, query, answer, filename="chat_history.txt"):
  """Updates chat history in a text file."""
  with open(filename, "a") as f:
    f.write(f"{query}|{answer}\n")


def chatbot_interface(query):
  """Main function for chatbot interaction, handling conversation flow and history."""
  chat_history = get_chat_history()
  result = load_db("your_data_history.pdf", "stuff", 4)({"question": query, "chat_history": chat_history})
  answer = result["answer"]
  update_chat_history(chat_history, query, answer)
  return answer


# Gradio Interface
iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.Textbox(lines=3, placeholder="Enter your query")],
    outputs=gr.Textbox(label="Chatbot's Response"),
    title="ChatWithYourData Bot",
)

iface.launch()
