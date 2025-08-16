import streamlit as st
from typing import List, Sequence, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage
from langgraph.graph import END, START,StateGraph, add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


import streamlit as st
from PyPDF2 import PdfReader

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.2)
class chatbot(TypedDict):
    qwery:str
    question:str
    context:str
state: chatbot = {
               "qwery": "",
               "question": "",
               "context": "",
            
            }

def chunk(state:chatbot) ->chatbot:
  splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=30)
  chunks = splitter.create_documents([state["text"]])
  state["text"]=chunks
  return state


def indexing(state:chatbot)->chatbot:
       model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

       vector_store = FAISS.from_documents(state["text"], model)
       state["text"]=vector_store
       return state


def retrieval(state:chatbot) ->chatbot:
    retriever = state["text"].as_retriever(search_type="similarity", search_kwargs={"k": 4})
    state["text"]=retriever
    return state


st.title("Ask Questions About Your PDF")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file:
    # Extract text from PDF
    pdf = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf.pages:
        raw_text += page.extract_text() or ""

    state["text"]=raw_text

    
    
    st.success("PDF loaded and text extracted! Ready for questions.(Please wait for few seconds)")

    state=chunk(state)
    state=indexing(state)
    state=retrieval(state)



final_prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer  from the provided pdf-text question and message. where message is the chat history.
      try to understand the question and be collorabative. If question is out of pdf-text then check the messages if it can help.
      Give answer based on the question,context, and chat history check if the question is related to past questions asked by the user. Give the best possible answer and dont mention about context in your answer.

      pdf-text:{pdf-text}
      question: {question}
      message:{message}
    ,
    input_variables = ['pdf-text', 'question','message']""")



    

    
    




















def main():
    st.title("ask  chatbot")

    try:
    
     if 'messages' not in st.session_state:
        st.session_state.messages=[]

     for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
     prompt=st.chat_input("Pass your prompt here")

    

     if prompt:
        state["question"]=prompt
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({"role":"user","content":prompt})
       
        
        result=state["text"].invoke(state["question"])
        context_text = "\n\n".join(doc.page_content for doc in result)
        state["context"]=context_text
        

        

        if state["question"] in ["exit","end","stop"]:
           st.chat_message("assistant").markdown("ok nice to talking you")
   
                
        else:
              final_prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer  from the provided pdf-text question and message. where message is the chat history.
      try to understand the question and be collorabative. If question is out of pdf-text then check the messages if it can help 

      pdf-text:{pdf-text}
      question: {question}
      message:{message}


    """,
    input_variables = ['pdf-text', 'question','message']
).invoke({"pdf-text": state["context"], "question": state["question"],"message":st.session_state.messages})
            #   state["message"].append(HumanMessage(content=state["question"]))
              response = llm.invoke(final_prompt) 
            #   print(response.content) 
              st.chat_message("assistant").markdown(response.content)
            #   state["message"].append(AIMessage(response.content))
              
             
              st.session_state.messages.append({"role":"assistant","content":response.content})
              



    except:
       
       st.chat_message("assistant").write("Please provide pdf")

        









   

if __name__=="__main__":
    main()
