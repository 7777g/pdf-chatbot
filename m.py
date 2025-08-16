from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import TypedDict
import PyPDF2
import io
from chatbot import llm, chunk ,retrieval, indexing, final_prompt

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class chatbot(TypedDict):
    text:str
    question:str
    context:str
state: chatbot = {
               "text": "",
               "question": "",
               "context": "",
            
            }
chat_history= []

# --- Upload page ---
@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    global chat_history
    global state
    state["context"]=""
    state["text"]=""
    state["question"]=""
    chat_history=[]
    return templates.TemplateResponse("3.html", {"request": request})

# --- Handle PDF upload, then redirect to /chat ---
@app.post("/upload-pdf/", response_class=HTMLResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    try:
        global state
        pdf_bytes = await file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))

        
        for page in reader.pages:
            state["text"] += page.extract_text() or ""

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "history": chat_history,
            "error": f"Error reading PDF: {e}",
        })
    
      
    


    # Go to chat page after successful upload but first we apply rag
    return RedirectResponse(url="/rag", status_code=303)


@app.get("/rag", response_class=HTMLResponse)
def rag(request: Request):
 global state

 state=chunk(state)
 state=indexing(state)
 state=retrieval(state)
 return RedirectResponse(url="/chat", status_code=303)

# --- Chat page (GET) ---
@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": chat_history,
        "pdf_loaded": state["text"],
    })

# --- Chat submit (POST) ---
@app.post("/chat", response_class=HTMLResponse)
def chat_submit(request: Request, user_message: str = Form(...)):
    # Guard: require a PDF first
    if not state["text"]:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "history": chat_history,
            "pdf_loaded": False,
            "error": "Please upload a PDF first.",
        })

    chat_history.append(("You", user_message))
    state["question"] = user_message
    result=state["text"].invoke(state["question"])
    context_text = "\n\n".join(doc.page_content for doc in result)
    state["context"]=context_text
    response=final_prompt.invoke({"pdf-text": state["context"], "question": state["question"],"message":chat_history})
   
       # bot reply 
    bot_reply = f"{llm.invoke(response).content}"
    chat_history.append(("Bot", bot_reply))


    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": chat_history,
        "pdf_loaded": True,

    })





