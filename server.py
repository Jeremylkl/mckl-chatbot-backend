from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from assistant import AIAssistant

app = FastAPI()

# Allow your frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict e.g. ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = AIAssistant()


@app.get("/")
def home():
    return {"message": "✅ MCKL AI Agent Backend is running!"}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("message", "")

    if not query.strip():
        return {"reply": "Please enter a valid question."}

    try:
        result = assistant.ask(query)
        return {
            "reply": result["answer"],
            "sources": result["sources"],
        }
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}
