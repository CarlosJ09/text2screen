from fastapi import FastAPI
from pydantic import BaseModel
from video import ai_model

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/ask")
async def generate(request: PromptRequest):
    prompt = request.prompt
    response = await ai_model.generate_video(prompt)
    return {"response": response}
