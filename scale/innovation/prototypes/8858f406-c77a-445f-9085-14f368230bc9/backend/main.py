from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Innovation Prototype API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class InnovationIdea(BaseModel):
    id: str
    title: str
    description: str
    status: str

# Routes
@app.get("/")
async def root():
    return {"message": "Innovation Prototype API"}

@app.post("/innovation/")
async def create_innovation(idea: InnovationIdea):
    # TODO: Implement innovation creation logic
    return {"message": "Innovation created", "idea": idea}

@app.get("/innovation/{idea_id}")
async def get_innovation(idea_id: str):
    # TODO: Implement innovation retrieval logic
    return {"id": idea_id, "title": "Sample Innovation"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
