from fastapi import FastAPI
from pydantic import BaseModel
from pinecone_db import store_text, search_similar

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify_text(request: TextRequest):
    text = request.text

    # Pineconeで類似検索
    similar_texts = search_similar(text)

    return {"input": text, "similar_texts": similar_texts}

@app.post("/store")
async def store_text_api(request: TextRequest):
    text = request.text
    store_text(text, id=text[:10])  # 簡単なID付与（最初の10文字）

    return {"message": "Text stored successfully!", "text": text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
