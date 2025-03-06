from fastapi import FastAPI
from pydantic import BaseModel
from pinecone_db import store_text, search_similar, assign_labels_to_text

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify_text(request: TextRequest):
    """
    入力された問題文に対して、類似検索を行い、推奨ラベルを返すAPI
    """
    text = request.text
    suggested_labels = assign_labels_to_text(text)

    return {"input": text, "suggested_labels": suggested_labels}


@app.post("/store")
async def store_text_api(request: TextRequest):
    text = request.text
    # IDが空でないことを確認する
    id = text[:10].encode('utf-8').decode('ascii', 'ignore') or "default_id"
    store_text(text, id=id)  # 簡単なID付与（最初の10文字）

    # Pineconeで類似検索
    similar_texts = search_similar(text)
    return {"input": text, "similar_texts": similar_texts}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
