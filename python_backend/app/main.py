from fastapi import FastAPI
from pydantic import BaseModel
from pinecone_db import store_text, search_similar

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify_text(request: TextRequest):
    text = request.text

    # 問題文からラベルを分類する
    # Pineconeで検索
    similar_texts = search_similar(text)

    if not similar_texts:
        # 類似テキストが見つからなければ、新たに格納してラベルを保存
        store_text(text, id=text[:10].encode('utf-8').decode('ascii', 'ignore') or "default_id")
        label = "新しいラベル"
    else:
        # 類似テキストが見つかった場合、そのラベルを使う
        label = similar_texts[0]["text"]

    return {"input": text, "predicted_label": label}

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
