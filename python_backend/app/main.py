import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone_db import store_text, search_similar

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class StoreRequest(BaseModel):
    text: str
    labels: list[str]  # ユーザーが確定したラベル

#=========確認用================
# class SimilarRequest(BaseModel):
#     text: str
#     labels: list[str]  # 確認したいラベル

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
    labels = request.labels
    # IDはUUIDを使用
    id = str(uuid.uuid4())

    # Pineconeに保存
    store_text(text, labels, id)

    # ラベルを使用した類似検索
    similar_texts = search_similar(text, labels)
    return {
        "message": "Text stored successfully!",
        "text": text,
        "labels": labels,
        "similar_texts": similar_texts
    }


#===========確認用=============
# @app.post("/similar")
# async def check_similar_texts(request: SimilarRequest):
#     """ 保存せずに、指定されたラベルの範囲で類似検索を行うAPI """
#     text = request.text
#     labels = request.labels

#     similar_texts = search_similar(text, labels)

#     return {
#         "input": text,
#         "labels": labels,
#         "similar_texts": similar_texts
#     }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
