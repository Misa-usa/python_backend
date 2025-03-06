from sentence_transformers import SentenceTransformer

# 軽量で高精度な Sentence-BERT モデルを使用
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    """テキストをBERTで埋め込みベクトルに変換"""
    return model.encode(text).tolist()
