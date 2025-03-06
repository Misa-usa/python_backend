from bert_model import classify_text
import os
from pinecone import Pinecone, ServerlessSpec

# Pineconeクラスのインスタンスを作成
pc = Pinecone(api_key="pcsk_4wZuZE_KHMfWqySwMKQKUk3q7q4tHjfBe9mAb5itrxvPTarNCTivwWipj6gAmKwENL1iDQ")

# インデックスが存在しない場合、作成する
index_name = 'myindex'
if index_name not in pc.list_indexes().names():
    pc.create_index(
    name=index_name,
    dimension=1536,
    metric='euclidean',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'  # 利用可能なリージョンに変更
    )
)


# インデックスに接続
index = pc.Index(index_name)

def store_text(text: str, id: str):
    """テキストをベクトル化してPineconeに保存"""
    vector = classify_text(text)  # 埋め込みベクトルを取得
    print('record:'+ vector)
    index.upsert([(id, vector, {"text": text})])  # Pineconeにアップサート

def search_similar(text: str, top_k=3):
    """テキストの埋め込みを作成し、類似検索"""
    vector = classify_text(text)  # 埋め込みベクトルを取得
    print('search:' + str(vector))
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)  # クエリ実行
    print('search:'+ str(results))
    # 結果を整形
    return [{"score": match["score"], "text": match["metadata"]["text"]} for match in results["matches"]]
