from bert_model import classify_text
import os
from pinecone import Pinecone, ServerlessSpec
import uuid


# Pineconeクラスのインスタンスを作成
pc = Pinecone(api_key="pcsk_4wZuZE_KHMfWqySwMKQKUk3q7q4tHjfBe9mAb5itrxvPTarNCTivwWipj6gAmKwENL1iDQ")

# インデックスが存在しない場合、作成する
index_name = 'myindex'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # 埋め込みベクトルの次元数
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # 利用可能なリージョンに変更
        )
    )

# インデックスに接続
index = pc.Index(index_name)

# 問題文にラベルを自動付け
def assign_labels_to_text(text, threshold=0.7, top_k=4):
    """
    問題文に対して複数のラベルを付ける関数
    :param text: 解析する問題文
    :param threshold: ラベルを付ける際の類似度スコアの閾値
    :param top_k: 検索するラベルの数
    :return: 類似度スコアが閾値以上のラベル
    """
    # 問題文をベクトル化
    text_vector = classify_text(text)
    
    # Pineconeで類似するラベルを検索
    results = index.query(vector=text_vector, top_k=top_k, include_metadata=True)
    
    labels = []
    for match in results['matches']:
        score = match['score']
        label = match['metadata']['label']
        
        if score >= threshold:  # スコアが閾値以上であれば、ラベルを保存
            labels.append((label, score))
    
    if not labels:  # ラベルが見つからなかった場合
        labels = [("その他 - その他", 1.0)]  # デフォルトのラベルを追加

    return [label for label, _ in labels]  # ラベルを返す


def store_text(text: str, labels: list[str], id: str):
    """
    テキストをベクトル化してPineconeに保存
    """
    vector = classify_text(text)
    metadata = {"text": text, "labels": labels}

    index.upsert([(id, vector, metadata)])

def search_similar(text: str, labels: list[str], top_k=3):
    """
    ラベルでフィルタリングしながら類似検索を実行
    """
    vector = classify_text(text)

    # ラベルでフィルタリング
    filter_conditions = {"labels": {"$in": labels}}

    results = index.query(vector=vector, top_k=top_k, include_metadata=True, filter=filter_conditions)

    return [{"score": match["score"], "text": match["metadata"]["text"]} for match in results["matches"]]
#========ラベルの保存コード=========

# def store_labels(labels):
#     """
#     ラベルセットをPineconeに保存する
#     :param labels: ラベルのリスト
#     """
#     for label in labels:
#         # ラベルに対応する埋め込みベクトルを生成
#         vector = classify_text(label)  # classify_textでラベルの埋め込みを取得
#         # ラベルに対して自動生成したUUIDをIDとして使用
#         generated_id = str(uuid.uuid4())  # 一意なIDを生成
#         index.upsert([(generated_id, vector, {"label": label})])  # ラベルをPineconeに保存


# # 最初にラベルセットを保存
# labels = [
#     "英語 - be動詞・一般動詞", "英語 - 冠詞・名詞の複数形", "英語 - 代名詞", 
#     "英語 - 命令文・感嘆文", "英語 - 疑問視", "英語 - 現在進行形", 
#     "英語 - 助動詞(can:～できる)", "英語 - 前置詞", "英語 - 過去形・過去進行形", 
#     "英語 - 未来を表す文", "英語 - 接続詞", "英語 - There is構文", 
#     "英語 - 不定詞", "英語 - 比較", "英語 - 現在完了", "英語 - 過去形・過去進行形", 
#     "英語 - 接続詞", "英語 - 助動詞(must)", "英語 - 動名詞と不定詞", 
#     "英語 - 比較", "英語 - 受動態", "英語 - 現在完了", "英語 - 受動態", 
#     "英語 - 現在完了進行形", "英語 - 文型", "英語 - いろいろな疑問文", 
#     "英語 - 不定詞の構文", "英語 - 分詞", "英語 - 関係代名詞", "英語 - 仮定法", 
#     "情報 - 情報社会の問題解決", "情報 - コミュニケーションと情報デザイン", 
#     "情報 - コンピュータとプログラミング", "情報 - 情報通信ネットワークとデータの活用"
# ]


# # ラベルを保存
# この文章を消したら保存されます。　　store_labels(labels)

