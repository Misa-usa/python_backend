from sentence_transformers import SentenceTransformer
import torch

# 1. Sentence-BERT モデルのロード
def load_model():
    # 使用するSentence-BERTのモデルを指定（例: 'all-MiniLM-L6-v2'）
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# 2. テキストをベクトル化する関数
def get_vector(text, model):
    # テキストをベクトルに変換
    # 結果はtorchのテンソルとして得られます
    vector = model.encode(text, convert_to_tensor=True)
    return vector

# 3. テキストを分類する関数
def classify_text(text):
    # モデルをロード
    model = load_model()

    # テキストをベクトル化
    vector = get_vector(text, model)

    # ベクトル（テンソル）をリスト形式に変換
    vector_list = vector.cpu().detach().numpy().tolist()

    return vector_list

if __name__ == "__main__":
    # テスト用の問題文
    test_text = "微分の公式を教えて"

    # テキストをベクトル化して表示
    vector_result = classify_text(test_text)
    #print(f"ベクトル化結果: {vector_result}")
