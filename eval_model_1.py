from langchain_community.embeddings.ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

# Khởi tạo mô hình embedding
embedding_function = OllamaEmbeddings(model="nomic-embed-text")

def evaluate_response(expected_response, model_response, threshold=0.8):
    """Đánh giá câu trả lời của mô hình bằng BLEU, METEOR và Accuracy dựa trên ngữ nghĩa"""
    
    # Tính BLEU score
    bleu_score = sentence_bleu([expected_response.split()], model_response.split())
    
    # Tính METEOR score
    meteor_score_value = meteor_score([expected_response.split()], model_response.split())
    
    # Tính Accuracy dựa trên cosine similarity của câu trả lời và câu kỳ vọng
    # Lấy embedding cho cả câu trả lời kỳ vọng và câu trả lời mô hình
    expected_embedding = embedding_function.embed_query(expected_response)
    model_embedding = embedding_function.embed_query(model_response)
    
    # Tính độ tương đồng cosine
    cosine_sim = cosine_similarity([expected_embedding], [model_embedding])[0][0]
    
    # So sánh với ngưỡng xác định (ví dụ: 0.8)
    accuracy = 1 if cosine_sim >= threshold else 0
    
    return bleu_score, meteor_score_value, accuracy

# Kiểm tra với ví dụ
expected_response = "Itachi sử dụng Amaterasu, ngọn lửa đen thiêu cháy mọi thứ đến khi không còn gì, để tiêu diệt kẻ thù nhanh chóng."
model_response = "Itachi sử dụng Amaterasu, một ngọn lửa đen thiêu cháy mọi thứ không ngừng cho đến khi nó biến mất."

bleu, meteor, accuracy = evaluate_response(expected_response, model_response)
print(f"BLEU: {bleu}, METEOR: {meteor}, Accuracy: {accuracy}")
