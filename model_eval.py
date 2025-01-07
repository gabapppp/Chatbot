import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Đường dẫn tới Chroma database
CHROMA_PATH = "chroma"

# Mẫu prompt cho LLM
TEMPLATE_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

def get_embedding_function():
    """Lấy embedding function cho Chroma database"""
    return OllamaEmbeddings(model="nomic-embed-text")


def evaluate_response(expected_response, model_response, embedding_function, threshold=0.8):
    """Đánh giá câu trả lời của mô hình bằng BLEU, METEOR và Accuracy dựa trên ngữ nghĩa"""
    
    # Tính BLEU score
    bleu_score = sentence_bleu([expected_response.split()], model_response.split(), smoothing_function=SmoothingFunction().method4)
    
    # Tính METEOR score
    meteor_score_value = meteor_score([expected_response.split()], model_response.split(), alpha=0.9, beta=3, gamma=0.5)
    # Tính Accuracy dựa trên cosine similarity của câu trả lời và câu kỳ vọng
    # Lấy embedding cho cả câu trả lời kỳ vọng và câu trả lời mô hình
    expected_embedding = embedding_function.embed_query(expected_response)
    model_embedding = embedding_function.embed_query(model_response)
    
    # Tính độ tương đồng cosine
    cosine_sim = cosine_similarity([expected_embedding], [model_embedding])[0][0]
    
    # So sánh với ngưỡng xác định (ví dụ: 0.8)
    accuracy = 1 if cosine_sim >= threshold else 0
    
    return bleu_score, meteor_score_value, accuracy

def query_rag(query_text: str):
    """Truy vấn cơ sở dữ liệu RAG và tạo câu trả lời từ mô hình LLM"""
    # Chuẩn bị embedding function và cơ sở dữ liệu
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Tìm kiếm trong cơ sở dữ liệu
    results = db.similarity_search_with_score(query_text, k=5)
    # Perform similarity search to retrieve relevant documents from the DB.
    results = db.similarity_search_with_score(query_text, k=1)

    # Extract the top k results' text to form the context for the response
    context = "\n".join([doc.page_content for doc, _score in results])
    # Tạo prompt
    prompt_template = ChatPromptTemplate.from_template(TEMPLATE_PROMPT)
    prompt = prompt_template.format(
        instruction=query_text,
        # input=context,
        input="",
        output=""
    )

    # Tạo câu trả lời từ mô hình LLM
    model = Ollama(model="chatbot")
    response_text = model.invoke(prompt)

    return response_text

def evaluate_model_on_test_set(test_set_path):
    """Đánh giá mô hình trên toàn bộ test set"""
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    bleu_scores = []
    meteor_scores = []
    accuracies = []

    # Duyệt qua từng test case trong bộ dữ liệu
    for test_case in test_set:
        query_text = test_case["instruction"]
        expected_response = test_case["output"]
        
        # Lấy câu trả lời từ mô hình qua RAG
        model_response = query_rag(query_text)
        
        # Đánh giá câu trả lời của mô hình
        bleu, meteor, accuracy = evaluate_response(expected_response, model_response, get_embedding_function())
        
        # Lưu kết quả đánh giá
        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        accuracies.append(accuracy)

    # Tính toán các chỉ số trung bình
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

    # In kết quả trung bình
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average METEOR: {avg_meteor:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")

    return {
        "avg_bleu": avg_bleu,
        "avg_meteor": avg_meteor,
        "avg_accuracy": avg_accuracy
    }

# Đường dẫn đến file test set
test_set_path = "./Film_Dataset/test_set.json"  # Thay đổi đường dẫn file test set của bạn
evaluation_results = evaluate_model_on_test_set(test_set_path)
