import openai
import logging
import json

# Cấu hình logging
logging.basicConfig(filename='naruto_entity_generator.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_api_key(file_path):
    """Tải API key từ file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def load_input_text(file_path):
    """Tải văn bản đầu vào từ file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def count_tokens(text):
    """Đếm số lượng token trong văn bản (đơn giản)."""
    return len(text.split())

def generate_entities(input_text, api_key):
    """Tạo entities từ văn bản đầu vào bằng GPT API."""
    openai.api_key = api_key
    entities = []

    # Chia nhỏ văn bản đầu vào thành các phần nhỏ hơn
    sentences = input_text.split('. ')  # Tách theo câu
    max_tokens_per_request = 500  # Số token tối đa cho mỗi yêu cầu

    current_part = ""
    for sentence in sentences:
        if count_tokens(current_part + ' ' + sentence) > max_tokens_per_request:
            # Nếu phần này quá lớn, gửi yêu cầu trước đó
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": f"Tạo entity theo định dạng sau cho văn bản: {current_part}\n"
                                                      "Định dạng: { 'instruction': 'Câu hỏi', 'input': '', 'output': 'Câu trả lời' }"}
                    ]
                )
                entities.append(response['choices'][0]['message']['content'])
            except Exception as e:
                logging.error(f"Error generating entities for part: {current_part} - {str(e)}")
            current_part = sentence  # Bắt đầu phần mới
        else:
            current_part += ' ' + sentence

    # Gửi phần còn lại (nếu có)
    if current_part:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"Tạo entity theo định dạng sau cho văn bản: {current_part}\n"
                                                  "Định dạng: { 'instruction': 'Câu hỏi', 'input': '', 'output': 'Câu trả lời' }"}
                ]
            )
            entities.append(response['choices'][0]['message']['content'])
        except Exception as e:
            logging.error(f"Error generating entities for part: {current_part} - {str(e)}")

    return entities

def main():
    api_key = load_api_key('api_key.txt')
    input_text = load_input_text('input_text.txt')

    # Tạo entities
    entities = generate_entities(input_text, api_key)
    
    if entities:
        # Chuyển đổi entities thành định dạng JSON
        formatted_entities = [json.loads(entity) for entity in entities if entity.strip()]
        
        # In ra các entity
        print(json.dumps(formatted_entities, ensure_ascii=False, indent=4))
        logging.info("Entities generated successfully.")
    else:
        logging.error("Failed to generate entities.")

if __name__ == "__main__":
    main()
