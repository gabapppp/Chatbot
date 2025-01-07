import json

# Đọc dữ liệu từ file JSON
try:
    with open('dataset.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
except UnicodeDecodeError:
    # Nếu xảy ra lỗi với UTF-8, thử với utf-16
    with open('dataset.json', 'r', encoding='utf-16') as file:
        data = json.load(file)

# Sử dụng một set để theo dõi các entity đã gặp
unique_entities = set()
unique_data = []

# Lặp qua từng entity trong dữ liệu
for entry in data:
    # Tạo một tuple với các giá trị của entity để dễ dàng so sánh
    entity_tuple = (entry['instruction'], entry['input'], entry['output'])
    
    # Nếu entity chưa có trong set, thêm nó vào danh sách unique_data và set
    if entity_tuple not in unique_entities:
        unique_entities.add(entity_tuple)
        unique_data.append(entry)

# Ghi dữ liệu đã loại bỏ entity trùng vào file JSON mới
with open('output.json', 'w', encoding='utf-8') as file:
    json.dump(unique_data, file, ensure_ascii=False, indent=4)

print("Đã loại bỏ các entity trùng nhau và lưu vào 'output.json'")
