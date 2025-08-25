import json
# Đọc các dòng từ file (giả sử file đang mở với biến `a`)
with open('douyin-video-links.txt', 'r') as a:
    data = [ {"id_video": i.strip()} for i in a.readlines() ]

# Ghi dữ liệu ra file JSON
with open('sourse_video_infor.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
