import re
import json

def extract_video_info(url):
    # Sử dụng regex để trích xuất video ID từ URL
    pattern = r'/video/(\d+)/'
    match = re.search(pattern, url)
    
    if match:
        video_id = match.group(1)
        return {"id_video": video_id}
    else:
        return None

def process_urls(file_path):
    results = []
    
    # Đọc file và xử lý từng dòng
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            url = line.strip()
            if url:  # Chỉ xử lý các dòng không rỗng
                video_info = extract_video_info(url)
                if video_info:
                    results.append(video_info)
    
    return results

def main():
    input_file = 'douyin-share-urls.txt'
    output_file = 'sourse_video_infor.json'
    
    # Xử lý các URL
    video_list = process_urls(input_file)
    
    # Ghi kết quả ra file JSON
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(video_list, json_file, ensure_ascii=False, indent=2)
    
    print(f"Đã xử lý {len(video_list)} URL và lưu kết quả vào {output_file}")
    
    # In kết quả ra console để xem
    print(json.dumps(video_list, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()