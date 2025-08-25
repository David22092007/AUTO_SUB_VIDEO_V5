import re
import requests
import json
import time

def extract_ids(url):
    pattern = r"space\.bilibili\.com/(\d+)/lists/(\d+)"
    match = re.search(pattern, url)
    if match:
        user_id = match.group(1)
        playlist_id = match.group(2)
        return user_id, playlist_id
    return None, None

def get_all_pages_data(user_id, playlist_id, headers):
    """Lấy dữ liệu từ tất cả các trang tự động"""
    all_data = []
    page_num = 1
    has_more = True
    
    while has_more:
        print(f"Đang lấy trang {page_num}...")
        
        params = {
            'mid': user_id,
            'season_id': playlist_id,
            'sort_reverse': 'false',
            'page_size': '30',
            'page_num': str(page_num),
            'web_location': '333.1387',
        }

        try:
            response = requests.get(
                'https://api.bilibili.com/x/polymer/web-space/seasons_archives_list',
                params=params,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                page_data = response.json()
                all_data.append(page_data)
                
                # Kiểm tra xem còn dữ liệu không
                archives = page_data.get('data', {}).get('archives', [])
                if not archives or len(archives) < 30:  # Nếu ít hơn page_size thì hết dữ liệu
                    has_more = False
                    print(f"Đã lấy hết dữ liệu ở trang {page_num}")
                else:
                    page_num += 1
            else:
                print(f"Lỗi khi lấy trang {page_num}: {response.status_code}")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Lỗi kết nối khi lấy trang {page_num}: {e}")
            break
        
        # Thêm delay để tránh bị block
        time.sleep(0.5)
    
    return all_data

def process_metadata(all_pages_data):
    """Xử lý và trích xuất metadata từ tất cả các trang"""
    metadata_list = []
    total_videos = 0
    
    for page_data in all_pages_data:
        if 'data' not in page_data or 'archives' not in page_data['data']:
            continue
            
        archives = page_data['data']['archives']
        total_videos += len(archives)
        
        for video in archives:
            metadata = {
                "id_video": 'https://www.bilibili.com/video/'+str(video.get('bvid', '')),
                "background_link": video.get('pic', ''),
                "title_video": video.get('title', ''),
                "view_count": video.get('stat', {}).get('view', 0),
                "like_count": video.get('stat', {}).get('like', 0),
                "danmaku_count": video.get('stat', {}).get('danmaku', 0),
                "duration": video.get('duration', 0),
                "pub_date": video.get('pubdate', 0)
            }
            metadata_list.append(metadata)
    
    return metadata_list, total_videos

def main(url):
    user_id, playlist_id = extract_ids(url)

    if not user_id or not playlist_id:
        print("Không thể trích xuất user_id hoặc playlist_id từ URL.")
        return
    
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'priority': 'u=0, i',
        'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Microsoft Edge";v="138"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
        'cookie': "buvid3=5AED4BF6-B8D7-FE85-A6C0-8CBD6B895E2872047infoc; b_nut=1753323372; _uuid=C8492226-D276-9421-A4BD-E5EC1E21E1DA73458infoc; enable_web_push=DISABLE; home_feed_column=4; browser_resolution=1358-650; buvid4=8546B771-875A-FF94-303A-E8FEA6CE2E8D75508-025072410-E4hNoZ1EQDUzrAX5Hd7a5w%3D%3D; rpdid=|(Rl~Ju~lYk0J'u~lJJkJRuu; fingerprint=0ba073ba3e1ba06eb7ec0a0151ea7cb5; buvid_fp_plain=undefined; buvid_fp=0ba073ba3e1ba06eb7ec0a0151ea7cb5; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NTQwMzI1NzcsImlhdCI6MTc1Mzc3MzMxNywicGx0IjotMX0.YdAWLKihvaDnfkq7X6qa2NnX03Y73SmqYa4DnQmOI_0; bili_ticket_expires=1754032517; PVID=1; b_lsid=7B101010A41_198598BA3E8; sid=66l3i9nr; CURRENT_FNVAL=4048",
    }

    # Lấy dữ liệu từ tất cả các trang
    all_pages_data = get_all_pages_data(user_id, playlist_id, headers)
    
    if not all_pages_data:
        print("Không lấy được dữ liệu từ bất kỳ trang nào.")
        return

    # Xử lý metadata
    metadata_list, total_videos = process_metadata(all_pages_data)
    
    if not metadata_list:
        print("Không tìm thấy dữ liệu video hợp lệ.")
        return

    # Lưu vào file JSON
    with open('sourse_video_infor.json', 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu {total_videos} video từ {len(all_pages_data)} trang vào 'sourse_video_infor.json'.")

if __name__ == "__main__":
    url = input('Vui lòng nhập link list Bili : ')
    main(url)

