import os
import sys
import json
import uuid
import time
import socket
import yt_dlp
import ffmpeg
import random
import shutil
import logging
import asyncio
import threading
import subprocess
from glob import glob
from queue import Queue
from pathlib import Path
from google import genai
from g4f.client import Client
from bs4 import BeautifulSoup
from datetime import timedelta
from faster_whisper import WhisperModel
from playwright.sync_api import sync_playwright
from concurrent.futures import ThreadPoolExecutor, as_completed

import googleapiclient.http
import requests.exceptions
import googleapiclient.errors
import http.client as httplib
import googleapiclient.discovery
import google_auth_oauthlib.flow
import google.oauth2.credentials
import google.auth.transport.requests

import whisper
import http.server
import socketserver
import moviepy as mp


from pydub import AudioSegment
import librosa
import numpy as np
from scipy.io.wavfile import write
import soundfile as sf
import requests
import re
import edge_tts

import os
import re
import cv2
import json
import time
import shutil 

import subprocess
import numpy as np
from PIL import Image
import uiautomator2 as u2

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# Cấu hình logging
logging.basicConfig(filename='dub_movie.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_and_use_proxy():
    # 1. Địa chỉ API lấy proxy của bạn (thay bằng link thật của bạn)
    api_url = f"https://app.2proxy.vn/api/proxy.php?key=cb32df521c3a1703fbcb64b745668fe3&sukien=listproxy&ma_don_hang=KIU85842"

    try:
        # Gọi API để lấy thông tin proxy
        response = requests.get(api_url)
        response.raise_for_status() # Kiểm tra nếu lỗi kết nối
        
        # Chuyển dữ liệu JSON trả về thành List
        data = response.json() 
        
        if data and data[0].get("maloi") == 0:
            proxy_info = data[0]
            proxy_str = proxy_info.get("proxy") # Dạng IP:Port:User:Pass
            
            # 2. Tách chuỗi proxy để đưa vào định dạng của thư viện requests
            # Cấu trúc: ip:port:user:pass
            parts = proxy_str.split(':')
            ip, port, user, password = parts[0], parts[1], parts[2], parts[3]
            
            # Định dạng chuẩn cho Requests: http://user:pass@ip:port
            proxy_format = f"http://{user}:{password}@{ip}:{port}"
            return proxy_format
            
        else:
            print(f"[-] Lỗi từ API: {data[0].get('msg', 'Không xác định')}")

    except Exception as e:
        print(f"[!] Có lỗi xảy ra: {e}")

def median(danh_sach):
    danh_sach_sap_xep = sorted(danh_sach)
    n = len(danh_sach_sap_xep)
    if n == 0:
        return None
    if n % 2 == 1:
        return danh_sach_sap_xep[n // 2]
    else:
        giua1 = danh_sach_sap_xep[n // 2 - 1]
        giua2 = danh_sach_sap_xep[n // 2]
        return (giua1 + giua2) / 2

def group_consecutive_times(times, threshold_ms=50):
    times.sort()
    groups = []
    current_group = [times[0]]
    for t in times[1:]:
        if t - current_group[-1] <= threshold_ms:
            current_group.append(t)
        else:
            groups.append(current_group)
            current_group = [t]
    groups.append(current_group)
    return groups
def save_video_douyin(url):
    #while True:
    print ('        LẤY LINK TẢI VIDEO DOUYIN TỪ TRANG TWDOWN.ONLINE ... ')
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'priority': 'u=0, i',
        'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        #'cookie': 's_id=JsS0bHBOOOwNbhutoOnzqygq6uZnqZ76asZkHZtf; fpestid=ug___VKqwt9ta_RROa68QV_FLUYmWhXTFaE8DUZKbvGaoP2ClSvua2i1bHSBvkzg6PhgUQ; __gads=ID=ae1597f5e1f5579c:T=1765787960:RT=1765788511:S=ALNI_MatiEYcqO41QzVTXYjjM-1O_Ztx-Q; __gpi=UID=000011cb2a0356f0:T=1765787960:RT=1765788511:S=ALNI_Mb5EsWitPBPKnb3jaTh3QqIVrs2fg; __eoi=ID=8019400a96b04078:T=1765787960:RT=1765788511:S=AA-AfjZGG7YsfY3H3iPJO9x2tyQs; FCCDCF=%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%224219edc5-2be2-4679-ab3c-fb9298fd30f1%5C%22%2C%5B1765787961%2C421000000%5D%5D%22%5D%5D%5D; FCNEC=%5B%5B%22AKsRol-ue_RLQwfvtkBdT_PFJmspkMkg_t2mA61o2cnnulXrsmQMrnJdxQ_LuTKwvDcW4_QlGLme96Mci9ldg7uyBzQZ8B2N6_-PpszgLjVEztyjOnyyLW_Rs3-ad9q_mftMQ3PytoiXkohnbXOk1cwGgFVEIYBqow%3D%3D%22%5D%5D; XSRF-TOKEN=eyJpdiI6ImtwTFpmYnR6ZU5iRncyb2xPc3duQkE9PSIsInZhbHVlIjoiTFRKSGZrT05NTmo4MGZhUjh6b3BBQm5jbVl2dnhIeG1vOWhRbmdXUlVBWnhvTXZoT2xBTlRyQkNwRTlzd2x3NiIsIm1hYyI6IjQ4YzM1MDdiYmJhZDRjZDdlNDU2ZTU2OGMxNjc5MDhkYWFlZDA2M2IxOWU2ZGE0ZWJjMWU4MjkzNjA5N2E4NzcifQ%3D%3D',
    }
    response = requests.get(f'https://downloader.twdown.online/search?search={url}',
        headers=headers,
    )
    soup=BeautifulSoup(response.text, 'html.parser');link=''
    five_links = soup.find_all('a')
    for i in five_links:
        if (i['href']).find('https://downloader.twdown.online?ref=&title=') >=0:
            link=i['href']
            break   
    #if link != '':
            
    #time.sleep(60)  
    print ('LINK VIDEO DOWLOAD DOUYIN LẤY ĐƯỢC:', link.split('=')[3])
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'priority': 'u=0, i',
        'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'cross-site',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        'cookie': 'fpestid=ug___VKqwt9ta_RROa68QV_FLUYmWhXTFaE8DUZKbvGaoP2ClSvua2i1bHSBvkzg6PhgUQ; s_id=4KBpnlWgB4YIvfNWkIaK68YHyl3IqKbqZ8Z0Ze2s; __gads=ID=ae1597f5e1f5579c:T=1765787960:RT=1765862708:S=ALNI_MatiEYcqO41QzVTXYjjM-1O_Ztx-Q; __gpi=UID=000011cb2a0356f0:T=1765787960:RT=1765862708:S=ALNI_Mb5EsWitPBPKnb3jaTh3QqIVrs2fg; __eoi=ID=8019400a96b04078:T=1765787960:RT=1765862708:S=AA-AfjZGG7YsfY3H3iPJO9x2tyQs; FCCDCF=%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%224219edc5-2be2-4679-ab3c-fb9298fd30f1%5C%22%2C%5B1765787961%2C421000000%5D%5D%22%5D%5D%5D; FCNEC=%5B%5B%22AKsRol9pfppTLeFn8E3KUTOaj6HPbRHQD26WVzJV_S5JKk4TkUh91YQ6Lg74ywUpu_kjAiXDHpKQJgMp0maR6bOnL0787wO9HF41V3Kx-Af0HHTmrjisb76M_hXhh6eGQaIUBRI2RcG8iAQzq8kWHtUWGJtav8060g%3D%3D%22%5D%5D; XSRF-TOKEN=eyJpdiI6IlwveE1Ld0IxaVpSTkRyNjNkbXREUWVnPT0iLCJ2YWx1ZSI6IjBTWTZ4bzI0b3k5cTgxTmRlR3VVVjJyd05rWEhyYWlFYmJLa0didjArSTA0UG1lUGR6SitON05UcXZEWmt4UUkiLCJtYWMiOiI1NTI5MTM2MTg1ZTY4YzUzMTQ1MmQ4OGQwNWJkYWMwNWQ3YWM1OGY3NjFiMGQ1NjNkMTdmZWU3NTRkZjAwZTY5In0%3D',
    }

    params = {
        'url': link.split('=')[3],
    }

    response = requests.get('https://downloader.twdown.online/load_url', params=params, headers=headers).text
    return response
def split_continuous(lst):
    result = []
    current = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            current.append(lst[i])
        else:
            result.append(current)
            current = [lst[i]]
    result.append(current)
    return result

def setup_directories(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    # Đảm bảo quyền ghi trên Linux
    os.chmod(temp_dir, 0o777)

def extract_audio_ffmpeg(video_path, start_time, end_time, output_audio_path):
    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", video_path,
        "-vn",
        "-acodec", "copy",
        output_audio_path
    ]
    subprocess.run(cmd, check=True)

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            filtered_data = [item for item in json.load(f)['segments'] if item is not None]
            return {'segments': filtered_data}
    else:
        return False

def save_checkpoint(checkpoint_file, metadata_list):
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4, ensure_ascii=False)

def extract_audio_from_video(video_path, temp_dir, duration_time):
    return split_video(video_path, duration_time, temp_dir)

def filter_audio(audio_files_path):
    y, sr = librosa.load(audio_files_path)
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_full,
        aggregate=np.median,
        metric='cosine',
        width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)
    mask_vocals = S_full - S_filter
    mask_background = S_filter
    S_foreground = mask_vocals * phase
    S_background = mask_background * phase
    y_vocals = librosa.istft(S_foreground)
    y_background = librosa.istft(S_background)
    sf.write('output/vocals.wav', y_vocals, sr)
    sf.write('output/background.wav', y_background, sr)

def srt_time_to_seconds(srt_time: str) -> float:
    try:
        time_part, ms_part = srt_time.split(',')
        hours, minutes, seconds = map(int, time_part.split(':'))
        milliseconds = int(ms_part)
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        return total_seconds
    except :
        print (srt_time)
def format_srt_time(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def filter_srt_detail(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    metadata_list = []
    i = 0
    while i < len(lines):
        if ' --> ' in lines[i]:
            detail = lines[i].split(' --> ')
            start_time, end_time = detail[0], detail[1]
            try:
                offset_sec = int(os.path.basename(srt_path).split('_')[1].split('.')[0])
            except:
                offset_sec = 0    
            try:
                metadata = {
                    "start_time": str(srt_time_to_seconds(start_time) + offset_sec),
                    "end_time": str(srt_time_to_seconds(end_time) + offset_sec),
                    "text": lines[i+1].replace('\n',''),
                    "speed_of_speech": "2007 WPM",
                    "status": "completed"
                }
                metadata_list.append(metadata)
            except:
                None
        i += 1
    return metadata_list
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milis = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milis:03}"
def transcript_video(audio_file_path, checkpoint_file, temp_dir, srt_subtitle_output_path, metadata_list, segment_index=0, MODEL=WhisperModel("large-v3", device="auto",cpu_threads=10,num_workers=5, compute_type="int8")):
    setup_directories(temp_dir)
    segment_id = str(uuid.uuid4())
    for audio in audio_file_path:
        print('EXTRACTING SPEECH CONTENT:', audio)
        srt_filename = os.path.basename(audio).replace('.aac', '.srt')
        srt_path = os.path.join(srt_subtitle_output_path, srt_filename)
        segments, info = MODEL.transcribe(audio, beam_size=5)  

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                start_str = format_time(segment.start)
                end_str = format_time(segment.end)
                text = segment.text.strip()
                
                print(f"[{start_str} -> {end_str}] {text}")
                
                f.write(f"{i}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{text}\n\n")
            f.close()    
        metadata_list.extend(filter_srt_detail(srt_path))
        save_checkpoint(checkpoint_file, {"segments": metadata_list})    
    return metadata_list
    """
    finally:
        if isinstance(audio_file_path, list):
            for path in audio_file_path:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logging.info(f"Deleted temp file: {path}")
                    except Exception as e:
                        logging.warning(f"Could not delete temp file {path}: {e}")
    """
def get_video_duration(video_path):
    """Lấy thời lượng video (giây)"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None

def fix_to_srt(input_text):
# Bước 1: Tách số thứ tự ra khỏi dòng thời gian
    # Tìm: (Số) (Thời gian) -> Thay bằng: (Số)\n(Thời gian)
    text = re.sub(r"^(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3} -->)", r"\1\n\2", input_text, flags=re.MULTILINE)
    
    # Bước 2: Tách nội dung ra khỏi dòng thời gian (nếu nó dính liền phía sau)
    # Tìm: (Thời gian) (Nội dung) -> Thay bằng: (Thời gian)\n(Nội dung)
    text = re.sub(r"(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\s+(?!\n)", r"\1\n", text)
    
    # Bước 3: Chuẩn hóa khoảng cách dòng
    # Xóa dòng trống thừa và đảm bảo mỗi khối cách nhau đúng 1 dòng trống
    blocks = []
    # Chia text thành các khối dựa trên số thứ tự ở đầu dòng
    raw_blocks = re.split(r'\n(?=\d+\n)', text.strip())
    
    for b in raw_blocks:
        lines = [line.strip() for line in b.split('\n') if line.strip()]
        if len(lines) >= 3:
            # Đúng chuẩn: Dòng 1 (Số), Dòng 2 (Time), Dòng 3+ (Nội dung)
            blocks.append("\n".join(lines))
        elif len(lines) == 2:
            # Trường hợp thiếu nội dung hoặc lỗi khác (dự phòng)
            blocks.append("\n".join(lines))
            
    return "\n\n".join(blocks)

def split_video(video_file, segment_duration, temp_dir):
    total_duration = get_video_duration(video_file)
    segments = []
    for i in range(0, int(total_duration), segment_duration):
        out_file = os.path.join(temp_dir, f'segment_{i}.aac')
        cmd = (
            f"ffmpeg -i {video_file} -ss {i} -t {segment_duration} "
            f"-vn -acodec copy {out_file} -y"
        )
        os.system(cmd)
        segments.append(out_file)
    return segments

def run_threads(audio_files, checkpoint_file, temp_dir, srt_subtitle_output_path, source_language, metadata_list, num_threads):
    chunks = [audio_files[i::num_threads] for i in range(num_threads)]
    threads = []
    for chunk in chunks:
        t = threading.Thread(target=transcript_video, args=(chunk, checkpoint_file, temp_dir, srt_subtitle_output_path, metadata_list))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

def extract_transcript(input_video_path, output_dir, output_json_path, source_language, metadata_list):
    temp_dir = "temp_segments"
    srt_subtitle_output_path = "srt"
    duration_time = 28
    
    setup_directories(temp_dir)
    setup_directories(srt_subtitle_output_path)
    logging.info(f"Processing video: {input_video_path}")
    
    try:
        audio_files_path = extract_audio_from_video(input_video_path, temp_dir, duration_time)
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        audio_files_path = []

    checkpoint = load_checkpoint(output_json_path)
    if check_volume_of_file(output_json_path, volume=1024):
        completed_segments = []
    else:
        completed_segments = [m["end_time"] for m in checkpoint["segments"] if m["status"] == "completed"]
        list_remove_file = []
        for file_path in audio_files_path:
            num_path = float(os.path.basename(file_path).split('_')[1].split('.')[0])
            if num_path == 0:
                list_remove_file.append(file_path)
            else:
                for value in completed_segments:
                    if float(value) >= num_path and num_path + duration_time >= float(value):
                        try:
                            list_remove_file.append(file_path)
                            break
                        except:
                            continue
        audio_files_path = [f for f in audio_files_path if f not in list_remove_file]

    try:
        video = mp.VideoFileClip(input_video_path)
        duration = video.duration
        video.close()
        logging.info(f"Video duration: {duration:.1f} seconds")
    except Exception as e:
        logging.error(f"Error loading video: {e}")
        print(f"Error loading video: {e}")
        return None

    if round(duration/28) == len(completed_segments):
        return None
    
    setup_directories(os.path.join(output_dir, 'original_voice'))
    output_path = os.path.join(output_dir, 'original_voice', 'main_stream.wav')
    
    try:
        run_threads(audio_files_path, output_json_path, temp_dir, srt_subtitle_output_path, source_language, metadata_list, num_threads=20)
        print("Completed Speech-to-Text Conversion")
        output_json = {"segments": metadata_list}
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=4, ensure_ascii=False)
        logging.info(f"Transcript saved to: {output_json_path}")
        print(f"Transcript saved to: {output_json_path}")
        return output_json
    except Exception as e:
        logging.error(f"Processing error: {e}")
        print(f"Processing error: {e}")
#-------------------------------------------------
class GeminiTranslationManager:
    def __init__(self, api_keys, segment_groups, target_language, output_sub_dir):
        self.api_keys = api_keys.copy()
        self.segment_groups = segment_groups
        self.target_language = target_language
        self.output_sub_dir = output_sub_dir
        self.available_keys = Queue()
        self.used_keys = set()
        self.failed_keys = set()
        self.translated_file_paths = []
        
        # Khởi tạo queue với tất cả API keys
        for key in api_keys:
            self.available_keys.put(key)
    
    def get_available_key(self):
        """Lấy API key từ Queue để đảm bảo xoay vòng công bằng"""
        try:
            # Lấy key từ hàng đợi, không block để tránh treo luồng
            # timeout=2 giây để chờ nếu các luồng khác đang giữ key
            return self.available_keys.get(timeout=2)
        except:
            # Nếu tất cả key đều bị lỗi hoặc đang bận
            if len(self.failed_keys) == len(self.api_keys):
                raise Exception("Tất cả API keys đã thất bại hoặc bị khóa!")
            return None

    def mark_key_success(self, api_key):
        """Đánh dấu key sử dụng thành công"""
        self.used_keys.add(api_key)
        if api_key not in self.failed_keys:
            self.available_keys.put(api_key)

    def mark_key_failed(self, api_key):
        """Đánh dấu key bị lỗi"""
        self.failed_keys.add(api_key)
        logging.warning(f"API key failed: {api_key[:10]}...")

    def translate_segment_with_retry(self, segment):
        """Dịch một segment với cơ chế retry tự động"""
        max_retries = len(self.api_keys)
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            api_key = self.get_available_key()
            if api_key is None:
                logging.error("No available API keys")
                return None
            
            try:
                result = self.translate_segment(segment, api_key)
                print (result)
                if result is not None:
                    self.mark_key_success(api_key)
                    return result
                else:
                    self.mark_key_failed(api_key)
                    
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed for {segment}: {str(e)}")
                self.mark_key_failed(api_key)
                time.sleep(retry_delay * (attempt + 1))
        
        logging.error(f"Failed to translate {segment} after {max_retries} attempts")
        return None

    def translate_segment(self, segment, api_key):
        """Dịch một segment cụ thể"""
        try:
            with open(segment, "r", encoding="utf-8") as file:
                srt_content = file.read()
            
            response = self.translated_with_gemini(api_key, srt_content)            
            base_name = os.path.basename(segment)
            try:
                segment_number = (base_name.split('_')[1]).replace('.srt','')
            except IndexError:
                segment_number = base_name.split('.')[0]
            
            file_name = f'segment_{segment_number}.srt'
            final_sup_text_path = os.path.join(self.output_sub_dir, file_name)
            if response is not None:
                with open(final_sup_text_path, 'w', encoding="utf-8") as f:
                    f.write(response)                
                return final_sup_text_path
            
        except Exception as e:
            logging.error(f"Error translating segment {segment}: {str(e)}")
            return None

    def translated_with_gemini(self, api_key, srt_content):
        """Gọi API Gemini để dịch"""
        client = genai.Client(api_key=api_key)
        try:
            prompt = (
                """
                -Bạn là một công cụ dịch thuật. Nhiệm vụ của bạn đoạn có nội dung là về hướng dẫn làm slime (*chất nhờ ma quái) từ các dụng cụ tại nhà cho các bé yêu và yêu cầu cụ thể cho cách dịch là:
                -0. **Phong cách nhẹ nhàng, trầm ấm**
                -1. **Không được giữ lại bất kỳ ký tự gốc nào của văn bản gốc (ví dụ tiếng Trung, tiếng Anh...) trong kết quả.**
                -2. **Một số tên được dịch không hay bạn có thể biến tên nhân vật theo phong cách Trung Hoa cổ trang, sử dụng cấu trúc "Tiểu + [Tên Nhân Vật]" . Mỗi tên cần thể hiện tính cách, ngoại hình, hoặc vai trò của nhân vật.
                -3. **Né các từ vi phạm trong chính sách của tiktok**
                -4. **Tôi nhấn mạnh lại một điểm phải giữ nguyên định dạng gốc . Ví dụ về một lỗi mà bạn thường sai  00:01 nhưng phải điền là 00:00:01,000 .Hãy chú ý vào những lỗi nhỏ đó.
                -5. **Hãy giữ nguyên định dạng tệp .srt. Không thêm bất kỳ câu dẫn hay giải thích nào, chỉ trả về nội dung của tệp. Ưu tiên sử dụng các từ đồng nghĩa ngắn nhất phù hợp với ngữ cảnh. Không thêm các danh từ riêng đặc biệt (ngoại trừ ‘tôi’ và ‘bạn’)
                -6. **Hãy dịch sao cho số lượng từ tiếng Việt tương đương hoặc ít hơn số từ tiếng gốc để đảm bảo tốc độ nói tự nhiên.**
                -7. **Nếu bản dịch tiếng Việt quá dài so với câu gốc, yêu cầu  dịch ngắn gọn hơn.**
                -LƯU Ý. **Có số chổ là do cách phát âm & những từ đồng nghĩa làm cho khi dịch câu trở nên sai các phương pháp về từ như sai logic ,sai ngữ pháp , sai cách dùng từ .Bạn hãy xem và chỉnh sữa những điểm đó lại bằng trí tuệ của bạn.Không cần phải chú thích gì cả 
        
                Kết quả dịch (giữ nguyên định dạng thay thế văn bản gốc bằng văn bản đã dịch không thêm gì khác):
                """
            )            
            for name_model in ['gemini-1.5-flash','gemini-2.5-pro','gemini-2.0-flash-lite','gemini-2.5-flash','gemini-2.5-flash-lite',"gemini-3-flash-preview"]:
                try:
                    response = client.models.generate_content(
                        model=name_model,
                        contents=[srt_content, prompt]
                    ).text
                    if response is not None:
                        if response.find("'''") >=0:
                            None
                        else:
                            return response   
                    print (api_key,' ---',name_model)                                                                         
                except Exception as e:                
                    logging.warning(f"Model {name_model} failed: {str(e)} -- apied {str(api_key)}")
                    continue     
            return None        
        except Exception as e:
            logging.error(f"Gemini API call failed: {str(e)}")
            return None          
    def run_translation(self):
        print ("""Chạy dịch thuật với quản lý API keys thông minh""")
        #logging.info(f"Starting translation with {len(self.api_keys)} API keys for {len(self.segment_groups)} segments")
        # Số luồng tối đa bằng số segment groups
        max_workers = min(len(self.segment_groups), 5)  # Giới hạn 20 luồng để tránh quá tải
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Gửi tất cả tasks
            future_to_segment = {
                executor.submit(self.translate_segment_with_retry, segment): segment 
                for segment in self.segment_groups
            }
            # Xử lý kết quả
            for future in as_completed(future_to_segment):
                segment = future_to_segment[future]
                try:
                    result = future.result()
                    if result:
                        self.translated_file_paths.append(result)
                        logging.info(f"Successfully translated: {segment}")
                    else:
                        logging.error(f"Failed to translate: {segment}")
                except Exception as e:
                    logging.error(f"Exception occurred for {segment}: {str(e)}")
        logging.info(f"Translation completed. Success: {len(self.translated_file_paths)}/{len(self.segment_groups)}")
        return self.translated_file_paths
def translate_with_gemini(api_keys, segment_groups, target_language, output_sub_dir):
    """Function chính để gọi từ bên ngoài"""
    try:
        manager = GeminiTranslationManager(api_keys, segment_groups, target_language, output_sub_dir)
        return manager.run_translation()
    except Exception as e:
        print (e)
        logging.error(f"Translation failed: {str(e)}")
        return []
#-------------------------------------------------

def speed_up_video(input_path, speed, output_path):
    # --- KHỐNG CHẾ CỨNG KHOẢNG AN TOÀN ---
    # Không để speed nhỏ hơn 0.5 (tránh lỗi FFmpeg)
    # Không để speed lớn hơn 4.0 (nhanh quá cũng không nghe được gì)
    safe_speed = max(1, min(4, speed))
    
    if safe_speed > 2.0:
        filter_str = f"atempo=2.0,atempo={safe_speed/2:.2f}"
    elif safe_speed < 0.5:
        # Đoạn này thực ra safe_speed đã là 0.5 nhờ hàm max ở trên
        # Nhưng viết lại cho chắc chắn
        filter_str = "atempo=0.5"
    else:
        filter_str = f"atempo={safe_speed:.2f}"

    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-y',
            '-filter:a', filter_str,
            '-vn',
            output_path
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error tại {input_path}: {e.stderr.decode()}")
        # Nếu vẫn lỗi thì copy file gốc qua để không làm hỏng chuỗi xử lý video sau này
        import shutil
        shutil.copy(input_path, output_path)

def get_duration(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return 0.0 # Trả về 0 thay vì báo lỗi
    
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return 0.0

def process_segment_dubbing(start_time, end_time, segment_index, translated_text, target_language, temp_dir, api_keys, translate_mode):
    os.makedirs(temp_dir, exist_ok=True)
    duration_ms = ((float(end_time) - float(start_time)) * 1000)
    final_audio_path = os.path.join(temp_dir, f"tts_{start_time}.mp3")
    
    try:
        if translated_text != '':
            final_audio_path = os.path.join(temp_dir, f"tts_{segment_index}.mp3")
            if translate_mode == 'edge_tts':
                try:
                    asyncio.run(text_api_to_speech(translated_text, final_audio_path, duration_ms))

                except Exception as e:
                    logging.error(f"Edge TTS error: {e}")
            elif translate_mode == 'fpt' or translate_mode == 'vclip':  
                #translated_text=translated_text.replace(' ','')              
                fpt_tts(translate_mode,translated_text, final_audio_path, duration_ms/1000,  api_keys, speed='1')  
        else:
            AudioSegment.silent(duration=duration_ms).export(final_audio_path, format="mp3")
        
        return {
            "start_time": start_time,
            "end_time": end_time,
            "output_path": final_audio_path,
            "status": "completed"
        }
    except Exception as e:
        logging.error(f"Dubbing error: {e}")
        return None

def detect_langue(audio_path, model_AI='base'):
    model = whisper.load_model(model_AI)
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    return detected_lang

def clear_folder(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def cut_video_clip_30s(input_file, output_file, duration=28):
    command = [
        "ffmpeg",
        "-i", input_file,
        "-y",
        "-t", str(duration),
        "-c", "copy",
        output_file
    ]
    try:
        subprocess.run(command, check=True)
        print("Video clipped successfully.")
    except subprocess.CalledProcessError as e:
        print("Error:", e)

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def check_volume_of_file(file_audio_path, volume):
    if os.path.exists(file_audio_path):
        file_size = os.path.getsize(file_audio_path)
        return int(file_size) < volume
    return True

async def generate_tts_with_pitch_and_rate(pitch, output_path, text, voice, duration_ms):
    temp_base_path = output_path.replace('.mp3', '_base.mp3') 
    
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, pitch=pitch)
        await communicate.save(temp_base_path)
        
        # Lấy thời lượng thực tế của file AI vừa đọc
        duration_seconds = get_duration(temp_base_path)
        # Thời lượng tối đa cho phép (ms chuyển sang s)
        target_duration_seconds = float(duration_ms) / 1000 
        
        # CƠ CHẾ QUAN TRỌNG: 
        # Nếu AI đọc dài hơn thời gian cho phép (nói không kịp)
        if duration_seconds > target_duration_seconds:
            # Tính tốc độ cần thiết để ép nó vừa khít
            calculated_speed = duration_seconds / target_duration_seconds
            # Thêm 5% an toàn để chắc chắn không chạm vào đoạn sau
            calculated_speed *= 1.05 
            print(f"⚠️ Nói không kịp! Đang ép tốc độ lên: {calculated_speed:.2f}x")
        else:
            # Nếu AI đọc nhanh hơn, giữ tốc độ 1.0 (hoặc tùy chỉnh nếu muốn đọc chậm lại)
            calculated_speed = 1.0
            
        # Giới hạn speed tối đa (ví dụ 3.0x để tránh tiếng bị biến dạng quá mức)
        final_speed = min(3.0, calculated_speed)
        
        speed_up_video(temp_base_path, final_speed, output_path)
                
    except Exception as e:
        print(f"Lỗi TTS: {str(e)}")
    finally:
        if os.path.exists(temp_base_path):
            os.remove(temp_base_path)

async def text_api_to_speech(translated_text, output_path, duration_ms, gender='Female'):
    print('***************************************\n', translated_text, output_path)
    voice = "ja-JP-NanamiNeural" if gender == 'Female' else "ja-JP-KeitaNeural"
    #voice = "vi-VN-HoaiMyNeural" if gender == 'Female' else "vi-VN-NamMinhNeural"
    pitch = "-0Hz"
    await generate_tts_with_pitch_and_rate(pitch, output_path, translated_text, voice, duration_ms/1000)

def ai_vclip_voice(api_key,text):
    print ('CONTENT LÀ',text)
    headers = {
        'Authorization': 'Bearer '+str(api_key),
        'Content-Type': 'application/json',
    }

    json_data = {
        'method': 'ttsLongText',
        'input': {
            'text': text,
            'userVoiceId': '8VXsCLxU7Pn55ADXQc6sAb',
            'speed': 1.0,
        },
    }
    while True:
        response = json.loads(requests.post('https://api-tts.vclip.io/json-rpc', headers=headers, json=json_data).text)
        try:
            print (text,response['error'])
        except:
            break   
    print (text,response)             
    YOUR_EXPORT_ID = response['result']['projectExportId']
    json_data = {
        'method': 'getExportStatus',
        'input': {
            'projectExportId': YOUR_EXPORT_ID,
        },
    }
    while True:
        response = json.loads(requests.post('https://api-tts.vclip.io/json-rpc', headers=headers, json=json_data).text)
        print (response)
        if response['result']['state'] == 'completed':
            return response['result']['url']    

def fpt_tts(translate_mode,text, output_path, duration_ms, api_keys, speed, voice_name='banmai'):#banmai#minhquang
    if len(text) <3:
        text = str(text)+' !'
    while True:
        if translate_mode == 'fpt':
            for api_key in api_keys:
                url = 'https://api.fpt.ai/hmi/tts/v5'    
                headers = {
                    'api-key': api_key,
                    'speed': speed,
                    'voice': voice_name
                }
                response = requests.post(url, data=text.encode('utf-8'), headers=headers)
                print (response.text)
                if response.status_code == 200:
                    url = json.loads(response.text)['async']
                    with open ('url_fpt_out_put_backurl.txt','a') as f:
                        f.write(output_path+'_'+url+'_'+str(duration_ms)+'\n');f.close()
                        return
        elif translate_mode == 'vclip':            
            for api_key in api_keys:
                url = ai_vclip_voice(api_key,text)  
                print (url)              
                with open ('url_fpt_out_put_backurl.txt','a') as f:
                    f.write(output_path+'_'+url+'_'+str(duration_ms)+'\n');f.close()
                    return      
def thread_saving_video_fpt(detail,type_tts):
    print ('Lưu video audio fpt or vclip về máy',type_tts,type(type_tts))
    if type_tts =='fpt':
        print ('Vô')
        try:
            parts = detail.strip().split('_')
            if len(parts) < 4: return
            
            output_path = f"{parts[0]}_{parts[1]}"
            url = parts[2]
            target_duration_ms = float(parts[3])
            
            out_put_base = output_path
            os.makedirs(os.path.dirname(out_put_base), exist_ok=True)

            # Tải file từ server FPT
            for i in range(15): # Tăng số lần thử lên 15
                response = requests.get(url, stream=True, timeout=20)
                if response.status_code == 200:
                    with open(out_put_base, 'wb') as f:
                        f.write(response.content)
                    
                    if os.path.exists(out_put_base) and os.path.getsize(out_put_base) > 500:
                        break
                time.sleep(3) # Đợi server FPT render file xong

            # Sau khi tải xong, tiến hành ép tốc độ
            duration_seconds = get_duration(out_put_base)
            if duration_seconds == 0:
                print(f"❌ File {out_put_base} hỏng hoặc không tải được.")
                return

            target_duration_seconds = target_duration_ms / 1000
            calculated_speed = duration_seconds / target_duration_seconds
            
            # FFmpeg filter atempo tối đa 2.0, nên nếu > 2.0 phải nối chuỗi
            speed_up_video(out_put_base, calculated_speed, output_path)
            print(f"✅ Thành công: {output_path}")

        except Exception as e:
            print(f"🔥 Lỗi nghiêm trọng tại thread_saving: {e}")
    elif type_tts=='vclip':
        try:
            parts = detail.strip().split('_')
            if len(parts) < 4: return
            
            output_path = f"{parts[0]}_{parts[1]}"
            url = f"{parts[2]}_{parts[3]}"
            target_duration_ms = float(parts[4])
            
            out_put_base = output_path
            os.makedirs(os.path.dirname(out_put_base), exist_ok=True)

            # Tải file từ server FPT
            for i in range(15): # Tăng số lần thử lên 15
                response = requests.get(url, stream=True, timeout=20)
                if response.status_code == 200:
                    with open(out_put_base, 'wb') as f:
                        f.write(response.content)
                    
                    if os.path.exists(out_put_base) and os.path.getsize(out_put_base) > 500:
                        break
                time.sleep(3) # Đợi server FPT render file xong

            # Sau khi tải xong, tiến hành ép tốc độ
            duration_seconds = get_duration(out_put_base)
            if duration_seconds == 0:
                print(f"❌ File {out_put_base} hỏng hoặc không tải được.")
                return

            target_duration_seconds = target_duration_ms / 1000
            calculated_speed = duration_seconds / target_duration_seconds
            
            # FFmpeg filter atempo tối đa 2.0, nên nếu > 2.0 phải nối chuỗi
            speed_up_video(out_put_base, calculated_speed, output_path)
            print(f"✅ Thành công: {output_path}")

        except Exception as e:
            print(f"🔥 Lỗi nghiêm trọng tại thread_saving: {e}")                    
def download_and_export_dual_formats(video_id, json_output_path, srt_dir='srt'):
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    temp_vtt_name = f'temp_{video_id}'
    
    if not os.path.exists(srt_dir):
        os.makedirs(srt_dir)

    ydl_opts = {
        'writeautomaticsub': True,
        'subtitleslangs': ['vi'],
        'skip_download': True,
        'outtmpl': temp_vtt_name,
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        file_path = f"{temp_vtt_name}.vi.vtt"
        if not os.path.exists(file_path): return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Biến đổi nội dung: Tìm các đoạn có chứa mốc thời gian từ nội bộ <hh:mm:ss.ms>
        blocks = content.split('\n\n')
        final_segments = []
        seen_texts = set()

        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 3: continue
            
            # Lấy văn bản thô (loại bỏ lặp dòng bằng cách chỉ lấy dòng cuối cùng của block)
            raw_text = lines[-1]
            # Loại bỏ các thẻ định dạng <c>
            clean_text = re.sub(r'</?c>', '', raw_text)
            
            # Tìm tất cả mốc thời gian của từng từ trong dòng đó
            word_timestamps = re.findall(r'<(\d{2}:\d{2}:\d{2}\.\d{3})>', clean_text)
            # Văn bản thuần túy sau khi xóa hết thẻ thời gian
            pure_text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', clean_text).strip()

            if pure_text and pure_text not in seen_texts:
                # Nếu dòng này có mốc thời gian từ nội bộ, đó mới là thời gian thực
                if word_timestamps:
                    start_time = word_timestamps[0].replace('.', ',')
                    end_time = word_timestamps[-1].replace('.', ',')
                else:
                    # Nếu không có, lấy thời gian của Block nhưng chuẩn hóa lại
                    time_match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', lines[1])
                    if time_match:
                        start_time = time_match.group(1).replace('.', ',')
                        end_time = time_match.group(2).replace('.', ',')
                    else: continue

                final_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': pure_text
                })
                seen_texts.add(pure_text)

        # Xuất file JSON và SRT
        json_list = []
        srt_lines = []
        
        for idx, seg in enumerate(final_segments, 1):
            # Tính giây cho JSON
            h, m, s_ms = seg['start'].replace(',', '.').split(':')
            start_sec = round(int(h)*3600 + int(m)*60 + float(s_ms), 2)
            h, m, s_ms = seg['end'].replace(',', '.').split(':')
            end_sec = round(int(h)*3600 + int(m)*60 + float(s_ms), 2)

            json_list.append({
                "start_time": str(start_sec),
                "end_time": str(end_sec),
                "text": seg['text'],
                "speed_of_speech": "2007 WPM",
                "status": "completed"
            })
            srt_lines.append(f"{idx}\n{seg['start']} --> {seg['end']}\n{seg['text']}\n")

        # Lưu file
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump({"segments": json_list}, f, indent=4, ensure_ascii=False)

        with open(os.path.join(srt_dir, f"{video_id}.srt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_lines))

        os.remove(file_path)
        print("Đã tái cấu trúc phụ đề dựa trên mốc thời gian thực của từ.")

    except Exception as e:
        print(f"Lỗi: {e}")

def dub_movie(input_video_path, output_dir, api_keys, source_language, target_language, full_option,target_id_video_bil,type_tts='fpt'):
    temp_dir = "temp_segments"
    metadata_list = []
    output_sub_dir = "sub"
    tts_forder_save_path = "tts"
    
    checkpoint_transcript_file = f"checkpoint_transcript_{os.path.basename(input_video_path)}.json"
    checkpoint_dub_file = f"checkpoint_dub_{os.path.basename(input_video_path)}.json"
    output_json_path = os.path.join(output_dir, "transcript.json")
    translated_json_path = os.path.join(output_dir, "translated_transcript.json")
    with open('Api_key/gemini_api_key.txt', 'r') as f:
            api_keys = [line.strip() for line in f if line.strip()]
    setup_directories(temp_dir)
    setup_directories(output_dir)
    setup_directories(output_sub_dir)
    #if target_id_video_bil.find('https://www.youtube.com') >=0:
    #    download_and_export_dual_formats(target_id_video_bil.replace('https://www.youtube.com/shorts/',''), translated_json_path)
    if True:
        print("Step 1: Extracting speech content...")
        if not check_volume_of_file(checkpoint_transcript_file, 2048):
            metadata_list = load_checkpoint(checkpoint_transcript_file)['segments']        
        extract_transcript(input_video_path, output_dir, checkpoint_transcript_file, source_language, metadata_list)
    print("Step 2: Translating text to Vietnamese...")
    subed_srt_path = glob(os.path.join(output_sub_dir, '*.srt'))
    srt_files = glob(os.path.join('srt', '*.srt'))
    metadata_list = []    
    for i in subed_srt_path:
        try:
            srt_files.remove(i.replace('sub', 'srt'))
        except:
            None    
    print ('DỊCH SỐ FILE SRT CẦN DỊCH:', len(srt_files))
    if srt_files:
        translate_with_gemini(api_keys, srt_files, target_language, output_sub_dir)   
    for srt_path in glob(os.path.join(output_sub_dir, '*.srt')):
        metadata_list.extend(filter_srt_detail(srt_path))   
    save_checkpoint(translated_json_path, {"segments": metadata_list})
    logging.info(f"Translated transcript saved to: {translated_json_path}")
    print(f"Successfully translated from {source_language} to {target_language}")
    if full_option:
        print("Step 3: Generating TTS files for segments...")
        checkpoint = load_checkpoint(translated_json_path);checkpoint_dub=[]        
        metadata_list = checkpoint["segments"]
        list_start_time_complete = []        
        if os.path.exists(checkpoint_dub_file):
            checkpoint_dub=load_checkpoint(checkpoint_dub_file)['segments']
            list_start_time_complete = [i['start_time'] for i in checkpoint_dub if i['status'] == 'completed']        
        metadata_list = [i for i in metadata_list if i['start_time'] not in list_start_time_complete];api_keys=[]
        if type_tts == 'fpt':
            api_keys=['NUkcBOc2wujsRErJ3PCGXmuZmt9hkIt2','Bn5q6fvXe5kaao1RaE1xKCfyU9sfRc3s','5OEDPYSmHyKEun04GRRhy5rDWDmvbpi4','jZZrH1LJtI3TlpaLiskcs4imdW3ZEWcz','90Rc9Olnrmpe5WE9aNsXaK7AeMSvCNd6','lJkoOxVwZ54pbhU9EVZOa2WO9NviS2jr','bES2wLGABlYCFA0xUDsbiDqCYEWdtnDh','n9rKtRYQWtT39nCBil4o9JsQsvZEyRDP','WDJxtVTv3EOUbpeXWszOd6WkxWs9LGJQ','pjLFlJzmN4yaYcZPHP2wwMMIx0Pn5lBh','jJ3vZAnoo6l93yoA58g4UYqUKgB3dCrA','pRpfjNY1nBoWbBDanWJQ3ZOzR7LhuThB']
        elif type_tts == 'vclip':
            api_keys=['sk_live_fynsdhLsb9P7q2LcWJvkrdmoc04tGoRp']
        max_workers = min(os.cpu_count() or 2, 2)
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_segment = {
                executor.submit(
                    process_segment_dubbing,
                    segment["start_time"],
                    segment["end_time"],
                    segment["start_time"],
                    segment["text"],
                    target_language,
                    tts_forder_save_path,
                    api_keys,
                    type_tts
                ): segment
                for segment in metadata_list
            }
        # Đọc danh sách URL nếu sử dụng type_tts là 'fpt'
        if len(metadata_list)!=0:
            if type_tts == 'fpt' or type_tts == 'vclip':
                with open('url_fpt_out_put_backurl.txt', 'r') as f:
                    list_url_fpt = f.readlines() 
                with ThreadPoolExecutor(max_workers=20) as executor:
                    for url in list_url_fpt:
                        executor.submit(thread_saving_video_fpt, url, type_tts)          
        # Kết hợp metadata từ các kết quả
        new_metadata = []
        for future in as_completed(future_to_segment):
            segment = future_to_segment[future]
            try:
                metadata = future.result()
                if metadata:
                    new_metadata.append(metadata)
            except Exception as e:
                print(f"Error processing segment: {e}")
        
        # Kết hợp metadata mới với dữ liệu cũ
        new_metadata.extend(checkpoint_dub)

        # Lưu lại kết quả vào file checkpoint
        save_checkpoint(checkpoint_dub_file, {"segments": new_metadata})
def increase_audio_volume(input_file, output_file, volume_factor, audio_quality=5):
    try:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        command = [
            'ffmpeg',
            '-i', input_file,
            '-af', f'volume={volume_factor}',
            '-c:v', 'copy',
            '-c:a', 'libmp3lame',
            '-q:a', str(audio_quality),
            '-y',
            output_file
        ]
        
        subprocess.run(command, check=True)
        print(f"Processing successful! File saved to: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def run_youtube_upload(file_path,title,description,keywords,category,privacy):
    # Các tham số từ command line
    file_path = '--file=\"video.mp4\"'
    title = '--title=\"[ LẬP TRÌNH  ] SỬ LÝ FILE ÂM THANH WAV - ANALYZE AUDIO FILE \"'
    description = '--description=\"Sử lý âm thanh một điều tuyệt vời để khai thác thông tin & trực quan hóa dữ liệu điều này .\"'
    keywords = '--keywords=\"lịch sử thế giới , kể chuyện đêm khuya , lịch sử trung hoa , trung hoa dân quốc , Trung Quốc , người kể chuyện , Truyện Đêm Khuya\"'
    category = '--category=\"22\"'
    privacy = '--privacyStatus=\"public\"'
    
    # Tạo command
    command = [
        "python", "run.py",
        file_path,
        title,
        description,
        keywords,
        category,
        privacy
    ]
    
    try:
        # Chạy subprocess
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        
        print("Command executed successfully!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return e.returncode
    except FileNotFoundError:
        print("Error: python or run.py not found")
        return 1

def convert_detail(segments, srt_content, counter):
    for segment in segments:
        total_chars = 0
        text = segment['text'].strip()
        start = float(segment['start_time'])
        end = float(segment['end_time'])
        
        if text != '':
            text_parts = re.split(r'[.,–?]', text)
            text_parts = [part.strip() for part in text_parts if part.strip()]
            for p in text_parts:
                total_chars += len(p.split(' '))
            
            total_duration = end - start
            if total_chars == 0:
                continue
            
            seconds_per_char = total_duration / total_chars
            current_start = start

            for part in text_parts:
                duration = len(part.split(' ')) * seconds_per_char
                current_end = current_start + duration
                srt_content += f"{counter}\n"
                srt_content += f"{format_srt_time(current_start)} --> {format_srt_time(current_end)}\n"
                srt_content += f"{part}\n\n"
                current_start = current_end
                counter += 1
    return srt_content
def authenticate_youtube():
    """Authenticates the user and returns a YouTube API service object."""
    credentials = None

    # Disable OAuthlib's HTTPS verification when running locally.
    # DO NOT leave this setting enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    # Check if a token file exists and load credentials from it
    if os.path.exists(TOKEN_FILE):
        try:
            credentials = google.oauth2.credentials.Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            print("Loaded credentials from token.json")
        except Exception as e:
            print(f"Error loading credentials from token.json: {e}")
            credentials = None # Force re-authentication if token is invalid

    # If no valid credentials, initiate the OAuth flow
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            # Refresh the token if it's expired and a refresh token is available
            print("Credentials expired, attempting to refresh...")
            try:
                credentials.refresh(google.auth.transport.requests.Request())
                print("Credentials refreshed successfully.")
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                credentials = None # Force full re-authentication if refresh fails
        
        if not credentials or not credentials.valid:
            print("Initiating new authentication flow...")
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            try:
                # This will start a local web server to handle the OAuth redirect
                credentials = flow.run_local_server(port=8080) # Explicitly set port
                print("Authentication successful.")
            except requests.exceptions.ConnectionError as e:
                print(f"\nERROR: Failed to connect to Google's authentication server during token exchange.")
                print(f"This often indicates a network issue, an interfering firewall/antivirus, or outdated SSL certificates.")
                print(f"Please check your internet connection, temporarily disable firewall/antivirus (for testing),")
                print(f"and ensure your Python's SSL certificates are up-to-date (e.g., run 'Install Certificates.command' on Windows).")
                sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred during authentication: {e}")
                sys.exit(1)

        # Save the credentials for future use
        with open(TOKEN_FILE, 'w') as token:
            token.write(credentials.to_json())
            print(f"Credentials saved to {TOKEN_FILE}")

    # Build the YouTube API service object
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", credentials=credentials)

    return youtube

def upload_video(youtube, options):
    """Uploads a video to YouTube with retry logic."""
    tags = None
    if options.keywords:
        tags = options.keywords.split(",")

    request_body = {
        "snippet": {
            "categoryId": options.category,
            "title": options.title,
            "description": options.description,
            "tags": tags
        },
        "status": {
            "privacyStatus": options.privacyStatus
        }
    }

    media_file = options.file

    if not os.path.exists(media_file):
        sys.exit(f"Error: Video file '{media_file}' not found.")

    insert_request = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=googleapiclient.http.MediaFileUpload(media_file, chunksize=-1, resumable=True)
    )

    response = None
    error = None
    retry = 0

    print(f"Starting upload for '{options.title}' from '{media_file}'...")

    while response is None:
        try:
            status, response = insert_request.next_chunk()
            if status:
                print(f"Upload {int(status.progress() * 100)}%")

            if response is not None:
                if 'id' in response:
                    print(f"Video '{options.title}' uploaded successfully with ID: {response['id']}")
                else:
                    sys.exit(f"The upload failed with an unexpected response: {response}")
        except googleapiclient.errors.HttpError as e:
            if e.resp.status in RETRIABLE_STATUS_CODES:
                error = f"A retriable HTTP error {e.resp.status} occurred:\n{e.content}"
            else:
                raise # Re-raise non-retriable HTTP errors
        except RETRIABLE_EXCEPTIONS as e:
            error = f"A retriable network error occurred: {e}"
        except Exception as e:
            # Catch any other unexpected errors during chunk upload
            print(f"An unexpected error occurred during upload: {e}")
            sys.exit(1)

        if error is not None:
            print(error)
            retry += 1
            if retry > MAX_RETRIES:
                sys.exit(f"Exceeded maximum retries ({MAX_RETRIES}). Giving up on upload.")

            max_sleep = 2 ** retry
            sleep_seconds = random.random() * max_sleep
            print(f"Sleeping {sleep_seconds:.2f} seconds and then retrying...")
            time.sleep(sleep_seconds)
            error = None # Reset error for next retry attempt

def authenticate_youtube():
    """Authenticates the user and returns a YouTube API service object."""
    credentials = None

    # Disable OAuthlib's HTTPS verification when running locally.
    # DO NOT leave this setting enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    # Check if a token file exists and load credentials from it
    if os.path.exists(TOKEN_FILE):
        try:
            credentials = google.oauth2.credentials.Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            print("Loaded credentials from token.json")
        except Exception as e:
            print(f"Error loading credentials from token.json: {e}")
            credentials = None # Force re-authentication if token is invalid

    # If no valid credentials, initiate the OAuth flow
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            # Refresh the token if it's expired and a refresh token is available
            print("Credentials expired, attempting to refresh...")
            try:
                credentials.refresh(google.auth.transport.requests.Request())
                print("Credentials refreshed successfully.")
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                credentials = None # Force full re-authentication if refresh fails
        
        if not credentials or not credentials.valid:
            print("Initiating new authentication flow...")
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            try:
                # This will start a local web server to handle the OAuth redirect
                credentials = flow.run_local_server(port=8080) # Explicitly set port
                print("Authentication successful.")
            except requests.exceptions.ConnectionError as e:
                print(f"\nERROR: Failed to connect to Google's authentication server during token exchange.")
                print(f"This often indicates a network issue, an interfering firewall/antivirus, or outdated SSL certificates.")
                print(f"Please check your internet connection, temporarily disable firewall/antivirus (for testing),")
                print(f"and ensure your Python's SSL certificates are up-to-date (e.g., run 'Install Certificates.command' on Windows).")
                sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred during authentication: {e}")
                sys.exit(1)

        # Save the credentials for future use
        with open(TOKEN_FILE, 'w') as token:
            token.write(credentials.to_json())
            print(f"Credentials saved to {TOKEN_FILE}")

    # Build the YouTube API service object
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", credentials=credentials)

    return youtube

def upload_video(youtube, category,title,description,keywords,privacy_status,video_path,CLIENT_SECRETS_FILE,SCOPES,RETRIABLE_STATUS_CODES,MAX_RETRIES,RETRIABLE_EXCEPTIONS):
    """Uploads a video to YouTube with retry logic."""
    tags = None
    if keywords:
        tags = keywords
    request_body = {
        "snippet": {
            "categoryId": category,
            "title": title,
            "description": description,
            "tags": tags
        },
        "status": {
            "privacyStatus": privacy_status
        }
    }

    media_file = video_path
    if not os.path.exists(media_file):
        sys.exit(f"Error: Video file '{media_file}' not found.")

    insert_request = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=googleapiclient.http.MediaFileUpload(media_file, chunksize=-1, resumable=True)
    )

    response = None
    error = None
    retry = 0

    print(f"Starting upload ")

    while response is None:
        try:
            status, response = insert_request.next_chunk()
            if status:
                print(f"Upload {int(status.progress() * 100)}%")

            if response is not None:
                if 'id' in response:
                    print(f"Video uploaded successfully with ID: {response['id']}")
                else:
                    sys.exit(f"The upload failed with an unexpected response: {response}")
        except googleapiclient.errors.HttpError as e:
            if e.resp.status in RETRIABLE_STATUS_CODES:
                error = f"A retriable HTTP error {e.resp.status} occurred:\n{e.content}"
            else:
                raise # Re-raise non-retriable HTTP errors
        except RETRIABLE_EXCEPTIONS as e:
            error = f"A retriable network error occurred: {e}"
        except Exception as e:
            # Catch any other unexpected errors during chunk upload
            print(f"An unexpected error occurred during upload: {e}")
            sys.exit(1)

        if error is not None:
            print(error)
            retry += 1
            if retry > MAX_RETRIES:
                sys.exit(f"Exceeded maximum retries ({MAX_RETRIES}). Giving up on upload.")

            max_sleep = 2 ** retry
            sleep_seconds = random.random() * max_sleep
            print(f"Sleeping {sleep_seconds:.2f} seconds and then retrying...")
            time.sleep(sleep_seconds)
            error = None # Reset error for next retry attempt

def upload_video_to_youtube_automation(category,title,description,keywords,privacy_status,video_path,CLIENT_SECRETS_FILE,SCOPES,RETRIABLE_STATUS_CODES,MAX_RETRIES,RETRIABLE_EXCEPTIONS):
    
    #category : string , title : string , description : string , keywords_list : string ngăn cách bởi dấu ( , ) ,privac
    
    # Ensure client_secrets.json exists
    if not os.path.exists(CLIENT_SECRETS_FILE):
        print(f"ERROR: '{CLIENT_SECRETS_FILE}' not found.")
        print(f"Please download your OAuth 2.0 client secrets JSON file from Google Cloud Console")
        print(f"and save it as '{CLIENT_SECRETS_FILE}' in the same directory as this script.")
        print(f"For more info: [https://developers.google.com/api-client-library/python/guide/aaa_client_secrets](https://developers.google.com/api-client-library/python/guide/aaa_client_secrets)")
        sys.exit(1)

    youtube = authenticate_youtube()
    try:
        upload_video(youtube, category,title,description,keywords,privacy_status,video_path,CLIENT_SECRETS_FILE,SCOPES,RETRIABLE_STATUS_CODES,MAX_RETRIES,RETRIABLE_EXCEPTIONS)
    except googleapiclient.errors.HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred during video upload:\n{e.content}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
def sending(BOT_TOKEN,CHANNEL_ID,contents):
    files={}
    data={}
    data.update({"content": contents})
    headers = {
        "Authorization": f"Bot {BOT_TOKEN}",
    }
    url = f"https://discordapp.com/api/v8/channels/{CHANNEL_ID}/messages"

    response = requests.post(url, headers=headers, data=data)
    print (response.text)
class VideoSplitter:
    def __init__(self, max_size_mb=30):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    def get_video_duration(self, video_path):
        """Lấy thời lượng video (giây)"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return None
    
    def get_video_bitrate(self, video_path):
        """Lấy bitrate của video (bps)"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=bit_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            bitrate_str = result.stdout.strip()
            if bitrate_str == 'N/A':
                # Ước tính bitrate nếu không có thông tin
                file_size = os.path.getsize(video_path)
                duration = self.get_video_duration(video_path)
                return (file_size * 8) / duration if duration else None
            return float(bitrate_str)
        except (subprocess.CalledProcessError, ValueError):
            return None
    
    def calculate_segment_duration(self, video_path):
        """
        Tính thời lượng mỗi segment dựa trên dung lượng tối đa
        """
        duration = self.get_video_duration(video_path)
        bitrate = self.get_video_bitrate(video_path)
        
        if duration is None:
            raise ValueError("Không thể lấy thời lượng video")
        
        if bitrate is None:
            # Fallback: ước tính dựa trên dung lượng file
            file_size = os.path.getsize(video_path)
            estimated_bitrate = (file_size * 8) / duration
            segment_duration = (self.max_size_bytes * 8) / estimated_bitrate
        else:
            # Tính thời lượng mỗi segment (giây)
            segment_duration = (self.max_size_bytes * 8) / bitrate
        
        # Thêm margin an toàn 10%
        return min(segment_duration * 0.9, duration)
    
    def split_by_size(self, input_path, output_pattern=None):
        """
        Chia video thành các phần nhỏ hơn 50MB
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"File không tồn tại: {input_path}")
        
        if output_pattern is None:
            output_pattern = f"{input_path.stem}_part%03d{input_path.suffix}"
        
        print(f"Đang xử lý video: {input_path.name}")
        print(f"Dung lượng tối đa mỗi phần: {self.max_size_mb}MB")
        
        # Tính thời lượng mỗi segment
        try:
            segment_duration = self.calculate_segment_duration(input_path)
            print(f"Thời lượng mỗi phần: {segment_duration:.2f} giây")
        except Exception as e:
            print(f"Lỗi khi tính toán: {e}")
            return False
        
        # Sử dụng FFmpeg để chia video
        command = [
            'ffmpeg', '-i', str(input_path),
            '-c', 'copy',  # Copy stream không re-encode (nhanh)
            '-fs', str(self.max_size_bytes),  # Giới hạn dung lượng file
            '-reset_timestamps', '1',
            str(output_pattern)
        ]
        
        print("Đang chia video...")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("✅ Chia video thành công!")
            
            # Liệt kê các file đã tạo
            output_files = list(Path('.').glob(output_pattern.replace('%03d', '*')))
            for file in output_files:
                size_mb = os.path.getsize(file) / (1024 * 1024)
                print(f"📁 {file.name} - {size_mb:.2f}MB")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi chia video: {e.stderr}")
            return False

def upgrade_to_60fps_mci(input_file, output_file):
    # Lệnh ffmpeg sử dụng nội suy chuyển động (Motion Interpolation)
    # fps=60: Đích đến là 60 khung hình/giây
    # mi_mode=mci: Chế độ nội suy bù chuyển động (cho kết quả mượt nhất)
    # mc_mode=aobmc: Giảm thiểu hiện tượng răng cưa/nhiễu khi chuyển động
    
    command = [
        'ffmpeg',
        '-i', input_file,
        '-filter:v', "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc",
        '-c:v', 'libx264',
        '-crf', '18',        # Chất lượng cao (giá trị từ 18-23 là ổn)
        '-preset', 'medium', # Tốc độ nén (slow sẽ đẹp hơn nhưng lâu hơn)
        '-c:a', 'copy',      # Giữ nguyên bản gốc âm thanh để tránh lệch sync
        output_file,
        '-y'                 # Tự động ghi đè nếu file output đã tồn tại
    ]

    print(f"--- Đang bắt đầu xử lý: {input_file} ---")
    print("Lưu ý: Filter 'minterpolate' rất nặng, máy sẽ khá nóng và mất thời gian.")

    try:
        # Chạy subprocess và hiển thị log trực tiếp ra màn hình console
        subprocess.run(command, check=True)
        print(f"\n✅ Thành công! File đã lưu tại: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Lỗi FFmpeg: {e}")
    except FileNotFoundError:
        print("\n❌ Lỗi: Không tìm thấy lệnh 'ffmpeg'. Hãy kiểm tra lại PATH của hệ thống.")

def run_adb_command(command):
    # command là một danh sách các từ khóa, ví dụ: ["adb", "devices"]
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"Lỗi: {result.stderr.strip()}"
def xml_to_json(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    nodes_list = []

    # Duyệt qua tất cả các thẻ 'node'
    for node in root.iter('node'):
        # Lấy tất cả thuộc tính của node đó (text, resource-id, bounds, class, etc.)
        node_data = node.attrib 
        nodes_list.append(node_data)

    # Chuyển danh sách thành chuỗi JSON (indent=4 để dễ đọc)
    with open('view.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(nodes_list, indent=4, ensure_ascii=False))    
    return json.dumps(nodes_list, indent=4, ensure_ascii=False)    

def draw_elements():
    # Đọc file XML và Ảnh
    tree = ET.parse('view.xml')
    root = tree.getroot()
    img = cv2.imread('screen.png')
    img_ = Image.open('screen.png')
    
    # Duyệt qua tất cả các node 'node' trong XML
    for node in root.iter('node'):
        # Lấy tọa độ từ attribute 'bounds' (định dạng: [x1,y1][x2,y2])
        bounds = node.get('bounds')
        resource_id = node.get('resource-id')
        
        # Dùng Regex để tách số từ chuỗi [x1,y1][x2,y2]
        coords = re.findall(r'\d+', bounds)
        if coords:
            x1, y1, x2, y2 = map(int, coords)
            try:
                cropped_img = img_.crop((x1, y1, x2, y2))

                # 4. Lưu hoặc hiển thị ảnh đã cắt
                cropped_img.save(f'image\\cropped_sample_{x1}_{y1}_{x2}_{y2}.png')
            except:
                None    
def create_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)
    else:
        None
def find_possition(text):
    # 1. Tải cấu hình mặc định của model vgg_transformer
    config = Cfg.load_config_from_name('vgg_transformer')

    # 2. CHỈNH SỬA CẤU HÌNH TRƯỚC KHI KHỞI TẠO MODEL
    # Ép buộc sử dụng CPU để tránh lỗi CUDA
    config['device'] = 'cpu' 

    # Nếu bạn dùng weights mặc định của hệ thống thì để True
    # Nếu bạn tự load file pth riêng mà chưa train xong thì mới để False
    config['cnn']['pretrained'] = True 

    # 3. KHỞI TẠO PREDICTOR 
    # Lúc này model sẽ được load trực tiếp vào CPU
    detector = Predictor(config)

    # 4. ĐƯỜNG DẪN ẢNH
    # Hãy đảm bảo file 'sample.png' nằm cùng thư mục với file code này
    list_path = glob('image\\*.png', recursive=False)
    for img_path in list_path:
        try:
            img = Image.open(img_path)
            
            # 5. NHẬN DIỆN CHỮ
            s = detector.predict(img)
            
            if s==text:
                x1,y1,x2,y2=((img_path.replace('image\cropped_sample_','')).replace('.png','')).split('_')
                return x1,y1,x2,y2
        except:
            None    

def defind_similar_possision(input_fill_path,path_of_list_image):
    template = cv2.imread(input_fill_path)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h_t, w_t = gray_template.shape
    ratio_template = round(w_t / h_t, 2) # Tỉ lệ ngang/dọc của icon
    print (ratio_template)

    list_path = glob(f'{path_of_list_image}/*.png')

    print(f"Mẫu: {w_t}x{h_t} (Tỉ lệ: {ratio_template})")
    print (list_path)
    for path in list_path:
        
        try:
            large_img = cv2.imread(path)
            if large_img is None: continue
            
            h_l, w_l = large_img.shape[:2];t_l=w_l/w_t-h_l/h_t
            if abs(t_l) <= 0.5:  
                
                gray_large = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(gray_large, gray_template, cv2.TM_CCOEFF_NORMED)

                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                threshold = 0.8

                if max_val >= threshold:                           
                    print(f"[MATCH] {max_val:.2f} | Tỉ lệ khớp: {t_l} | File: {os.path.basename(path)}")
                    return path
            else:
                None        
                
        except:
            None

def get_android_data(device_port_ip):
    print("🚀 Đang kết nối và lấy dữ liệu bằng uiautomator2...")
    # Kết nối tới thiết bị đầu tiên tìm thấy qua ADB
    d = u2.connect(device_port_ip) 
    
    try:
        # Cách sửa lỗi: Sử dụng đúng tên thuộc tính cho bản u2 mới
        # Thường là 'wait_for_idle' hoặc đặt trực tiếp khi dump
        d.settings['wait_for_idle'] = False 
    except AttributeError:
        # Nếu vẫn báo lỗi attribute, ta sẽ bỏ qua và xử lý ở bước dump
        pass

    print("📸 Đang chụp ảnh màn hình...")
    d.screenshot("screen.png")

    print("📄 Đang lấy XML (Bypass Idle)...")
    # compressed=True giúp lấy XML nhanh hơn và gọn hơn
    xml_data = d.dump_hierarchy(compressed=True)
    
    with open("view.xml", "w", encoding="utf-8") as f:
        f.write(xml_data)
        
    print("✅ Thành công: Đã lưu view.xml và screen.png")

def parse_bounds(bounds_str):
    """Chuyển đổi chuỗi [x1,y1][x2,y2] thành list số nguyên [x1, y1, x2, y2]"""
    return list(map(int, re.findall(r'\d+', bounds_str)))

def analyze_xml(file_path):
    # Đọc file XML
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    slots = []

    # Tìm các node là ô chọn ảnh/video
    # Dựa vào file của bạn, các ô này có resource-id chứa 'my2'
    for node in root.findall(".//node[@resource-id='com.ss.android.ugc.trill:id/mzz']"):
        # Lấy node cha của ImageView (thường là FrameLayout bao quanh ô đó)
        # Trong XML này, node cha chứa bounds thực của ô
        parent = root.find(f".//node[@index='{node.get('index')}']/..") 
        
        bounds_str = node.get('bounds')
        if bounds_str:
            coords = parse_bounds(bounds_str)
            slots.append({
                'resource_id': node.get('resource-id'),
                'coords': coords, # [x1, y1, x2, y2]
                'width': coords[2] - coords[0],
                'height': coords[3] - coords[1]
            })

    # Sắp xếp các ô theo tọa độ Y (hàng) rồi đến X (cột)
    slots.sort(key=lambda s: (s['coords'][1], s['coords'][0]))
    
    return slots

def check_screen():
    # Chỉ gọi lệnh lấy thông tin nguồn
    cmd = "adb shell dumpsys power"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Dùng Python để tìm chuỗi, không phụ thuộc vào grep hay findstr
    if "mWakefulness=Awake" in result.stdout:
        return "ON"
    else:
        return "OFF"

def send_key_vietnamese(text,device_port_ip):
    d = u2.connect(device_port_ip) 

    # Click vào ô nhập liệu (ví dụ ô tìm kiếm hoặc tin nhắn)
    # d(focused=True).click()

    # Gõ tiếng Việt trực tiếp
    d.set_fastinput_ime(True) # Kích hoạt bàn phím hỗ trợ gõ nhanh
    d.clear_text()            # Xóa chữ cũ (nếu cần)
    d.send_keys(text)
    d.set_fastinput_ime(False) # Trả về bàn phím cũ sau khi xong    

def get_douyin_video_src(url):
    with sync_playwright() as p:
        # Mở trình duyệt (headless=True nếu bạn không muốn hiện cửa sổ)
        browser = p.chromium.launch_persistent_context(r'C:\Users\Sunny\AppData\Local\Google\Chrome\User Data\Profile 3', headless=True)
        page = browser.new_page()
        
        # Truy cập vào link video
        page.goto(url)
        
        # Chờ một chút để player kịp load dữ liệu
        page.wait_for_timeout(6000) 
        # Thực hiện đoạn code "console" của bạn
        try:
            video_src = page.evaluate("player.videoList[0].playAddr[0].src")
            return video_src
        except Exception as e:
            print("Không tìm thấy biến player. Có thể trang web đã thay đổi cấu trúc.")
        #i=input('player.videoList[0].playAddr[0].src')
        browser.close()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Header giúp Chrome hiểu đây là file tải về, không phải nội dung xem trực tiếp
        self.send_header('Content-Disposition', 'attachment')
        super().end_headers()

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

def run_server(httpd):
    httpd.serve_forever()

def xong_luon(video_name, phone_id, pc_ip):
    PORT = 8000
    # Đảm bảo đường dẫn file là tương đối so với nơi chạy script
    # Ví dụ: Sourse_Videos/subed_1_video.mp4
    
    with ThreadedTCPServer(("", PORT), MyHandler) as httpd:
        video_path=video_name.replace('\\','/')
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        print(f"--- Server đang chạy tại http://{pc_ip}:{PORT} ---")

        file_only = os.path.basename(video_name)
        url = f"http://{pc_ip}:{PORT}/{video_path}"
        print (url)
        # Lệnh tải file bằng ADB (Dùng curl hoặc wget tích hợp trong Android)
        print(f"--- Đang ra lệnh cho {phone_id} tải file qua ADB shell ---")
        
        # Cách 1: Mở Chrome (như bạn đang làm)
        # Thêm tham số package của Chrome
        run_adb_command(f'adb -s {phone_id} shell am start -n com.android.chrome/com.google.android.apps.chrome.Main -d "https://www.google.com/recaptcha/api2/demo"')
        run_adb_command(f'adb -s {phone_id} shell am start -n com.android.chrome/com.google.android.apps.chrome.Main -d "{url}"')
        
        # Cách 2 (Khuyên dùng): Tải ngầm bằng curl (nếu máy có sẵn) để chính xác hơn
        # run_adb_command(f'adb -s {phone_id} shell curl -o /sdcard/Download/{file_only} {url}')

        # Đợi tải...
        is_finished = False
        for i in range(30): 
            time.sleep(5)
            check_cmd = run_adb_command(f'adb -s {phone_id} shell ls /sdcard/Download/')
            if file_only in check_cmd:
                # Kiểm tra thêm nếu file .crdownload (đang tải) đã biến mất chưa
                if ".crdownload" not in check_cmd:
                    print(f"==> [OK] Đã tải xong: {file_only}")
                    is_finished = True
                    break
            print(f"[{i}] Đang đợi điện thoại tải file...")

        httpd.shutdown()

def clear_folder(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)    

def merge_and_trim_audio(video_input, audio_input, video_output):
    # CHỈNH Ở ĐÂY: 0.1 là bé tí, 0.5 là vừa, 1.0 là giữ nguyên
    mp3_vol = 0.2 
    
    # Gom hết vào một dòng filter cho gọn
    # Giải thích: Lấy audio2 (file mp3) giảm volume -> trộn với audio1 (video) -> cắt theo video
    filter_cmd = f"[1:a]volume={mp3_vol}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[aout]"

    command = [
        'ffmpeg', '-i', video_input, '-i', audio_input,
        '-filter_complex', filter_cmd,
        '-map', '0:v:0', '-map', '[aout]',
        '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', 
        video_output
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"✅ Xong! Đã thêm âm thanh vào video: {video_output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Toang rồi: {e.stderr.decode()}")

if __name__ == "__main__":
    try:
        # Lấy trực tiếp kết quả dạng text từ lệnh adb
        output = subprocess.check_output('adb devices', text=True)
        # Tìm kiếm cấu trúc IP 192.168.x.x...device
        if re.search(r'device', output) != None:
            #print ((output))
            device_port_ip=((str(output).split('\t')[0]).split('\n')[1])
            print("Đã tìm thấy thiết bị mạng. Đang thoát...",device_port_ip)
        else:
            while True:
                device_port_ip=input('Nhập ip-port thiết bị (*ví dụ 192:168.1.11:38287) : ')
                subprocess.run('adb kill-server',check=True)  
                status=subprocess.check_output(f'adb connect {device_port_ip}',text=True)    
                print (status)  
                if status.find('connected to') >=0:
                    subprocess.run(f'adb -s {device_port_ip} tcpip 5555',check=True)
                    device_port_ip=str(device_port_ip.split(':')[0])+':5555'
                    subprocess.run(f'adb connect {device_port_ip}',check=True)
                    break
                else:
                    print ('KẾT NỐI ADB THẤT BẠI CHECK LẠI IP-PORT THIẾT BỊ')              
            
    except subprocess.CalledProcessError:
        print("Lỗi khi chạy lệnh ADB.")   
    # Cấu hình biến toàn cục
    select_options = '123'
    full_option = True
    id_video = ''
    file_infor_json_file_path = 'sourse_video_infor.json'
    complete_json_path = 'video_id_completed.txt'
    lauching_file_path = 'running_id_video.txt'    
    counter = 1
    BOT_TOKEN = "8375989233:AAHrckOR07fxa0G0gUbmcs47RaaLfcSVDkg"
    CHAT_ID = "7531993744"
    # Các hằng số cho YouTube API
    RETRIABLE_STATUS_CODES = [500, 502, 503, 504]
    SCOPES = "https://www.googleapis.com/auth/youtube.upload"
    TOKEN_FILE = 'token.json'
    CLIENT_SECRETS_FILE = "client_secrets.json"
    VALID_PRIVACY_STATUSES = ("public", "private", "unlisted")
    MAX_RETRIES = 10
    RETRIABLE_EXCEPTIONS = (httplib.NotConnected, httplib.IncompleteRead, 
                           httplib.ImproperConnectionState, httplib.CannotSendRequest,
                           httplib.CannotSendHeader, httplib.ResponseNotReady,
                           httplib.BadStatusLine, IOError) 
    if os.path.exists(complete_json_path):
        with open(complete_json_path, 'r', encoding='utf-8') as f:
            check_point_id_video_completed = [value.strip() for value in f.readlines()]        
    else:
        check_point_id_video_completed = []
    with open(file_infor_json_file_path, 'r', encoding='utf-8') as f:
        list_video_bil = [value for value in json.load(f) if value['id_video'] not in check_point_id_video_completed]   
    for range_id in range (len(list_video_bil)): 
        try:   
            if os.path.exists(lauching_file_path):
                with open(lauching_file_path, 'r', encoding='utf-8') as f:
                    id_video = f.read().strip();f.close()        
            # Chọn video tiếp theo để xử lý
            target_id_video_bil = list_video_bil[range_id]['id_video'];srt_path = os.path.join('dubbed_movie', f'translated_{str(len(check_point_id_video_completed)+1)}_subs.srt');srt_content = ""
            if id_video != target_id_video_bil:
                if target_id_video_bil.find('https://www.bilibili.com') >=0 or target_id_video_bil.find('https://www.youtube.com') >= 0:
                    # Tải video mới nếu cần
                    cmd = [
                        "yt-dlp",
                        f"{target_id_video_bil}",
                        "--cookies", "cookies.txt",
                        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", # Cách viết an toàn hơn
                        "--merge-output-format", "mp4",
                        "-o", "Videos/video.%(ext)s",
                    ]
                    subprocess.run(cmd, check=True)            
                elif target_id_video_bil.find('https://www.iesdouyin.com') ==0:
                    print ("Downloading video from Douyin...")
                    setup_directories('Videos')
                    with open('Videos/video.mp4', 'wb') as f:
                        while True:                            
                            try:
                                link_dowload_video_douyin=get_douyin_video_src(target_id_video_bil)
                                print ('LINK DOWLOAD VIDEO CỦA BẠN : '+link_dowload_video_douyin)  
                                if link_dowload_video_douyin != None:
                                    break
                            except:   
                                link_dowload_video_douyin=(save_video_douyin(target_id_video_bil)) 
                                print ('LINK DOWLOAD VIDEO CỦA BẠN : '+link_dowload_video_douyin)
                                if link_dowload_video_douyin !=None:
                                    break
                        while True:
                            try:
                                #proxy_format=get_and_use_proxy()    
                                #proxies = {
                                #    "http": proxy_format,
                                #    "https": proxy_format
                                #}
                                content_from_link_dowload_video_dy=requests.get(link_dowload_video_douyin).content    
                                break 
                            except:
                                time.sleep(10)                 
                        f.write(content_from_link_dowload_video_dy)                    
                else:
                    setup_directories('Videos')
                    with open('Videos/video.mp4', 'wb') as f:
                        f.write(requests.get(target_id_video_bil).content)                             
                # Đổi tên file video
                video_path = glob("Videos/*.mp4")[0]
                
                # Lưu ID video đang xử lý
                with open(lauching_file_path, 'w', encoding='utf-8') as f:
                    f.write(target_id_video_bil)
            if select_options == '12':
                full_option = True        
            input_video = f"Videos/video.mp4"
            decrease_video_path = input_video.replace(os.path.basename(input_video), 'decrease_video.mp4')
            final_video_path = input_video.replace(os.path.basename(input_video), f'final_video.mp4')
            output_dir = "dubbed_movie"
            setup_directories(output_dir)
            
            # Đọc API keys
            with open('Api_key/gemini_api_key.txt', 'r') as f:
                api_keys = [line.strip() for line in f if line.strip()]
            
            # Phát hiện ngôn ngữ
            setup_directories(os.path.join(output_dir, 'analyze'))
            analyze_file_path = os.path.join(output_dir, 'analyze', 'video_analyze_languae.mp4')
            cut_video_clip_30s(input_video, analyze_file_path)
            source_language = detect_langue(analyze_file_path)
            script_letter = ''
            target_language = "vi"
            
            # Bắt đầu quá trình lồng tiếng
            dub_movie(input_video, output_dir, api_keys, source_language, target_language, full_option, target_id_video_bil);segments=[]
            with open(os.path.join('dubbed_movie', 'translated_transcript.json'), 'r', encoding='utf-8') as file:
                segments = json.load(file)['segments'];file.close()  
            srt_content = convert_detail(segments, srt_content, counter)
            with open ('srt_content.txt', 'w', encoding='utf-8') as f:
                f.write(srt_content)
            if srt_content=='':
                continue    
            with open(srt_path, 'a', encoding='utf-8') as f:
                f.write(srt_content)
            print(f"File phụ đề SRT đã được tạo tại: {srt_path}")
            success = increase_audio_volume(input_video, decrease_video_path, volume_factor=0.05);file_path = 'checkpoint_dub_demo.mp4.json';output_path_voice_compare = os.path.join('Videos', 'output_video.mp4')
            # Đọc dữ liệu từ file JSON
            with open(f"checkpoint_dub_{os.path.basename(input_video)}.json", 'r', encoding='utf-8') as file:
                data = [o for o in json.load(file)['segments'] if o != None]
                file.close()
            # Tạo danh sách các input audio và thời gian bắt đầu
            audio_inputs = [];setup_directories('tts_speeded')
            for segment in data:
                index = segment['start_time']
                start_time = segment['start_time']
                end_time = segment['end_time']
                audio_path = segment['output_path']
                
                # Kiểm tra xem file audio có tồn tại không
                if os.path.exists(audio_path):
                    #tăng tốc đoạn âm thanh audio                    
                    speed=float(get_duration(audio_path))/(float(end_time)-float(start_time));audio_path_speeded=audio_path.replace('tts\\','tts_speeded\\') 
                    speed_up_video(audio_path, speed, audio_path_speeded)
                    audio_inputs.append({
                        'audio_path': audio_path_speeded,
                        'start_time': start_time,
                        'end_time': end_time
                    })
                    print ('Đã tăng tốc file âm thanh',audio_path_speeded)
                else:
                    print(f"File {audio_path} không tồn tại, bỏ qua đoạn này.")
                   
            # Xây dựng lệnh ffmpeg
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', decrease_video_path,  # Input video (có audio gốc)
            ]
#            # Thêm tất cả file audio làm input
            for audio in audio_inputs:
                ffmpeg_cmd.extend(['-i', audio['audio_path']])
#            # Tạo bộ lọc phức tạp (complex filter)
            filter_complex = ''
            # 1. Tạo delay cho các file TTS như cũ
            for i, audio in enumerate(audio_inputs):
                delay_ms = round(float(audio["start_time"]) * 1000)
                filter_complex += f'[{i+1}:a]adelay={delay_ms}|{delay_ms}[a{i+1}];'

            # 2. CHỈNH SỬA QUAN TRỌNG: Không đưa [a0] vào amix
            # Thay vì: filter_complex += '[a0]' + ''.join(...)
            # Hãy sửa thành:
            filter_complex += ''.join(f'[a{i+1}]' for i in range(len(audio_inputs))) + f'amix=inputs={len(audio_inputs)}:normalize=0[aout]'
            # Thêm các tham số output
            ffmpeg_cmd.extend([
                '-filter_complex',
                filter_complex,  # ← Dùng biến đã tạo
                '-map', '0:v',
                '-map', '[aout]',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                output_path_voice_compare,
                '-y'
            ])
#            # In lệnh ffmpeg để kiểm tra (tùy chọn)
            print('Lệnh ffmpeg:', ' '.join(ffmpeg_cmd))
#            # Chạy lệnh ffmpeg
            try:
                result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                print(f"Video đã được xử lý và lưu tại {output_path_voice_compare}")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Lỗi khi chạy ffmpeg: {e.stderr}")
            except FileNotFoundError:
                print("Lỗi: Không tìm thấy ffmpeg. Hãy đảm bảo ffmpeg đã được cài đặt và thêm vào PATH.")
            increase_audio_volume(output_path_voice_compare, final_video_path, volume_factor=1.5)
            if True:    
                # File đầu ra
                setup_directories("Sourse_Videos")
                output_video = (f'Sourse_Videos\\subed_{str(len(check_point_id_video_completed)+1)}_video.mp4');sub_path_tile=srt_path.replace('\\','/')
                # Câu lệnh FFmpeg dưới dạng list (tránh lỗi escape)
                cmd = [
                    "ffmpeg",
                    "-i", final_video_path,
                    # Chữ trắng, viền đen (Outline), Size 24 để dễ đọc hơn
                    "-vf", f"subtitles={sub_path_tile}:force_style='FontName=Arial,FontSize=10,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=50'",
                    "-c:v", "libx264",       # Bộ mã hóa H.264
                    "-crf", "26",            # Giữ chất lượng tốt hơn 28 một chút, nhưng vẫn nhẹ
                    "-preset", "veryfast",   # TĂNG TỐC ĐỘ XỬ LÝ ĐÁNG KỂ
                    "-tune", "film",         # Tối ưu cho video thông thường
                    "-movflags", "+faststart", # Tối ưu cho phát trực tuyến
                    "-c:a", "aac",           # Mã hóa lại audio sang AAC
                    "-b:a", "96k",           # GIẢM BITRATE AUDIO (tùy chọn) để file nhẹ hơn
                    "-threads", "0",
                    "-f", "mp4",
                    "-y",
                    output_video.replace("\\",'/')
                    
                ]
                # Chạy lệnh và hiển thị log
                try:
                    subprocess.run(cmd, check=True)
                    print("✅ Hoàn tất! Video đã được chèn phụ đề.")
                except subprocess.CalledProcessError as e:
                    print("❌ Lỗi khi chạy FFmpeg:", e)  
            
#            #upload_video_to_youtube_automation(category,'HƯỚNG DẪN LÀM SLIME TẠI NHÀ .🍓',description,keywords,privacy_status,output_video,CLIENT_SECRETS_FILE,SCOPES,RETRIABLE_STATUS_CODES,MAX_RETRIES,RETRIABLE_EXCEPTIONS)              
            #setup_directories('output')
            #merge_and_trim_audio(output_video, 'music_bg/music.mp3', output_video.replace('Sourse_Videos', 'output'))
            
            status = run_adb_command(f'adb -s {device_port_ip} shell "dumpsys display | grep mScreenState"')
            if status.find('ON') < 0:
                run_adb_command(f'adb -s {device_port_ip} shell input keyevent 26')
            #setup_directories('output');upgrade_to_60fps_mci(output_video.replace("\\",'/'), (output_video.replace("\\",'/')).replace('Sourse_Videos','output'))
            # Ví dụ: Liệt kê thiết bị
            formatted_path = output_video.replace("\\", "/")#.replace('Sourse_Videos', 'output')
            pc_ip=get_local_ip()    

            run_adb_command('adb shell input keyevent 3')
            run_adb_command(f'''adb -s {device_port_ip} shell "find /sdcard/ -name '*.mp4' -delete"''')
            xong_luon(formatted_path, device_port_ip, pc_ip)
            print ('Video Đã được tải ')
            run_adb_command('adb shell input keyevent 3')
            print ('Tiếp tục')
            #print (f'adb -s {device_port_ip} push {formatted_path} /sdcard/DCIM/Camera/')
            #run_adb_command(f'adb -s {device_port_ip} push {formatted_path} /sdcard/DCIM/Camera/')           
            #ft_path=formatted_path.replace('output','')
            #print (f'adb -s {device_port_ip} shell content insert --uri content://media/external/video/media --bind _data:s:/sdcard/DCIM/Camera{ft_path}')
            #run_adb_command(f'adb -s {device_port_ip} shell content insert --uri content://media/external/video/media --bind _data:s:/sdcard/DCIM/Camera{ft_path}')
            content_text=(f'hôm nay tôi chỉ bạn cái này. P{str(len(check_point_id_video_completed)+1)} ');status = check_screen()            
            
            run_adb_command(f'adb -s {device_port_ip} shell am start -a android.intent.action.VIEW -d "https://www.tiktok.com/"')
            time.sleep(10)            
            get_android_data(device_port_ip);json_output = xml_to_json('view.xml');json_output = json.loads(xml_to_json('view.xml'))
            posision_located=([element['bounds'] for element in (json_output) if element['content-desc']=="Quay"][0]).split(']')
            x1,y1= (((posision_located[0])[1:99]).split(','))
            x2,y2= (((posision_located[1])[1:99]).split(','))            
            run_adb_command(f'adb -s {device_port_ip} shell input tap {(int(x1)+int(x2))/2} {(int(y1)+int(y2))/2}')
            time.sleep(5)            
            get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
            posision_located=([element['bounds'] for element in (json_output) if element['index']=='0' and element['text']=='' and element['resource-id']=='' and element['class']=='android.widget.RelativeLayout' and element['package']=='com.ss.android.ugc.trill' and element['content-desc']==''][0]).split(']')
            x1,y1= (((posision_located[0])[1:99]).split(','))
            x2,y2= (((posision_located[1])[1:99]).split(','))
            run_adb_command(f'adb -s {device_port_ip} shell input tap {(int(x1)+int(x2))/2} {(int(y1)+int(y2))/2}')
            time.sleep(3)
            get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
            posision_located=([element['bounds'] for element in (json_output) if element['text']=='Video'][0]).split(']')
            x1,y1= (((posision_located[0])[1:99]).split(','))
            x2,y2= (((posision_located[1])[1:99]).split(','))
            run_adb_command(f'adb -s {device_port_ip} shell input tap {(int(x1)+int(x2))/2} {(int(y1)+int(y2))/2}')
            time.sleep(3)
            get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
            posision_located=([element['bounds'] for element in (json_output) if element['resource-id']=='com.ss.android.ugc.trill:id/n8g'][0]).split(']')
            x1,y1= (((posision_located[0])[1:99]).split(','))
            x2,y2= (((posision_located[1])[1:99]).split(','))
            run_adb_command(f'adb -s {device_port_ip} shell input tap {(int(x1)+int(x2))/2} {(int(y1)+int(y2))/2}')
            time.sleep(3)
            get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
            posision_located=([element['bounds'] for element in (json_output) if element['text']=='Tiếp'][0]).split(']')
            x1,y1= (((posision_located[0])[1:99]).split(','))
            x2,y2= (((posision_located[1])[1:99]).split(','))
            run_adb_command(f'adb -s {device_port_ip} shell input tap {int((int(x1)+int(x2))/2)} {int((int(y1)+int(y2))/2)}')
            time.sleep(5)
            if False:
                try:
                    print ('Thêm nhạc vào video')
                    time.sleep(3)
                    get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
                    posision_located=([element['bounds'] for element in (json_output) if element['content-desc']=='Nhạc'][0]).split(']')
                    x1,y1= (((posision_located[0])[1:99]).split(','))
                    x2,y2= (((posision_located[1])[1:99]).split(','))
                    run_adb_command(f'adb -s {device_port_ip} shell input tap {(int(x1)+int(x2))/2} {(int(y1)+int(y2))/2}')
                    time.sleep(3)   
                    get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
                    posision_located=([element['bounds'] for element in (json_output) if element['content-desc']=='Tìm kiếm'][0]).split(']')
                    x1,y1= (((posision_located[0])[1:99]).split(','))
                    x2,y2= (((posision_located[1])[1:99]).split(','))
                    run_adb_command(f'adb -s {device_port_ip} shell input tap {(int(x1)+int(x2))/2} {(int(y1)+int(y2))/2}')
                    time.sleep(3)     
                    get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
                    posision_located=([element['bounds'] for element in (json_output) if element['class']=='android.widget.EditText' and element['package']=='com.ss.android.ugc.trill' and element['content-desc']=='' and element['checkable']=='false' and element['checked']=='false' and element['clickable']=='true' and element['enabled']=='true' and element['focusable']=='true' and element['focused']=='true' and element['scrollable']=='false' and element['long-clickable']=='true' and element['password']=='false' and element['selected']=='false' and element['visible-to-user']=='true'][0]).split(']')
                    x1,y1= (((posision_located[0])[1:99]).split(','))
                    x2,y2= (((posision_located[1])[1:99]).split(','))
                    run_adb_command(f'adb -s {device_port_ip} shell input tap {(int(x1)+int(x2))/2} {(int(y1)+int(y2))/2}')
                    time.sleep(3)   
                    send_key_vietnamese('Những Cuộc Đua Thúng Chai Trên Sông',device_port_ip)  
                    time.sleep(3)
                    get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
                    posision_located=([element['bounds'] for element in (json_output) if element['text']=='Tìm kiếm'][0]).split(']')
                    x1,y1= (((posision_located[0])[1:99]).split(','))
                    x2,y2= (((posision_located[1])[1:99]).split(','))
                    run_adb_command(f'adb -s {device_port_ip} shell input tap {(int(x1)+int(x2))/2} {(int(y1)+int(y2))/2}')    
                    time.sleep(20)
                    get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
                    posision_located=([element['bounds'] for element in (json_output) if element['index']=='1' and element['text']=='' and element['resource-id']=='' and element['class']=='android.view.ViewGroup' and element['package']=='com.ss.android.ugc.trill' and element['content-desc']=='' and element['checkable']=='false' and element['checked']=='false' and element['clickable']=='false' and element['enabled']=='true' and element['focusable']=='true' and element['focused']=='false' and element['scrollable']=='false' and element['long-clickable']=='false' and element['password']=='false' and element['selected']=='false' and element['visible-to-user']=='true'][0]).split(']')
                    x1,y1= (((posision_located[0])[1:99]).split(','))
                    x2,y2= (((posision_located[1])[1:99]).split(','))
                    run_adb_command(f'adb -s {device_port_ip} shell input tap {(int(x1)+int(x2))/2} {(int(y1)+int(y2))/2}')     
                    time.sleep(3)
                    get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
                    posision_located=([element['bounds'] for element in (json_output) if element['class']=='android.view.ViewGroup' and element['package']=='com.ss.android.ugc.trill' and element['content-desc']=='' and element['checkable']=='false' and element['checked']=='false' and element['clickable']=='false' and element['enabled']=='true' and element['focusable']=='false' and element['focused']=='false' and element['scrollable']=='false' and element['long-clickable']=='false' and element['password']=='false' and element['selected']=='false' and element['long-clickable']=='false' and element['password']=='false' and element['selected']=='false' and element['visible-to-user']=='true'][0]).split(']')
                    x1,y1= (((posision_located[0])[1:99]).split(','))
                    run_adb_command(f'adb -s {device_port_ip} shell input tap {x1} {y1}')  
                    time.sleep(5) 
                    get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
                    posision_located=([element['bounds'] for element in (json_output) if element['text']=='Tiếp'][0]).split(']')
                    x1,y1= (((posision_located[0])[1:99]).split(','))
                    x2,y2= (((posision_located[1])[1:99]).split(','))
                except:
                    None    
            run_adb_command(f'adb -s {device_port_ip} shell input tap {int((int(x1)+int(x2))/2)} {int((int(y1)+int(y2))/2)}')
            time.sleep(10)
            for i in range (20):
                try:
                    get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
                    posision_located=([element['bounds'] for element in (json_output) if element['text']=='Thêm mô tả...'][0]).split(']')
                    x1,y1= (((posision_located[0])[1:99]).split(','))
                    x2,y2= (((posision_located[1])[1:99]).split(','))
                    run_adb_command(f'adb -s {device_port_ip} shell input tap {int((int(x1)+int(x2))/2)} {int((int(y1)+int(y2))/2)}')
                    break
                except:
                    get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
                    posision_located=([element['bounds'] for element in (json_output) if element['text']=='Cho phép'][0]).split(']')
                    x1,y1= (((posision_located[0])[1:99]).split(','))
                    x2,y2= (((posision_located[1])[1:99]).split(','))
                    run_adb_command(f'adb -s {device_port_ip} shell input tap {int((int(x1)+int(x2))/2)} {int((int(y1)+int(y2))/2)}')

            time.sleep(5)
            send_key_vietnamese(content_text,device_port_ip)
            run_adb_command(f'adb -s {device_port_ip} shell input tap {int((int(x1)+int(x2))/2)} {int((int(y1)+int(y2))/2)}')
            time.sleep(5)
            get_android_data(device_port_ip);json_output = json.loads(xml_to_json('view.xml'))
            posision_located=([element['bounds'] for element in (json_output) if element['text']=='Đăng'][0]).split(']')
            x1,y1= (((posision_located[0])[1:99]).split(','))
            x2,y2= (((posision_located[1])[1:99]).split(','))
            run_adb_command(f'adb -s {device_port_ip} shell input tap {int((int(x1)+int(x2))/2)} {int((int(y1)+int(y2))/2)}')
            remove_file(f"checkpoint_dub_{os.path.basename(input_video)}.json");clear_folder('tts');clear_folder('sub');clear_folder('srt');remove_file(f"checkpoint_transcript_{os.path.basename(input_video)}.json");clear_folder("temp_segments");clear_folder("dubbed_movie");clear_folder('Videos');remove_file("url_fpt_out_put_backurl.txt");#clear_folder('Sourse_Videos')    
            check_point_id_video_completed.append(target_id_video_bil)
            with open(complete_json_path, 'a') as f:
                f.write(f'{target_id_video_bil}\n')
                f.close()             
            #run_adb_command(f'adb -s {device_port_ip} shell input keyevent 187') 
            #ti
            # me.sleep(3)
            #run_adb_command(f'adb -s {device_port_ip} shell input swipe 500 1000 500 0')
            #run_adb_command(f'adb -s {device_port_ip} shell am force-stop com.ss.android.ugc.trill')   
            run_adb_command(f'adb -s {device_port_ip} shell input keyevent 26') 
            
         
        except Exception as e:    
            print (e)
 
