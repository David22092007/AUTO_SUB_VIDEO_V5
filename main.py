import os
import sys
import json
import glob
import uuid
import time
import ffmpeg
import random
import shutil
import logging
import asyncio
import edge_tts
import threading
import subprocess
from queue import Queue
import concurrent.futures
from datetime import timedelta
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
import moviepy.editor as mp

from pydub import AudioSegment
import librosa
import webrtcvad
import numpy as np
from scipy.io.wavfile import write
import soundfile as sf
import requests
import re
from google import genai

# Cấu hình logging
logging.basicConfig(filename='dub_movie.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
            offset_sec = int(os.path.basename(srt_path).split('_')[1].split('.')[0])
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
                continue
        i += 1
    return metadata_list

def transcript_video(audio_file_path, checkpoint_file, temp_dir, srt_subtitle_output_path, metadata_list, segment_index=0, MODEL='base'):
    setup_directories(temp_dir)
    segment_id = str(uuid.uuid4())
    try:
        for audio in audio_file_path:
            print('EXTRACTING SPEECH CONTENT:', audio)
            srt_filename = os.path.basename(audio).replace('.aac', '.srt')
            srt_path = os.path.join(srt_subtitle_output_path, srt_filename)
            cmd = (
                f"whisper {audio} --model {MODEL} --output_format srt "
                f"--language {source_language} --fp16 False --output_dir {srt_subtitle_output_path} --threads 10"
            )
            os.system(cmd)
            metadata_list.extend(filter_srt_detail(srt_path))
            save_checkpoint(checkpoint_file, {"segments": metadata_list})
        return metadata_list
    except Exception as e:
        logging.error(f"Error processing segment: {e}")
    finally:
        if isinstance(audio_file_path, list):
            for path in audio_file_path:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logging.info(f"Deleted temp file: {path}")
                    except Exception as e:
                        logging.warning(f"Could not delete temp file {path}: {e}")

def get_video_duration(video_file):
    probe = ffmpeg.probe(video_file)
    return float(probe['format']['duration'])

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
        run_threads(audio_files_path, output_json_path, temp_dir, srt_subtitle_output_path, source_language, metadata_list, num_threads=10)
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
        """Lấy API key khả dụng, ưu tiên key chưa dùng"""
        # Ưu tiên key chưa dùng
        unused_keys = [key for key in self.api_keys if key not in self.used_keys]
        if unused_keys:
            return unused_keys[0]
        
        # Nếu không có key chưa dùng, lấy từ queue
        if not self.available_keys.empty():
            return self.available_keys.get()
        
        # Nếu tất cả key đều bị lỗi, raise exception
        if len(self.failed_keys) == len(self.api_keys):
            raise Exception("All API keys have failed")
        
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
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            api_key = self.get_available_key()
            if api_key is None:
                logging.error("No available API keys")
                return None
            
            try:
                result = self.translate_segment(segment, api_key)
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
            
            if response is None or not hasattr(response, 'text') or response.text is None:
                return None
            
            # Tạo tên file output
            base_name = os.path.basename(segment)
            try:
                segment_number = base_name.split('_')[1]
            except IndexError:
                segment_number = base_name.split('.')[0]
            
            file_name = f'segment_{segment_number}'
            final_sup_text_path = os.path.join(self.output_sub_dir, file_name)
            
            # Ghi file
            with open(final_sup_text_path, 'w', encoding="utf-8") as f:
                f.write(response.text)
            
            return final_sup_text_path
            
        except Exception as e:
            logging.error(f"Error translating segment {segment}: {str(e)}")
            return None

    def translated_with_gemini(self, api_key, srt_content):
        """Gọi API Gemini để dịch"""
        try:
            client = genai.Client(api_key=api_key)
            prompt = (
                """
                -Bạn là một công cụ dịch thuật. Nhiệm vụ của bạn là:
                -1. **Không được giữ lại bất kỳ ký tự gốc nào của văn bản gốc (ví dụ tiếng Trung, tiếng Anh...) trong kết quả.**
                -2. **Một số tên được dịch không hay bạn có thể biến tên nhân vật theo phong cách Trung Hoa cổ trang, sử dụng cấu trúc "Tiểu + [Tên Nhân Vật]" . Mỗi tên cần thể hiện tính cách, ngoại hình, hoặc vai trò của nhân vật.
                -3. **Tôi nhấn mạnh lại một điểm phải giữ nguyên định dạng gốc . Ví dụ về một lỗi mà bạn thường sai  00:01 nhưng phải điền là 00:00:01,000 .Hãy chú ý vào những lỗi nhỏ đó.
                -LƯU Ý. **Có số chổ là do cách phát âm & những từ đồng nghĩa làm cho khi dịch câu trở nên sai các phương pháp về từ như sai logic ,sai ngữ pháp , sai cách dùng từ .Bạn hãy xem và chỉnh sữa những điểm đó lại bằng trí tuệ của bạn.Không cần phải chú thích gì cả 
        
                Kết quả dịch (giữ nguyên định dạng thay thế văn bản gốc bằng văn bản đã dịch không thêm gì khác):
                """
            )            
            for name_model in ['gemini-1.5-flash','gemini-2.0-flash-lite','gemini-2.5-flash','gemini-2.5-flash-lite']:
                try:
                    response = client.models.generate_content(
                        model=name_model,
                        contents=[srt_content, prompt]
                    )
                    if response.text is not None:
                        return response
                except Exception as e:
                    logging.warning(f"Model {name_model} failed: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            logging.error(f"Gemini API call failed: {str(e)}")
            return None

    def run_translation(self):
        """Chạy dịch thuật với quản lý API keys thông minh"""
        logging.info(f"Starting translation with {len(self.api_keys)} API keys for {len(self.segment_groups)} segments")
        
        # Số luồng tối đa bằng số segment groups
        max_workers = min(len(self.segment_groups), 20)  # Giới hạn 20 luồng để tránh quá tải
        
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
        logging.error(f"Translation failed: {str(e)}")
        return []
#-------------------------------------------------

def speed_up_video(input_video_path, speed, output_dir):
    subprocess.run([
        'ffmpeg', '-i', input_video_path,
        '-y',
        '-filter:a', f'atempo={speed}',
        '-vn',
        output_dir
    ], check=True)

def get_duration(input_path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ]
    return float(subprocess.check_output(cmd))

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
            elif translate_mode == 'fpt':                
                fpt_tts(translated_text, final_audio_path, duration_ms/1000,  api_keys, speed='1')
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

def detect_langue(audio_path, model_AI='tiny'):
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
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, pitch=pitch)
        out_put_base = output_path.replace('/tts', '/tts_base')
        await communicate.save(out_put_base)
        duration_timeslap = get_duration(out_put_base)
        try:
            speed_up_video(out_put_base, max(1, min(1.5, float(duration_timeslap) / float(duration_ms)), output_path))
        except Exception as e:
            logging.error(f"Speed adjust error: {e}")
        print(f"Generated TTS file: {output_path}")
    except Exception as e:
        print(f"Error generating TTS for {output_path}: {str(e)}")

async def text_api_to_speech(translated_text, output_path, duration_ms, gender='Female'):
    print('***************************************\n', translated_text, output_path)
    voice = "vi-VN-HoaiMyNeural" if gender == 'Female' else "vi-VN-NamMinhNeural"
    pitch = "-0Hz"
    await generate_tts_with_pitch_and_rate(pitch, output_path, translated_text, voice, duration_ms/1000)

def fpt_tts(text, output_path, duration_ms, api_keys, speed, voice_name='banmai'):
    if len(text) <3:
        text = str(text)+' !'
    while True:
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
def thread_saving_video_fpt(detail):
    output_path_part_1,output_path_part_2,url,duration_ms=detail.split('_');output_path=output_path_part_1+'_'+output_path_part_2;duration_ms=duration_ms.replace('\n','')
    out_put_base = output_path.replace('/tts', '/tts_base')
    for i in range (100):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(out_put_base, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            duration_timeslap = get_duration(out_put_base);speed_up_video(out_put_base, max(1, min(1.3, float(duration_timeslap) / float(duration_ms))), output_path)
            return
        else:
            continue
def dub_movie(input_video_path, output_dir, api_keys, source_language, target_language, full_option,type_tts='fpt'):
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
    
    print("Step 1: Extracting speech content...")
    if not check_volume_of_file(checkpoint_transcript_file, 2048):
        metadata_list = load_checkpoint(checkpoint_transcript_file)['segments']
    
    extract_transcript(input_video_path, output_dir, checkpoint_transcript_file, source_language, metadata_list)
    
    print("Step 2: Translating text to Vietnamese...")
    subed_srt_path = glob.glob(os.path.join(output_sub_dir, '*.srt'))
    
    srt_files = glob.glob(os.path.join('srt', '*.srt'))
    metadata_list = []
    
    for i in subed_srt_path:
        srt_files.remove(i.replace('sub', 'srt'))
    
    if srt_files:
        translate_with_gemini(api_keys, srt_files, target_language, output_sub_dir)
    
    for srt_path in glob.glob(os.path.join(output_sub_dir, '*.srt')):
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
        metadata_list = [i for i in metadata_list if i['start_time'] not in list_start_time_complete]
        max_workers = min(os.cpu_count() or 2, 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_segment = {
                executor.submit(
                    process_segment_dubbing,
                    segment["start_time"],
                    segment["end_time"],
                    segment["start_time"],
                    segment["text"],
                    target_language,
                    tts_forder_save_path,
                    ['n9rKtRYQWtT39nCBil4o9JsQsvZEyRDP','WDJxtVTv3EOUbpeXWszOd6WkxWs9LGJQ','pjLFlJzmN4yaYcZPHP2wwMMIx0Pn5lBh'],
                    type_tts
                ): segment
                for segment in metadata_list
            }
        # Đọc danh sách URL nếu sử dụng type_tts là 'fpt'
        if len(metadata_list)!=0:
            if type_tts == 'fpt':
                with open('url_fpt_out_put_backurl.txt', 'r') as f:
                    list_url_fpt = f.readlines() 
                with ThreadPoolExecutor(max_workers=10) as executor:
                    executor.map(thread_saving_video_fpt, list_url_fpt)          
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
if __name__ == "__main__":
    # Cấu hình biến toàn cục
    select_options = '123'
    full_option = True
    id_video = ''
    file_infor_json_file_path = 'sourse_video_infor.json'
    complete_json_path = 'video_id_completed.txt'
    lauching_file_path = 'running_id_video.txt'
    srt_content = ""
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
    
    while True:
        # Kiểm tra video đã hoàn thành
        if os.path.exists(complete_json_path):
            with open(complete_json_path, 'r', encoding='utf-8') as f:
                check_point_id_video_completed = [value.strip() for value in f.readlines()]
                name_video = f'[ TU TIÊN ] VÔ TÌNH THÔN PHỆ KẺ ĐỊCH TA ĐỘT PHÁ ĐẠI VIÊN MÃN - TẬP {len(check_point_id_video_completed)+1}'
        else:
            name_video = '[ TU TIÊN ] VÔ TÌNH THÔN PHỆ KẺ ĐỊCH TA ĐỘT PHÁ ĐẠI VIÊN MÃN - TẬP 1'
            check_point_id_video_completed = []
        
        # Đọc danh sách video từ file JSON
        with open(file_infor_json_file_path, 'r', encoding='utf-8') as f:
            list_id_video_bil = [value['id_video'] for value in json.load(f)]
        
        # Kiểm tra video đang xử lý
        if os.path.exists(lauching_file_path):
            with open(lauching_file_path, 'r', encoding='utf-8') as f:
                id_video = f.read().strip()
        
        # Chọn video tiếp theo để xử lý
        target_id_video_bil = [vid for vid in list_id_video_bil if vid not in check_point_id_video_completed][0]
        
        if id_video != target_id_video_bil:
            # Tải video mới nếu cần
            cmd = [
                "yt-dlp",
                f"https://www.bilibili.com/video/{target_id_video_bil}",
                "--cookies", "cookies.txt",
                "-N", "8",
                "--concurrent-fragments", "4",
                "--throttled-rate", "100K",
                "-o", "Videos/%(title)s.%(ext)s",
                "--retries", "10",
                "--fragment-retries", "10",
                "--buffer-size", "16K",
                "--http-chunk-size", "1M",
            ]
            subprocess.run(cmd, check=True)
            
            # Đổi tên file video
            video_path = glob.glob("Videos/*.mp4")[0]
            new_name = f'Videos/{target_id_video_bil}.mp4'
            os.rename(video_path, new_name)
            
            # Lưu ID video đang xử lý
            with open(lauching_file_path, 'w', encoding='utf-8') as f:
                f.write(target_id_video_bil)
        
        if select_options == '12':
            full_option = True        
        input_video = f"Videos/{target_id_video_bil}.mp4"
        decrease_video_path = input_video.replace(os.path.basename(input_video), 'decrease_video.mp4')
        final_video_path = input_video.replace(os.path.basename(input_video), f'final_{target_id_video_bil}_video.mp4')
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
        dub_movie(input_video, output_dir, api_keys, source_language, target_language, full_option)
        with open(os.path.join('dubbed_movie', 'translated_transcript.json'), 'r', encoding='utf-8') as file:
            segments = json.load(file)['segments'];file.close()            
        srt_content = convert_detail(segments, srt_content, counter)
        srt_path = os.path.join('dubbed_movie', f'translated_{str(len(check_point_id_video_completed)+1)}_subs.srt')
        with open(srt_path, 'a', encoding='utf-8') as f:
            f.write(srt_content)
        print(f"File phụ đề SRT đã được tạo tại: {srt_path}")
        success = increase_audio_volume(input_video, decrease_video_path, volume_factor=0.05);file_path = 'checkpoint_dub_demo.mp4.json';output_path_voice_compare = os.path.join('Videos', 'output_video.mp4')
        # Đọc dữ liệu từ file JSON
        with open(f"checkpoint_dub_{os.path.basename(input_video)}.json", 'r', encoding='utf-8') as file:
            data = [o for o in json.load(file)['segments'] if o != None]
            file.close()
        # Tạo danh sách các input audio và thời gian bắt đầu
        audio_inputs = []
        for segment in data:
            index = segment['start_time']
            start_time = segment['start_time']
            audio_path = segment['output_path']
            
            # Kiểm tra xem file audio có tồn tại không
            if os.path.exists(audio_path):
                audio_inputs.append({
                    'audio_path': audio_path,
                    'start_time': start_time
                })
            else:
                print(f"File {audio_path} không tồn tại, bỏ qua đoạn này.")

        # Xây dựng lệnh ffmpeg
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', decrease_video_path,  # Input video (có audio gốc)
        ]

        # Thêm tất cả file audio làm input
        for audio in audio_inputs:
            ffmpeg_cmd.extend(['-i', audio['audio_path']])

        # Tạo bộ lọc phức tạp (complex filter)
        filter_complex = ''
        # Giữ nguyên âm lượng gốc, không chỉnh
        filter_complex += '[0:a]anull[a0];'

        # Chỉ chèn audio đúng thời điểm, không tăng volume
        for i, audio in enumerate(audio_inputs):
            delay_ms = round(float(audio["start_time"]) * 1000)
            filter_complex += f'[{i+1}:a]adelay={delay_ms}|{delay_ms}[a{i+1}];'

        # Gộp tất cả audio (audio gốc + các audio được thêm)
        filter_complex += '[a0]' + ''.join(f'[a{i+1}]' for i in range(len(audio_inputs))) + f'amix=inputs={len(audio_inputs)+1}:normalize=0[aout]'
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

        # In lệnh ffmpeg để kiểm tra (tùy chọn)
        print('Lệnh ffmpeg:', ' '.join(ffmpeg_cmd))

        # Chạy lệnh ffmpeg
        try:
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print(f"Video đã được xử lý và lưu tại {output_path_voice_compare}")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Lỗi khi chạy ffmpeg: {e.stderr}")
        except FileNotFoundError:
            print("Lỗi: Không tìm thấy ffmpeg. Hãy đảm bảo ffmpeg đã được cài đặt và thêm vào PATH.")

        increase_audio_volume(output_path_voice_compare, final_video_path, volume_factor=1.5)
        if False:    
            subtitle_file = srt_path
            # File đầu ra
            output_video = os.path.join("Videos", f"subed_{target_id_video_bil}_video.mp4")
            # Câu lệnh FFmpeg dưới dạng list (tránh lỗi escape)
            cmd = [
                "ffmpeg",
                "-i", final_video_path,
                "-vf", f"subtitles={subtitle_file}:force_style='FontName=STIXNonUnicode,FontSize=12,PrimaryColour=&H00FFFF&,Outline=2,Shadow=1'",
                "-c:v", "libx264",  # Bộ mã hóa H.264
                "-crf", "28",       # Chất lượng (18-28 là tốt, cao hơn = nhỏ hơn)
                "-preset", "slow",  # Tốt hơn 'ultrafast' để nén tốt hơn
                "-tune", "film",    # Tối ưu cho video thông thường
                "-movflags", "+faststart",  # Tối ưu cho phát trực tuyến
                "-c:a", "aac",      # Mã hóa lại audio sang AAC
                "-b:a", "128k",     # Bitrate audio
                "-threads", "0",    # Tự động chọn số luồng
                "-f", "mp4",        # Định dạng đầu ra
                "-y",
                output_video
            ]
            # Chạy lệnh và hiển thị log
            try:
                subprocess.run(cmd, check=True)
                print("✅ Hoàn tất! Video đã được chèn phụ đề.")
            except subprocess.CalledProcessError as e:
                print("❌ Lỗi khi chạy FFmpeg:", e)              
        while True:
            with open(final_video_path, 'rb') as video_file:
                # Tạo dictionary cho dữ liệu và files
                files = {
                    "video": video_file # Khóa 'video' là bắt buộc cho phương thức sendVideo
                }
                data = {
                    "chat_id": CHAT_ID,
                    "caption": name_video # Chú thích (không bắt buộc)
                }

                # Gọi API sendVideo
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
                response = requests.post(url, data=data, files=files)
            if response.status_code==200:
                with open(srt_path, 'rb') as file_path_txt:
                    # Tạo dictionary cho dữ liệu và files
                    files = {
                        "video": file_path_txt # Khóa 'video' là bắt buộc cho phương thức sendVideo
                    }
                    data = {
                        "chat_id": CHAT_ID,
                        "caption": 'SRT : '+name_video # Chú thích (không bắt buộc)
                    }

                    # Gọi API sendVideo
                    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
                    response = requests.post(url, data=data, files=files)
                if response.status_code==200:
                    break 
        remove_file(f"checkpoint_dub_{os.path.basename(input_video)}.json");clear_folder('tts');clear_folder('sub');clear_folder('srt');remove_file(f"checkpoint_transcript_{os.path.basename(input_video)}.json");clear_folder("temp_segments");clear_folder('Videos');remove_file("url_fpt_out_put_backurl.txt")      
        with open(complete_json_path, 'a') as f:
            f.write(f'{target_id_video_bil}\n')
            f.close()


