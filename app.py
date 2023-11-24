import os
import sys
import gradio as gr
import datetime
from TTS.api import TTS
import subprocess
from faster_whisper import WhisperModel
import requests
from moviepy.editor import VideoFileClip

language_map = {
    "ar": "Arabic",
    "pt": "Brazilian Portuguese",
    "zh-cn": "Chinese",
    "cs": "Czech",
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pl": "Polish",
    "ru": "Russian",
    "es": "Spanish",
    "tr": "Turkish",
    "ja": "Japanese",
    "ko": "Korean",
    "hu": "Hungarian"
}

# get current file directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))
video_length_seconds = 30

def get_video_length(video_path):
    video = VideoFileClip(video_path)
    return video.duration # 视频时长（秒）

# def get_audio_length(audio_path):
#     audio = AudioSegment.from_file(audio_path)
#     duration = len(audio) / 1000  # 音频时长（秒）
#     return duration

def update_extract_end_time(video_path):
    if video_path is not None:
        return get_video_length(video_path)
    return 0

# def update_translated_speech_audio_speed(translated_speech_audio):
#     if translated_speech_audio is not None:
#         return round(get_audio_length(translated_speech_audio) / video_length_seconds, 2)
#     return 1.0

# time segment text example:
# [0.00s -> 0.50s]: Hello, my name is wallezen.
# [0.50s -> 1.00s]: I am a software engineer.
# [1.00s -> 1.50s]: I am from China.
def time_segment_text_to_srt(text):
    lines = text.strip().split('\n')
    srt_text = []
    pure_speech_text = []
    
    for i, line in enumerate(lines):
        timestamps, content = line.split("]: ")
        start_time, end_time = timestamps.strip("[").split(" -> ")
        start_time = start_time.replace("s", "").strip()
        end_time = end_time.replace("s", "").strip()
        
        srt_entry = []
        srt_entry.append(str(i + 1))

        srt_start_time = "00:00:0" + start_time.replace('.',',') if len(start_time.split('.')[0]) == 1 else "00:00:" + start_time.replace('.',',')
        srt_end_time = "00:00:0" + end_time.replace('.',',') if len(end_time.split('.')[0]) == 1 else "00:00:" + end_time.replace('.',',')
        srt_entry.append(f"{srt_start_time} --> {srt_end_time}")

        srt_entry.append(content)
        
        srt_text.append('\n'.join(srt_entry))

        pure_speech_text.append(content)
    
    return '\n\n'.join(srt_text), ' '.join(pure_speech_text)


def extract_audio_and_text(video_path, raw_speech_language, extract_start_time_seconds, extract_end_time_seconds):
    # 上传视频，提取人声和文本
    if get_video_length(video_path) < 10:
        raise Exception("视频时长须大于 10 秒,不超过 60 秒")

    video_file_name = video_path.split("/")[-1].split(".")[0]
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    raw_audio_file_name = f"{video_file_name}_{current_time}_{extract_start_time_seconds}_{extract_end_time_seconds}.wav"
    raw_audio_file_path = f"{current_file_dir}/output/raw_audio/{raw_audio_file_name}"

    raw_speech_path = f"{current_file_dir}/output/raw_speech/"
    raw_speech_file_path = f"{raw_speech_path}/{video_file_name}_{current_time}_{extract_start_time_seconds}_{extract_end_time_seconds}/vocals.wav"
    raw_accompaniment_file_path = f"{raw_speech_path}/{video_file_name}_{current_time}_{extract_start_time_seconds}_{extract_end_time_seconds}/accompaniment.wav"

    ## step 1. use ffmpeg to extract audio from video
    duration = extract_end_time_seconds - extract_start_time_seconds
    if duration <= 0:
        raise Exception("提取时间错误，结束时间必须大于起始时间")
    if duration > 60:
        raise Exception("提取时间错误，最长提取60秒")

    extract_raw_audio_cmd = [
        "ffmpeg", 
        "-i", f"{video_path}", 
        "-ss", f"00:00:{extract_start_time_seconds}", 
        "-t", f"00:00:{duration}", 
        "-vn", 
        "-acodec", "pcm_s16le", 
        "-ar", "44100", 
        "-ac", "2", 
        f"{raw_audio_file_path}"]
    print(" ".join(extract_raw_audio_cmd))
    result = subprocess.run(extract_raw_audio_cmd, capture_output=True, text=True)
    print(result)
    # To check if the command was successful
    if result.returncode == 0:
        print("从视频中提取音频成功")
    else:
        raise Exception("从视频中提取音频失败")

    ## step 2. use spleeter to extract speech from audio
    spleeter_cmd_env = os.path.join(os.environ.get("CONDA_VIRTUAL_ENV_PATH"), "video-speech-localization-spleeter")
    extract_speech_cmd = [
        "conda", 
        "run", 
        "-p", f"{spleeter_cmd_env}", 
        "spleeter", 
        "separate", 
        "-p", "spleeter:2stems", 
        "-o", f"{raw_speech_path}", 
        f"{raw_audio_file_path}"]
    print(" ".join(extract_speech_cmd))
    result = subprocess.run(extract_speech_cmd, capture_output=True, text=True)
    print(result)
    # To check if the command was successful
    if result.returncode == 0:
        print("从音频中提取人声成功")
    else:
        raise Exception("从音频中提取人声失败")

    whisper_model = "/data/.hugggingface/cache/hub/models--guillaumekln--faster-whisper-large-v2/refs/main"
    ## step 3. call faster-whisper to recognize speech transcript from speech
    # Run on GPU with FP16
    # model = WhisperModel("large-v2", device="cuda", compute_type="float16")
    # or run on GPU with INT8
    # model = WhisperModel("large-v2", device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")

    segments, info = model.transcribe(f"{raw_speech_file_path}", beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    raw_speech_text = []
    raw_speech_text_segment = []
    for segment in segments:
        tmp = "[%.2fs -> %.2fs]: %s" % (segment.start, segment.end, segment.text)
        print(tmp)
        raw_speech_text.append(segment.text)  # TODO: reserve start and end time of every segment in order to align with original video
        raw_speech_text_segment.append(tmp)

    raw_speech_text = " ".join(raw_speech_text)
    raw_speech_text_segment = "\n".join(raw_speech_text_segment)

    return raw_speech_file_path, raw_accompaniment_file_path, raw_speech_text, raw_speech_text_segment


def translate(raw_speech_audio, raw_speech_text, raw_speech_text_segment, target_language):
    # 翻译为目标语言
    ## step 1. call chatGPT to translate speech text
    prompt = f"""
    Translate the following text to {language_map[target_language]}, add missing punctuation marks, preserving the format, not translate words between < and >, not include any other instructions in response:\n\n
    {raw_speech_text}
    """

    # request OpenAI API using OpenAI python client
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful translator."},
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # translated_speech_text = response.choices[0].message.content

    # request OpenAI API using requests
    openai_chat_api_url = os.environ.get("OPENAI_CHAT_API_URL")
    headers = {
        "Authorization": os.environ.get("OPENAI_API_KEY"),  # FIXME: change this to "Bear <YOUR_OPENAI_API_KEY>" if you want to request OpenAI API directly
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful translator for localizing advertisement creatives."},
            {"role": "user", "content": f"{prompt}"}
        ]
    }

    translated_speech_text = ""
    try:
        with requests.post(openai_chat_api_url, headers=headers, json=payload) as response:
            if response.status_code == 200:
                response_data = response.json()
                print(response_data)
                translated_speech_text = response_data["choices"][0]["message"]["content"]
            else:
                # print("请求 iGateway OpenAI Chat 接口出错: ", response)
                raise Exception("请求 iGateway OpenAI Chat 接口出错: ", response)
    except requests.exceptions.RequestException as e:
        # print("request openai failed:", e)
        raise Exception("请求 iGateway OpenAI Chat 接口失败: ", e)

    if translated_speech_text == "":
        raise Exception("请求 iGateway OpenAI 翻译失败")

    ## step 2. call chatGPT to translate speech text segment
    prompt = f"""
    Translate the following text to {language_map[target_language]}, add missing punctuation marks, preserving the format, not translate words between < and >, not include any other instructions in response:\n\n
    {raw_speech_text_segment}
    """

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful translator for localizing advertisement creatives."},
            {"role": "user", "content": f"{prompt}"}
        ]
    }

    translated_speech_text_segment = ""
    try:
        with requests.post(openai_chat_api_url, headers=headers, json=payload) as response:
            if response.status_code == 200:
                response_data = response.json()
                print(response_data)
                translated_speech_text_segment = response_data["choices"][0]["message"]["content"]
            else:
                # print("请求 iGateway OpenAI Chat 接口出错: ", response)
                raise Exception("请求 iGateway OpenAI Chat 接口出错: ", response)
    except requests.exceptions.RequestException as e:
        # print("request openai failed:", e)
        raise Exception("请求 iGateway OpenAI Chat 接口失败: ", e)

    if translated_speech_text_segment == "":
        raise Exception("请求 iGateway OpenAI 翻译按时间分段文本失败")

    return translated_speech_text, translated_speech_text_segment


def compose_target_language_audio(raw_speech_audio, translated_speech_text, translated_speech_text_segment, target_language, speech_speed):
    # 合成目标语言人声
    ## use coqui-xTTS-V2 to synthesize target language speech audio and clone raw speech tone
    raw_speech_file_name = raw_speech_audio.split("/")[-2] + "_" + raw_speech_audio.split("/")[-1].split(".")[0]
    translated_speech_file_path = f"{current_file_dir}/output/translated_speech/{target_language}_{raw_speech_file_name}.wav"
    translated_speech_srt_file_path = f"{current_file_dir}/output/translated_speech/{target_language}_{raw_speech_file_name}.srt"

    # write translated speech text to srt file
    srt_text, pure_speech_text = time_segment_text_to_srt(translated_speech_text_segment)
    print(pure_speech_text)
    with open(translated_speech_srt_file_path, "w", encoding="utf-8") as f:
        f.write(srt_text)

    # use TTS to synthesize target language speech audio
    # Init coqui 🐸TTS
    tts = TTS(
        #model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        model_path="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/",
        config_path="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json"
    ).to("cuda")

    # Run TTS
    # ❗ Since xtts_v2 model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # Text to speech list of amplitude values as output
    # wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
    # Text to speech to a file
    tts.tts_to_file(text=f"{pure_speech_text}", speaker_wav=f"{raw_speech_audio}", language=f"{target_language}", file_path=f"{translated_speech_file_path}", speed=speech_speed)

    return translated_speech_file_path, translated_speech_srt_file_path


def compose_lip_sync_video(original_video, translated_speech_audio):
    # 合成口型对齐视频
    ## use video-retalking to generate lip-synced video
    lip_sync_video_file_path = f"{current_file_dir}/output/lip_synced_video/{original_video.split('/')[-1].split('.')[0]}-{translated_speech_audio.split('/')[-1].split('.')[0]}.mp4"

    video_retalking_env = os.path.join(os.environ.get("CONDA_VIRTUAL_ENV_PATH"), "video-speech-localization-video-retalking")
    compose_lip_sync_video_cmd = [
        # "export", "CUDA_HOME=/usr/local/cuda-11.8/",
        # "&&",
        # "export", "CUDA_VISIBLE_DEVICES=4",
        # "&&",
        # "cd", f"{current_file_dir}/video-retalking", 
        # "&&",
        "conda", 
        "run", 
        "-p",  f"{video_retalking_env}", 
        "python",
        "inference.py",
        "--face",  f"{original_video}",
        "--audio", f"{translated_speech_audio}",
        "--outfile", f"{lip_sync_video_file_path}"
    ]
    video_retalking_env_vars = {
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH")
    }
    video_retalking_workdir = f"{current_file_dir}/video-retalking"

    print(" ".join(compose_lip_sync_video_cmd))
    result = subprocess.run(compose_lip_sync_video_cmd, capture_output=True, text=True, cwd=video_retalking_workdir, env=video_retalking_env_vars)
    print(result)
    # To check if the command was successful
    if result.returncode == 0:
        print("合成口型对齐视频成功")
    else:
        raise Exception("合成口型对齐视频失败") 

    return lip_sync_video_file_path


def compose_final_video_without_lip_sync(original_video, translated_speech_audio, raw_accompaniment_audio, translated_speech_srt):
    # 合成最终视频
    finale_video_file_path = f"{current_file_dir}/output/final_video/{translated_speech_audio.split('/')[-1].split('.')[0]}.mp4"
    ## use ffmpeg to replace original video speech with translated speech, and mix with raw accompaniment audio
    compose_cmd = [
        "ffmpeg", 
        "-i", f"{original_video}", 
        "-i", f"{translated_speech_audio}", 
        "-i", f"{raw_accompaniment_audio}", 
        "-vf", f"subtitles={translated_speech_srt},drawtext=text='wallezen@CrowAI':x=10:y=10:fontsize=24:fontcolor=white:fontfile=/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "-filter_complex", "[1:a]atempo=1.0[a1];[2:a]atempo=1.0[a2];[a1][a2]amix=inputs=2[a]", 
        "-map", "0:v",
        "-map", "[a]",
        "-c:a", "aac",
        "-ac", "2",
        f"{finale_video_file_path}"
    ]
    print(" ".join(compose_cmd))
    result = subprocess.run(compose_cmd, capture_output=True, text=True)
    print(result)
    # To check if the command was successful
    if result.returncode == 0:
        print("合成最终视频成功")
    else:
        raise Exception("合成最终视频失败")

    return finale_video_file_path


def compose_final_video_with_lip_sync(lip_sync_video, raw_accompaniment_audio, translated_speech_srt):
    # 合成最终视频
    finale_video_file_path = f"{current_file_dir}/output/final_video/{lip_sync_video.split('/')[-1].split('.')[0]}.mp4"
    ## use ffmpeg to replace original video speech with translated speech
    compose_cmd = [
        "ffmpeg", 
        "-i", f"{lip_sync_video}", 
        "-i", f"{raw_accompaniment_audio}", 
        "-vf", f"subtitles={translated_speech_srt},drawtext=text='wallezen@CrowAI':x=10:y=10:fontsize=24:fontcolor=white:fontfile=/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "-filter_complex", "[0:a][1:a]amerge=inputs=2[a]",
        "-map", "0:v",
        "-map", "[a]",
        "-c:a", "aac",
        "-ac", "2",
        f"{finale_video_file_path}"]
    print(" ".join(compose_cmd))
    result = subprocess.run(compose_cmd, capture_output=True, text=True)
    print(result)
    # To check if the command was successful
    if result.returncode == 0:
        print("合成最终视频成功")
    else:
        raise Exception("合成最终视频失败")

    return finale_video_file_path

def compose_final_video(original_video, translated_speech_audio, raw_accompaniment_audio, translated_speech_srt, lip_sync_radio, subtitle_radio):
    # compose final video
    # step 1. compose lip-synced video
    lip_sync_video = ""
    if lip_sync_radio == "是":
        lip_sync_video = compose_lip_sync_video(original_video, translated_speech_audio)

    finale_video_file_path = ""
    if lip_sync_video != "":
        finale_video_file_path = compose_final_video_with_lip_sync(lip_sync_video, raw_accompaniment_audio, translated_speech_srt)
    else:
        finale_video_file_path = compose_final_video_without_lip_sync(original_video, translated_speech_audio, raw_accompaniment_audio, translated_speech_srt)

    if finale_video_file_path == "":
        raise Exception("合成最终视频失败")

    return finale_video_file_path

with gr.Blocks() as app:
    gr.Markdown("## 视频本地化 Demo [Github](https://github.com/crowaixyz/video-speech-localization)")
    # step 1. 上传视频
    gr.Markdown("### Step 1. 上传视频")
    with gr.Row():
        original_video = gr.Video(label="原始视频(注意视频时长须大于10秒，不超过60秒, 30~60秒为佳)")
        with gr.Column():
            original_videl_speech_language = gr.Dropdown(choices=["ar","pt","zh-cn","cs","nl","en","fr","de","it","pl","ru","es","tr","ja","ko","hu"], label="视频人声语言")
            gr.Markdown("ar:    Arabic <br />pt: Brazilian    Portuguese <br />zh-cn: Chinese <br />cs:    Czech <br />nl:    Dutch <br />en:    English <br />fr:    French <br />de:    German <br />it:    Italian <br />pl:    Polish <br />ru:    Russian <br />es:    Spanish <br />tr:    Turkish <br />ja:    Japanese <br />ko:    Korean <br />hu: Hungarian")

    # step 2. 提取人声，文本和背景音乐
    gr.Markdown("### Step 2. 提取人声，文本和背景音乐")
    with gr.Row():
        with gr.Column():
            extract_start_time_seconds = gr.Slider(label="提取起始时间(秒)", minimum=0.0, maximum=60.0, value=0.0, interactive=False)
            extract_end_time_seconds = gr.Slider(label="提取结束时间(秒)", minimum=0.0, maximum=60.0, value=30.0, interactive=False)

        audio_extract_button = gr.Button("点击提取")

    raw_speech_audio = gr.Audio(label="人声", type="filepath", interactive=False)
    raw_accompaniment_audio = gr.Audio(label="背景音乐", type="filepath", interactive=False)
    raw_speech_text_segment = gr.Textbox(label="按时间分段的人声文本（可修改，对于不需要进行后续翻译的词使用'<>'）", interactive=True)
    raw_speech_text = gr.Textbox(label="完整不分段的人声文本")

    # step 3. 翻译为目标语言
    gr.Markdown("### Step 3. 翻译为目标语言")
    with gr.Row():
        target_speech_language = gr.Dropdown(choices=["ar","pt","zh-cn","cs","nl","en","fr","de","it","pl","ru","es","tr","ja","ko","hu"], label="选择目标语言")
        translate_button = gr.Button("点击翻译")

    translated_speech_text_segment = gr.Textbox(label="翻译后的按时间分段的人声文本(可修改)", interactive=True)
    translated_speech_text = gr.Textbox(label="翻译后的完整不分段的人声文本", interactive=False)

    # step 4. 合成目标语言人声
    gr.Markdown("### Step 4. 合成目标语言人声")
    with gr.Row():
        speech_speed = gr.Slider(label="调整人声语速", minimum=0.5, maximum=2.0, value=1.0, step=0.01, interactive=True, info="通过调整语速，尽量保证生成的语音和原视频时长一致")
        compose_target_language_audio_button = gr.Button("点击合成")

    translated_speech_audio = gr.Audio(label="目标语言人声", type="filepath", interactive=False)
    translated_speech_srt = gr.File(label="目标语言字幕 SRT", type="filepath", interactive=False)

    # Step 5. 合成最终视频
    gr.Markdown("### Step 5. 合成最终视频")
    with gr.Row():
        with gr.Column():
            lip_sync_radio = gr.Radio(choices=["是", "否"], label="是否对齐口型", value="否", interactive=True, info="仅适合单人说话场景，需要始终保持露脸，且耗时较长！")
            subtitle_radio = gr.Radio(choices=["是", "否"], label="是否添加字幕", value="是", interactive=False, info="目标语言字幕")
        compose_final_video_button = gr.Button("点击合成")
    with gr.Row():
        final_video = gr.Video(label="最终视频", interactive=False)

    # 回调事件
    ## 上传视频后，自动获取视频时长，并更新提取结束时间 slider 组件的值
    original_video.change(fn=update_extract_end_time, inputs=original_video, outputs=extract_end_time_seconds)
    
    # 处理按钮点击事件
    audio_extract_button.click(
        extract_audio_and_text,
        inputs=[original_video, original_videl_speech_language, extract_start_time_seconds, extract_end_time_seconds],
        outputs=[raw_speech_audio, raw_accompaniment_audio, raw_speech_text, raw_speech_text_segment]
    )
    translate_button.click(
        translate,
        inputs=[raw_speech_audio, raw_speech_text, raw_speech_text_segment, target_speech_language],
        outputs=[translated_speech_text, translated_speech_text_segment]
    )
    compose_target_language_audio_button.click(
        compose_target_language_audio,
        inputs=[raw_speech_audio, translated_speech_text, translated_speech_text_segment, target_speech_language, speech_speed],
        outputs=[translated_speech_audio, translated_speech_srt]
    )
    compose_final_video_button.click(
        compose_final_video,
        inputs=[original_video, translated_speech_audio, raw_accompaniment_audio, translated_speech_srt, lip_sync_radio, subtitle_radio],
        outputs=[final_video]
    )

app.launch(server_name=os.environ.get("VSL_SERVER_NAME"), server_port=int(os.environ.get("VSL_SERVER_PORT")), share=False)
