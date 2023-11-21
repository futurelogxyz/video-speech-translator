# video-speech-localization
localize video speech from one language to another language, and clone the voice of the speaker.

## Feature
- [x] localize video speech from one language to another language
- [x] clone the voice of the speaker

## TODO
- [ ] lip sync

## Installation
### 1. Install ffmpeg
```
conda install -c conda-forge ffmpeg
``` 

### 2. create main virtual environment, and install dependencies [faster-whisper](https://github.com/guillaumekln/faster-whisper) and ~~[openai](https://platform.openai.com/docs/introduction)~~ and [Requests](https://requests.readthedocs.io/en/latest/) and [Gradio](https://www.gradio.app/) and [TTS](https://github.com/coqui-ai/TTS) and [moviepy](https://github.com/Zulko/moviepy) and ~~[pydub](https://github.com/jiaaro/pydub)~~
```
conda create -n video-speech-localization python=3.10
conda activate video-speech-localization

pip install faster-whisper

pip install requests

pip install gradio

pip install TTS

pip install moviepy

conda deactivate
```


### 3. create spleeter virtual environment, and install [Spleeter](https://github.com/deezer/spleeter)
```
conda create -n video-speech-localization-spleeter python=3.10
conda activate video-speech-localization-spleeter

pip install spleeter

conda deactivate
```

## Usage
```
# activate main virtual environment
conda activate video-speech-localization

# export environment variables
export OPENAI_CHAT_API_URL=YOUR_OPENAI_CHAT_API_URL  # eg. https://api.openai.com/v1/chat/completions
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY            # eg. sk-xxxxxxxxxxxxxxxxxxxxxxxx
export SERVER_NAME=YOUR_SERVER_NAME                  # eg. 127.0.0.1
export SERVER_PORT=YOUR_SERVER_PORT                  # eg. 7860

# run app, then open url(default http://localhost:7860/ ) in your browser
python app.py
```