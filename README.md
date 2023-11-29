# video-speech-localization
localize video speech from one language to another language, and clone the voice of the speaker.

![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202311281146734-video-speech-localization.png)

## Demo
- chinese speech to french speech

chinese speech: https://futurelog-1251943639.cos.ap-shanghai.myqcloud.com/video/linzhiling-shorts.mp4

french speech: https://futurelog-1251943639.cos.ap-shanghai.myqcloud.com/video/linzhiling-shorts-french.mp4


- chinese speech to english speech

chinese speech: https://futurelog-1251943639.cos.ap-shanghai.myqcloud.com/video/mayun-shorts-1.mp4

english speech: https://futurelog-1251943639.cos.ap-shanghai.myqcloud.com/video/mayun-shorts-1-en.mp4


## Feature
- [x] translate speech audio
- [x] clone the voice of original speaker
- [x] lip sync
- [x] subtitle
- [x] video watermark

## TODO
- [ ] support Japanese speech composition
- [ ] improve generated video quality
- [ ] face swap


## Installation
### 0. clone this repository and create nessary directories
```
git clone https://github.com/crowaixyz/video-speech-localization.git

mkdir -p output/raw_audio
mkdir -p output/raw_speech
mkdir -p output/translated_speech
mkdir -p output/lip_synced_video
mkdir -p output/final_video
mkdir -p pretrained_models
```

### 1. Install ffmpeg and ~~[Mecab](https://taku910.github.io/mecab/)~~
```
conda install -c conda-forge ffmpeg

# install additional packages in order to compose Japanese speech
# yum install mecab  # MeCab is a popular Japanese morphological analyzer
# ln -s /etc/mecabrc /usr/local/etc/mecabrc  # symbolic link
# yum install mecab-ipadic
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
Tips: use an isolated virtual environment to install Spleeter, because Spleeter may install some dependencies which may conflict with other packages(like TTS).
```
conda create -n video-speech-localization-spleeter python=3.10
conda activate video-speech-localization-spleeter

pip install spleeter

conda deactivate
```

### 4. create video-retalking virtual environment, and install [video-retalking](https://github.com/OpenTalker/video-retalking)
Tips: use an isolated virtual environment to install video-retalking, because video-retalking may install some dependencies which may conflict with other packages.
```
# clone video-retalking github repository into current directory
git clone https://github.com/vinthony/video-retalking.git
cd video-retalking

conda create -n video-speech-localization-video-retalking python=3.8
conda activate video-speech-localization-video-retalking

# Please follow the instructions from https://pytorch.org/get-started/previous-versions/
# eg. CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# eg. CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2

pip install -r requirements.txt

conda deactivate
```

## Usage
```
# activate main virtual environment
conda activate video-speech-localization

# export environment variables
export OPENAI_CHAT_API_URL=YOUR_OPENAI_CHAT_API_URL        # eg. https://api.openai.com/v1/chat/completions
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY                  # eg. sk-xxxxxxxxxxxxxxxxxxxxxxxx
export VSL_SERVER_NAME=YOUR_SERVER_NAME                    # eg. 127.0.0.1
export VSL_SERVER_PORT=YOUR_SERVER_PORT                    # eg. 7860
export CONDA_VIRTUAL_ENV_PATH=YOUR_CONDA_VIRTUAL_ENV_PATH  # eg. /data/anaconda3/envs/
export CUDA_HOME=YOUR_CUDA_HOME                            # eg. /usr/local/cuda-11.8/
export CUDA_VISIBLE_DEVICES=YOUR_CUDA_VISIBLE_DEVICES      # eg. 0
export LD_LIBRARY_PATH=YOUR_LD_LIBRARY_PATH                # eg. /usr/local/cuda/lib64::/usr/local/cuda/lib64:/usr/local/lib::$LD_LIBRARY_PATH

# run app, then open url(default http://localhost:7860/ ) in your browser
python app.py
```

## FAQ & Tips
1. manually download models in advance in order to use spleeter
- download [pretrained_models](https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz), and unarchive, then put them in ./pretrained_models/2stems.
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202311231205340.png)


2. manually download faster-whisper pretrained models and configs
- download [pretrained_models and configs](https://huggingface.co/guillaumekln/faster-whisper-large-v2/tree/main), put them in `/data/.hugggingface/cache/hub/models--guillaumekln--faster-whisper-large-v2/refs/main`
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202311231202276.png)
then load model like this:
```
whisper_model = "/data/.hugggingface/cache/hub/models--guillaumekln--faster-whisper-large-v2/refs/main"
model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
```

3. manually download coqui-xTTS pretrained models and configs
- download [pretrained_models and configs](https://huggingface.co/coqui/XTTS-v2/tree/main) and put them in `/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2`
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202311231203076.png)
- then load model like this:
```
tts = TTS(
    model_path="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/",
    config_path="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json"
).to("cpu")
```
- if you encounter error: `Model is not multi-lingual but "language" is provided.`, you can try to modify code in `/path/to/conda-envs/video-speech-localization/lib/python3.10/site-packages/TTS/api.py`
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202311231623699.png)

4. manually download models in advance in order to use video-retalking
- download  [pre-trained](https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0) models and put them in ./checkpoints under video-retalking directory. 
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202311231206734.png)
- download [detection_Resnet50_Final.pth](https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth) and [parsing_parsenet.pth](https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth), put them under `/path/to/conda-envs/video-speech-localization-video-retalking/lib/python3.8/site-packages/facexlib/weights/`
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202311231206206.png)

<!-- 5. may encounter following error when compose Janaese speech
```
File "/data/anaconda3/envs/video-speech-localization/lib/python3.10/site-packages/cutlet/cutlet.py", line 148, in __init__
    self.tagger = fugashi.Tagger(mecab_args)
  File "fugashi/fugashi.pyx", line 394, in fugashi.fugashi.Tagger.__init__
RuntimeError: Unknown dictionary format, use a GenericTagger.
```
you can try to change the `fugashi.Tagger` to `fugashi.GenericTagger`. -->