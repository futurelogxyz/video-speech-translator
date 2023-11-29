from TTS.api import TTS

def main():
    tts = TTS(
        #model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        model_path="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/",
        config_path="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json"
    ).to("cpu")

    # Run TTS
    # ❗ Since xtts_v2 model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # Text to speech list of amplitude values as output
    # wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
    # Text to speech to a file
    tts.tts_to_file(
        text="彼らはたった一部の、多分社交の場でしか、 あるいは夜の生活のような場でしか、 自分をゆるんでみたり、 解放してみたりすることができる。", 
        speaker_wav="output/raw_speech/linzhiling-shorts_20231129-113004_0_11.5/vocals.wav", 
        language="ja", 
        file_path="output.mp4"
    )

if __name__ == "__main__":
    main()
