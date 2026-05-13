import streamlit as st
import whisperx
import torch
import tempfile
import os
import re
from difflib import SequenceMatcher

st.set_page_config(
    page_title="Subtitle Generator",
    layout="centered"
)

st.title("German Subtitle Generator")

st.write("Upload MP3 + TXT script")

uploaded_audio = st.file_uploader(
    "Upload MP3",
    type=["mp3", "wav"]
)

uploaded_script = st.file_uploader(
    "Upload TXT Script",
    type=["txt"]
)

MAX_CHARS = st.slider(
    "Caption Width",
    10,
    40,
    20
)

if uploaded_audio and uploaded_script:

    if st.button("Generate Subtitles"):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with st.spinner("Uploading files..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp3"
            ) as temp_audio:

                temp_audio.write(uploaded_audio.read())
                audio_path = temp_audio.name

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".txt"
            ) as temp_txt:

                temp_txt.write(uploaded_script.read())
                txt_path = temp_txt.name

        with st.spinner("Loading WhisperX model..."):

            model = whisperx.load_model(
                "large-v2",
                device,
                language="de"
            )

        audio = whisperx.load_audio(audio_path)

        with st.spinner("Transcribing audio..."):

            result = model.transcribe(audio)

        with st.spinner("Aligning subtitles..."):

            model_a, metadata = whisperx.load_align_model(
                language_code="de",
                device=device
            )

            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device
            )

        with open(txt_path, "r", encoding="utf-8") as f:

            script_text = f.read()

        # REMOVE TIMESTAMPS
        script_text = re.sub(
            r'\d{1,2}:\d{2}(?::\d{2})?',
            '',
            script_text
        )

        # CLEAN SPACES
        script_text = re.sub(
            r'\s+',
            ' ',
            script_text
        )

        script_words = script_text.split()

        audio_words = result["word_segments"]

        # SMART WORD CORRECTION
        for i in range(min(len(audio_words), len(script_words))):

            if "word" not in audio_words[i]:
                continue

            transcribed = audio_words[i]["word"]
            original = script_words[i]

            similarity = SequenceMatcher(
                None,
                transcribed.lower(),
                original.lower()
            ).ratio()

            # ONLY REPLACE SIMILAR WORDS
            if similarity > 0.65:

                audio_words[i]["word"] = original

        # FORMAT TIME
        def format_time(seconds):

            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)

            return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

        # CREATE SINGLE-LINE CHUNKS
        chunks = []
        current_chunk = []
        current_length = 0

        for word in audio_words:

            if "word" not in word:
                continue

            word_length = len(word["word"]) + 1

            if current_length + word_length > MAX_CHARS:

                chunks.append(current_chunk)
                current_chunk = []
                current_length = 0

            current_chunk.append(word)
            current_length += word_length

        if current_chunk:
            chunks.append(current_chunk)

        # GENERATE SRT
        srt_content = ""

        index = 1

        for chunk in chunks:

            if "start" not in chunk[0]:
                continue

            if "end" not in chunk[-1]:
                continue

            start = chunk[0]["start"]
            end = chunk[-1]["end"]

            text = " ".join([
                w["word"] for w in chunk
            ])

            srt_content += f"{index}\n"
            srt_content += (
                f"{format_time(start)} --> "
                f"{format_time(end)}\n"
            )
            srt_content += f"{text}\n\n"

            index += 1

        st.success("Subtitles Generated!")

        st.download_button(
            "Download SRT",
            srt_content,
            file_name="subtitles.srt"
        )

        os.remove(audio_path)
        os.remove(txt_path)