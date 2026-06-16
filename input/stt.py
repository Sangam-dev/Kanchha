import sounddevice as sd
import numpy as np
import time
from dotenv import load_dotenv
import os
import io
import wave
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SAMPLE_RATE = 16000

print("Groq STT ready!")


def record_until_enter():

    recording = []

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        recording.append(indata.copy())

    print("\nStart speaking...")
    print("Press ENTER when finished.\n")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
        callback=callback
    )

    stream.start()
    input()
    stream.stop()
    stream.close()

    if len(recording) == 0:
        return None

    return np.concatenate(recording, axis=0)



def to_wav_bytes(audio):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf




def transcribe(audio):

    wav_bytes = to_wav_bytes(audio)

    start = time.time()

    result = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=("audio.wav", wav_bytes, "audio/wav"),
        language="en"
    )

    end = time.time()

    print(f"\nTranscription Time: {end-start:.2f} sec")

    return result.text


while True:

    try:

        audio = record_until_enter()

        if audio is None:
            continue

        print("\nTranscribing...")

        result = transcribe(audio)

        print("\n===== TRANSCRIPT =====")
        print(result)
        print("======================")

    except KeyboardInterrupt:
        print("\nExiting...")
        break

    except Exception as e:
        print("\nERROR:")
        print(e)