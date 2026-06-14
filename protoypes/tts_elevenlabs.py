"""
KANCHA — Text in, Streaming Audio out
Jarvis-style natural conversational AI
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import sys
import time
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
except ImportError:
    sys.exit("Run: pip install google-genai")

try:
    import httpx
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError:
    sys.exit("Run: pip install httpx sounddevice soundfile numpy")


# ── Config ─────────────────────────────────────────────────────────────
MODEL         = "gemini-flash-lite-latest"
MIN_CHUNK_LEN = 10
MAX_CHUNK_LEN = 160
BOUNDARIES    = {'.', '!', '?', '—', '…'}
MAX_HISTORY   = 12

# ── ElevenLabs TTS config ───────────────────────────────────────────────

#   Daniel  → onwK4e9ZLuTAKqWW03F9  (British, warm, deep — closest to Jarvis)
#   Adam    → 29vD33N1CtxCmqQRPOHJ  (authoritative, calm)
#   Antoni  → ErXwobaYiN019PkySvjV  (natural, conversational)

ELEVENLABS_KEY      = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = "onwK4e9ZLuTAKqWW03F9"   # Daniel
ELEVENLABS_MODEL    = "eleven_turbo_v2_5"        # fastest + most expressive
ELEVENLABS_SETTINGS = {
    "stability":         0.35,   # 0=wild expressive, 1=flat robotic 
    "similarity_boost":  0.75,
    "style":             0.45,   # speaking style energy
    "use_speaker_boost": True,
}

# Sentence-ending abbreviations — don't split here
ABBREV = re.compile(
    r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|approx|dept|est|govt|inc|ltd)\.$',
    re.IGNORECASE,
)

SYSTEM = """You are KANCHA — a sharp, warm personal AI in the vein of Jarvis.

VOICE STYLE:
- Speak like a smart friend thinking out loud — not a textbook, not a report.
- Use contractions naturally: "you're", "it's", "I'd", "we've", "there's".
- Start responses with a brief verbal acknowledgment when it fits:
  e.g. "Right, so...", "Good question —", "Honestly,", "Yeah, that's...", "So here's the thing —"
- Use connective tissue between ideas: "and what's interesting is", "the tricky part is",
  "which basically means", "so what happens is", "and that's why".
- Occasionally trail a thought: "...which is kind of fascinating, actually."
- Vary sentence length — mix short punchy ones with longer flowing ones.
- It's fine to say "I think" or "I'd say" — that's human.

WHAT TO AVOID:
- Never say "Certainly!", "Of course!", "Absolutely!", "Great question!" — robotic filler.
- No bullet points, numbered lists, markdown, asterisks, or headers.
- No stiff transitions like "Furthermore", "Moreover", "In conclusion", "To summarize".
- No "I'm happy to help with that" — just help.
- Don't restate the question back. Just answer it.

FORMAT:
- Flowing prose only. Everything in connected paragraphs.
- Keep it appropriately brief — don't pad. If it's a quick answer, give a quick answer.
- For longer explanations, use natural spoken pacing with short bridging phrases.
- Everything must sound good read aloud.
"""

client  = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
history: list[dict] = []


# ── Sentence splitter ───────────────────────────────────────────────────
def _is_abbreviation(text: str, pos: int) -> bool:
    preceding = text[:pos + 1]
    return bool(ABBREV.search(preceding))


def extract_sentences(buffer: str) -> tuple[list[str], str]:
    sentences = []
    while True:
        boundary_pos = -1
        for i, char in enumerate(buffer):
            if char in BOUNDARIES and i >= MIN_CHUNK_LEN:
                if char == '.' and _is_abbreviation(buffer, i):
                    continue
                if char == '.' and i + 1 < len(buffer) and buffer[i + 1] == '.':
                    continue
                boundary_pos = i
                break

        if boundary_pos == -1 and len(buffer) >= MAX_CHUNK_LEN:
            space_pos = buffer.rfind(" ", 0, MAX_CHUNK_LEN)
            boundary_pos = space_pos if space_pos > MIN_CHUNK_LEN else MAX_CHUNK_LEN - 1

        if boundary_pos == -1:
            break

        sentence = buffer[:boundary_pos + 1].strip()
        buffer   = buffer[boundary_pos + 1:].lstrip()
        if sentence and len(sentence) > 2:
            sentences.append(sentence)

    return sentences, buffer


def clean_for_tts(text: str) -> str:
    """Strip markdown that would sound weird when spoken."""
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'`[^`]+`', lambda m: m.group(0)[1:-1], text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'—', ' — ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── ElevenLabs TTS ──────────────────────────────────────────────────────
async def synthesize(sentence: str) -> tuple | None:
    """
    Synthesize via ElevenLabs → returns (numpy_array, samplerate).
    Falls back to a warning and None if the key is missing.
    """
    sentence = clean_for_tts(sentence)
    if not sentence.strip():
        return None

    if not ELEVENLABS_KEY:
        print("  ⚠️  Set ELEVENLABS_API_KEY in your .env file")
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

    payload = {
        "text": sentence,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": ELEVENLABS_SETTINGS,
    }
    headers = {
        "xi-api-key": ELEVENLABS_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    async with httpx.AsyncClient(timeout=15) as http:
        resp = await http.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        print(f"  ⚠️  ElevenLabs error {resp.status_code}: {resp.text[:120]}")
        return None

    audio_buffer = io.BytesIO(resp.content)
    data, samplerate = sf.read(audio_buffer, dtype="float32")
    return data, samplerate


def play_blocking(data, samplerate: int) -> None:
    sd.play(data, samplerate)
    sd.wait()


# ── LLM streaming with history ──────────────────────────────────────────
async def stream_sentences(user_input: str):
    history.append({"role": "user", "parts": [{"text": user_input}]})
    while len(history) > MAX_HISTORY:
        history.pop(0)

    buffer        = ""
    full_response = ""

    response = client.models.generate_content_stream(
        model=MODEL,
        contents=history,
        config={"system_instruction": SYSTEM},
    )

    for chunk in response:
        text = getattr(chunk, "text", "")
        if not text:
            continue
        buffer        += text
        full_response += text
        sentences, buffer = extract_sentences(buffer)
        for s in sentences:
            yield s

    if buffer.strip() and len(buffer.strip()) > 2:
        yield buffer.strip()

    if full_response:
        history.append({"role": "model", "parts": [{"text": full_response}]})


# ── Overlapped synthesize-and-play pipeline ─────────────────────────────
async def process(user_input: str) -> None:
    """
    While sentence N is playing, sentence N+1 is being synthesized.
    Zero perceptible gap between sentences.

        S1: [synth]──[play]
        S2:        [synth]──[play]
        S3:               [synth]──[play]
    """
    print(f"\nKANCHA thinking...\n")
    start        = time.perf_counter()
    loop         = asyncio.get_event_loop()
    first        = True
    audio_future = None

    async for sentence in stream_sentences(user_input):
        if first:
            print(f"  [first sentence in {time.perf_counter() - start:.2f}s]\n")
            first = False

        print(f"  🔊 {sentence}")
        this_future = asyncio.create_task(synthesize(sentence))

        # Play the previously synthesized sentence while this one synths
        if audio_future is not None:
            audio = await audio_future
            if audio:
                await loop.run_in_executor(None, play_blocking, *audio)

        audio_future = this_future

    # Play the last sentence
    if audio_future is not None:
        audio = await audio_future
        if audio:
            await loop.run_in_executor(None, play_blocking, *audio)

    print(f"\n  [done in {time.perf_counter() - start:.2f}s]\n")


# ── Input loop ───────────────────────────────────────────────────────────
async def main() -> None:
    print("=" * 50)
    print("  KANCHA — Text in, Audio out")
    print("  Type your message. Press Ctrl+C to exit.")
    print("=" * 50)

    loop = asyncio.get_event_loop()

    while True:
        try:
            user_input = await loop.run_in_executor(
                None, lambda: input("\nYou: ").strip()
            )
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("KANCHA: Later.")
            break

        if user_input.lower() in {"clear", "reset", "forget"}:
            history.clear()
            print("  [memory cleared]\n")
            continue

        try:
            await process(user_input)
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())