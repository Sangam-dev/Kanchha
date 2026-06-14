"""
KANCHA — Text in, Streaming Audio out
Jarvis-style natural conversational AI
Uses gemini_client.py for LLM (multi-key rotation + model fallback)
Uses edge_tts for TTS
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


from reasoning.llm_client_mulapi import get_pool, _stream_raw, REQUEST_TIMEOUT, ALL_MODELS

try:
    import edge_tts
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sys.exit("Run: pip install edge-tts sounddevice soundfile")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

VOICE         = "en-GB-RyanNeural"
MIN_CHUNK_LEN = 10
MAX_CHUNK_LEN = 160
BOUNDARIES    = {'.', '!', '?', '—', '…'}
MAX_HISTORY   = 12

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


# ═══════════════════════════════════════════════════════════════════════════════
# SENTENCE SPLITTER + TEXT CLEANER
# ═══════════════════════════════════════════════════════════════════════════════

def _is_abbreviation(text: str, pos: int) -> bool:
    return bool(ABBREV.search(text[:pos + 1]))

def _extract_sentences(buffer: str) -> tuple[list[str], str]:
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
            sp = buffer.rfind(" ", 0, MAX_CHUNK_LEN)
            boundary_pos = sp if sp > MIN_CHUNK_LEN else MAX_CHUNK_LEN - 1

        if boundary_pos == -1:
            break

        sentence = buffer[:boundary_pos + 1].strip()
        buffer   = buffer[boundary_pos + 1:].lstrip()
        if sentence and len(sentence) > 2:
            sentences.append(sentence)

    return sentences, buffer

def _clean(text: str) -> str:
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'`[^`]+`', lambda m: m.group(0)[1:-1], text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'—', ' — ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# LLM — delegates entirely to gemini_client
# ═══════════════════════════════════════════════════════════════════════════════

async def stream_sentences(history: list[dict], user_input: str):
    """
    Adds user turn to history, streams via gemini_client (key rotation +
    model fallback), yields complete speakable sentences.
    """
    history.append({"role": "user", "parts": [{"text": user_input}]})
    while len(history) > MAX_HISTORY:
        history.pop(0)

    pool = get_pool()

    for model in ALL_MODELS:
        buffer        = ""
        full_response = ""
        try:
            async for chunk in _stream_raw(pool, model, history, REQUEST_TIMEOUT):
                buffer        += chunk
                full_response += chunk
                sentences, buffer = _extract_sentences(buffer)
                for s in sentences:
                    yield s

            if buffer.strip() and len(buffer.strip()) > 2:
                yield buffer.strip()

            if full_response:
                history.append({"role": "model", "parts": [{"text": full_response}]})
            return   # success

        except Exception as exc:
            sys.stderr.write(f"  [{model} failed: {exc!r} — trying next]\n")
            continue

    history.pop()
    yield "All models and keys are currently exhausted. Please try again in a moment."


# ═══════════════════════════════════════════════════════════════════════════════
# TTS — edge_tts + overlapped synth/play
# ═══════════════════════════════════════════════════════════════════════════════

async def _synthesize(sentence: str) -> tuple | None:
    sentence = _clean(sentence)
    if not sentence.strip():
        return None
    audio_bytes = b""
    tts = edge_tts.Communicate(text=sentence, voice=VOICE, rate="+12%", pitch="-4Hz")
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    if not audio_bytes:
        return None
    data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return data, samplerate

def _play(data, samplerate: int) -> None:
    sd.play(data, samplerate); sd.wait()


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

async def process(history: list[dict], user_input: str) -> None:
    """
    Overlapped synthesize-and-play:

        S1: [synth]──[play]
        S2:        [synth]──[play]
        S3:               [synth]──[play]

    While S_n plays, S_n+1 is already synthesizing — zero gap.
    """
    print("\nKANCHA thinking...\n")
    start        = time.perf_counter()
    loop         = asyncio.get_event_loop()
    first        = True
    audio_future = None

    async for sentence in stream_sentences(history, user_input):
        if first:
            print(f"  [first sentence in {time.perf_counter() - start:.2f}s]\n")
            first = False

        print(f"  🔊 {sentence}")
        this_future = asyncio.create_task(_synthesize(sentence))

        if audio_future is not None:
            audio = await audio_future
            if audio:
                await loop.run_in_executor(None, _play, *audio)

        audio_future = this_future

    if audio_future is not None:
        audio = await audio_future
        if audio:
            await loop.run_in_executor(None, _play, *audio)

    print(f"\n  [done in {time.perf_counter() - start:.2f}s]\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    history: list[dict] = []

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
            await process(history, user_input)
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())