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
from collections import deque
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
except ImportError:
    sys.exit("Run: pip install google-genai")

try:
    import edge_tts
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sys.exit("Run: pip install edge-tts sounddevice soundfile")


# ── Config ─────────────────────────────────────────────────────────────
VOICE         = "en-GB-RyanNeural"
MODEL         = "gemini-flash-lite-latest"
MIN_CHUNK_LEN = 10
MAX_CHUNK_LEN = 160
BOUNDARIES    = {'.', '!', '?', '—', '…'}
MAX_HISTORY   = 12   # keep last N turns in memory

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

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Conversation memory: list of {"role": "user"|"model", "parts": [{"text": ...}]}
history: list[dict] = []


# ── Sentence splitter ───────────────────────────────────────────────────
def _is_abbreviation(text: str, pos: int) -> bool:
    """Check if the period at `pos` is part of an abbreviation."""
    preceding = text[:pos + 1]
    return bool(ABBREV.search(preceding))


def extract_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Extract complete, speakable sentences from buffer.
    Returns (list of sentences, leftover buffer).
    """
    sentences = []

    while True:
        boundary_pos = -1

        for i, char in enumerate(buffer):
            if char in BOUNDARIES and i >= MIN_CHUNK_LEN:
                # Don't split on abbreviation periods
                if char == '.' and _is_abbreviation(buffer, i):
                    continue
                # Don't split "..." as three separate sentences
                if char == '.' and i + 1 < len(buffer) and buffer[i + 1] == '.':
                    continue
                boundary_pos = i
                break

        # Force split on very long buffer
        if boundary_pos == -1 and len(buffer) >= MAX_CHUNK_LEN:
            space_pos = buffer.rfind(" ", 0, MAX_CHUNK_LEN)
            boundary_pos = space_pos if space_pos > MIN_CHUNK_LEN else MAX_CHUNK_LEN - 1

        if boundary_pos == -1:
            break

        sentence = buffer[:boundary_pos + 1].strip()
        buffer   = buffer[boundary_pos + 1:].lstrip()

        # Skip stray punctuation / empty fragments
        if sentence and len(sentence) > 2:
            sentences.append(sentence)

    return sentences, buffer


def clean_for_tts(text: str) -> str:
    """Strip any markdown or symbols that would sound weird when spoken."""
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # bold/italic
    text = re.sub(r'`[^`]+`', lambda m: m.group(0)[1:-1], text)  # inline code
    text = re.sub(r'#+\s*', '', text)                             # headers
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)         # links
    text = re.sub(r'—', ' — ', text)                              # em dash spacing
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── TTS ─────────────────────────────────────────────────────────────────
async def speak(sentence: str) -> None:
    """Synthesize and play one sentence via edge_tts + sounddevice."""
    sentence = clean_for_tts(sentence)
    if not sentence.strip():
        return

    print(f"  🔊 {sentence}")

    audio_bytes = b""
    tts = edge_tts.Communicate(
        text=sentence,
        voice=VOICE,
        rate="+12%",
        pitch="-4Hz",
        volume="+0%",
    )
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    if not audio_bytes:
        return

    audio_buffer = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(audio_buffer, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


# ── LLM streaming with history ──────────────────────────────────────────
async def stream_sentences(user_input: str):
    """
    Stream Gemini response with full conversation history.
    Yields complete sentences as they form.
    """
    # Add user turn to history
    history.append({"role": "user", "parts": [{"text": user_input}]})

    # Keep history bounded
    while len(history) > MAX_HISTORY:
        history.pop(0)

    buffer = ""
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

        buffer += text
        full_response += text

        sentences, buffer = extract_sentences(buffer)
        for s in sentences:
            yield s

    # Flush remainder
    if buffer.strip() and len(buffer.strip()) > 2:
        yield buffer.strip()

    # Add model response to history
    if full_response:
        history.append({"role": "model", "parts": [{"text": full_response}]})


# ── Pipeline ─────────────────────────────────────────────────────────────
async def process(user_input: str) -> None:
    print(f"\nKANCHA thinking...\n")
    start = time.perf_counter()
    first = True

    async for sentence in stream_sentences(user_input):
        if first:
            ttfs = time.perf_counter() - start
            print(f"  [first sentence in {ttfs:.2f}s]\n")
            first = False
        await speak(sentence)

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

        # Allow clearing memory mid-session
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