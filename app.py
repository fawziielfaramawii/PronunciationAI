from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from difflib import SequenceMatcher
from gtts import gTTS
import tempfile
import os
import uuid

app = FastAPI(title="Pronunciation Checker API")

# -------- Load Whisper Model --------
model = WhisperModel("base", device="cpu", compute_type="int8")

# -------- Transcribe Audio --------
def transcribe_audio(audio_path):
    segments, _ = model.transcribe(audio_path, language="ar")
    return " ".join([segment.text for segment in segments]).strip()

# -------- Advanced Mistake Analysis --------
def analyze_mistakes_advanced(expected_word, recognized_text):
    seq = SequenceMatcher(None, expected_word, recognized_text)
    display_text = []
    mistakes = []

    for tag, i1, i2, j1, j2 in seq.get_opcodes():
        if tag == "equal":
            display_text.append(expected_word[i1:i2])
        elif tag == "replace":
            wrong = recognized_text[j1:j2]
            expected = expected_word[i1:i2]
            display_text.append(f"🔴{wrong}")
            mistakes.append(f"استبدلت '{expected}' بـ '{wrong}'")
        elif tag == "delete":
            deleted = expected_word[i1:i2]
            display_text.append("⚪")
            mistakes.append(f"حُذفت '{deleted}'")
        elif tag == "insert":
            inserted = recognized_text[j1:j2]
            display_text.append(f"🟢{inserted}")
            mistakes.append(f"تمت إضافة '{inserted}' غير موجودة")

    similarity = seq.ratio()
    return "".join(display_text), mistakes, similarity

# -------- Generate Correct Pronunciation Audio --------
def say_word(word):
    tts_text = f"{word}. {word}."
    tts = gTTS(text=tts_text, lang='ar', slow=True)
    filename = f"{uuid.uuid4()}.mp3"  # اسم فريد لكل ملف
    file_path = os.path.join(tempfile.gettempdir(), filename)
    tts.save(file_path)
    return file_path

# -------- API Endpoint to analyze --------
@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    expected_word: str = Form(...)
):
    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(await file.read())
        audio_path = f.name

    # Transcribe audio
    recognized_text = transcribe_audio(audio_path)

    # Analyze mistakes
    display_text, mistakes, similarity = analyze_mistakes_advanced(expected_word, recognized_text)
    success = similarity > 0.8 and not mistakes

    # Generate correct word audio if there were mistakes
    correct_audio_path = say_word(expected_word) if not success else None
    correct_audio_url = (
        str(request.base_url) + f"download_correct_audio/{os.path.basename(correct_audio_path)}"
        if correct_audio_path else None
    )

    # Return JSON response
    return {
        "recognized": recognized_text,
        "similarity": round(similarity * 100, 2),  # نسبة مئوية أوضح
        "display": display_text,
        "mistakes": mistakes,
        "success": success,
        "correct_audio_url": correct_audio_url
    }

# -------- API Endpoint to download correct audio --------
@app.get("/download_correct_audio/{filename}")
async def download_correct_audio(filename: str):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(path=file_path, filename="correct_pronunciation.mp3", media_type="audio/mpeg")
