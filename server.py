"""snap2read — 사진 찍으면 읽어주는 가정용 AI 리더
폰 웹브라우저 → 사진 업로드 → OCR → (선택) 번역 → TTS 스트리밍 → MP3 다운로드

실행:  python server.py   →   http://<PC IP>:8000  (폰과 같은 WiFi)
"""
import asyncio
import os
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

app = FastAPI(title="snap2read")

JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_TASKS: set = set()

# ---------------------------------------------------------------- OCR
def _pick_ocr_engine() -> str:
    """GPU PaddleOCR가 설치돼 있으면 우선 사용, 아니면 Tesseract."""
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        return "paddle"
    except Exception:
        return "tesseract"

OCR_ENGINE = _pick_ocr_engine()

# Tesseract 언어코드 → PaddleOCR 언어코드
_PADDLE_LANG = {"eng": "en", "kor": "korean", "jpn": "japan",
                "chi_sim": "ch", "rus": "cyrillic", "vie": "latin",
                "spa": "latin", "fra": "latin", "deu": "latin"}


def _ocr_paddle(path: str, lang: str) -> str:
    from paddleocr import PaddleOCR
    engine_name = f"_paddle_{lang}"
    if not hasattr(app.state, engine_name):
        setattr(app.state, engine_name,
                PaddleOCR(use_angle_cls=True, lang=_PADDLE_LANG.get(lang, "en"),
                          show_log=False))
    engine = getattr(app.state, engine_name)
    result = engine.ocr(path, cls=True)
    lines = []
    for page in result or []:
        for item in page or []:
            try:
                lines.append(item[1][0])
            except Exception:
                continue
    return "\n".join(lines)


def _ocr_tesseract(path: str, lang: str) -> str:
    import pytesseract
    from PIL import Image, ImageOps
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("L")      # 회전 보정 + 회색조
    w, h = img.size
    if max(w, h) < 1600:                                  # 저해상도 사진 업스케일
        k = 1600 / max(w, h)
        img = img.resize((int(w * k), int(h * k)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    return pytesseract.image_to_string(img, lang=lang, config="--oem 1 --psm 6")


def ocr_image(path: str, lang: str) -> str:
    if OCR_ENGINE == "paddle":
        try:
            return _ocr_paddle(path, lang).strip()
        except Exception:
            pass  # GPU 경로 실패 시 Tesseract로 대체
    return _ocr_tesseract(path, lang).strip()


# ---------------------------------------------------------------- 번역
_TRANS_TARGET = {"kor": "en", "chi_sim": "ko", "jpn": "ko", "rus": "ko",
                 "vie": "ko", "spa": "ko", "fra": "ko", "deu": "ko", "eng": "ko"}
_TRANS_SRC = {"eng": "english", "kor": "korean", "chi_sim": "chinese (simplified)",
              "jpn": "japanese", "rus": "russian", "vie": "vietnamese",
              "spa": "spanish", "fra": "french", "deu": "german"}


def translate_text(text: str, lang: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        target = _TRANS_TARGET.get(lang, "ko")
        if target == "ko" and lang not in _TRANS_SRC:
            return ""
        src = _TRANS_SRC.get(lang, "english")
        if src == _TRANS_TARGET.get(lang, ""):
            return text
        parts, chunk = [], ""
        for para in text.split("\n\n"):
            if len(chunk) + len(para) > 4000 and chunk:
                parts.append(chunk)
                chunk = para
            else:
                chunk = (chunk + "\n\n" + para).strip()
        if chunk:
            parts.append(chunk)
        out = [GoogleTranslator(source=src, target=target).translate(p) or "" for p in parts]
        return "\n\n".join(out)
    except Exception:
        return ""  # 번역 실패는 치명적이지 않음 — 읽어주기는 계속


# ---------------------------------------------------------------- TTS
# 언어별 기본 음성 (edge-tts)
VOICES = {"eng": "en-US-AriaNeural", "kor": "ko-KR-SunHiNeural",
          "jpn": "ja-JP-NanamiNeural", "chi_sim": "zh-CN-XiaoxiaoNeural",
          "rus": "ru-RU-SvetlanaNeural", "vie": "vi-VN-HoaiMyNeural",
          "spa": "es-ES-ElviraNeural", "fra": "fr-FR-DeniseNeural",
          "deu": "de-DE-KatjaNeural"}
_SAPI_HINT = {"eng": ["David", "Zira"], "kor": ["SunHi", "Heami", "Hyeon"]}


def _clean_for_tts(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t[:5000] if t else ""


async def tts_edge(text: str, voice: str, path: str, timeout: float = 45) -> bool:
    """온라인 TTS (edge-tts, MP3 출력). 실패 시 False."""
    try:
        import edge_tts

        async def _run():
            await edge_tts.Communicate(text, voice).save(path)

        await asyncio.wait_for(_run(), timeout=timeout)
        return os.path.getsize(path) > 1000
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return False


def tts_sapi(text: str, out_wav: str) -> bool:
    """오프라인 TTS (Windows SAPI, WAV 출력). 네트워크 없이 동작."""
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        hints = _SAPI_HINT.get("eng", [])  # 텍스트 언어 힌트는 호출부에서 확장 가능
        try:
            for v in voice.GetVoices():
                desc = v.GetDescription()
                if any(h in desc for h in hints):
                    voice.Voice = v
                    break
        except Exception:
            pass
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(out_wav, 3, False)          # 3 = SSFMCreateForWrite
        prev = voice.AudioOutputStream
        voice.AudioOutputStream = stream
        voice.Speak(text)
        stream.Close()
        voice.AudioOutputStream = prev
        return os.path.getsize(out_wav) > 1000
    except Exception:
        return False


# ---------------------------------------------------------------- 파이프라인
def _stage(msg: str):
    def wrap():
        pass
    return wrap


async def _run_job(jid: str, img_path: Path, lang: str, do_translate: bool):
    job = JOBS[jid]
    loop = asyncio.get_running_loop()
    t0 = time.time()
    try:
        # 1) OCR
        job["status"] = "ocr"
        text = await loop.run_in_executor(None, ocr_image, str(img_path), lang)
        job["text"] = text
        if not text:
            job["status"] = "error"
            job["error"] = "글자를 인식하지 못했습니다. 페이지를 더 밝고 정면으로, 가까이 찍어보세요."
            return

        # 2) 번역 (선택)
        if do_translate:
            job["status"] = "translate"
            job["translation"] = await loop.run_in_executor(
                None, translate_text, text, lang)

        # 3) TTS — edge-tts → SAPI 대체
        job["status"] = "tts"
        spoken = _clean_for_tts(text)
        audio = UPLOADS / f"{jid}.mp3"
        if await tts_edge(spoken, VOICES.get(lang, VOICES["eng"]), str(audio)):
            job["audio"], job["audio_ext"] = audio.name, "mp3"
        else:
            wav = UPLOADS / f"{jid}.wav"
            ok = await loop.run_in_executor(None, tts_sapi, spoken, str(wav))
            if not ok:
                job["status"] = "error"
                job["error"] = "음성 생성에 실패했습니다. (인터넷 연결 시 edge-tts 사용, 없으면 Windows SAPI 사용)"
                return
            job["audio"], job["audio_ext"] = wav.name, "wav"

        job["status"] = "done"
        job["ms"] = int((time.time() - t0) * 1000)
    except Exception as exc:  # 절대 조용히 죽지 않게
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- API
@app.post("/api/process")
async def process(image: UploadFile = File(...),
                  lang: str = Form("eng"),
                  translate: str = Form("false")):
    jid = uuid.uuid4().hex[:12]
    suffix = Path(image.filename or "photo.jpg").suffix or ".jpg"
    img_path = UPLOADS / f"{jid}{suffix}"
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "빈 이미지입니다.")
    img_path.write_bytes(raw)

    job = {"id": jid, "status": "queued", "text": "", "translation": "",
           "audio": "", "audio_ext": "", "error": "", "ms": 0,
           "image": img_path.name}
    with _LOCK:
        JOBS[jid] = job
    task = asyncio.create_task(_run_job(jid, img_path, lang, translate == "true"))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return {"id": jid}


@app.get("/api/status/{jid}")
async def status(jid: str):
    with _LOCK:
        job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return job


@app.get("/api/audio/{jid}")
async def audio(jid: str):
    with _LOCK:
        job = JOBS.get(jid)
    if not job or not job.get("audio"):
        raise HTTPException(404, "아직 음성이 없습니다.")
    mime = "audio/mpeg" if job["audio_ext"] == "mp3" else "audio/wav"
    return FileResponse(UPLOADS / job["audio"], media_type=mime,
                        filename=f"snap2read_{jid}.{job['audio_ext']}")


@app.get("/api/image/{jid}")
async def image(jid: str):
    with _LOCK:
        job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return FileResponse(UPLOADS / job["image"], media_type="image/jpeg")


@app.get("/")
async def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    print("snap2read 서버 시작 → http://0.0.0.0:8000  (폰에서는 PC의 IP 사용)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
