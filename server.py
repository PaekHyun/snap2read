"""snap2read — 사진 찍으면 읽어주는 가정용 AI 리더
폰 웹브라우저 → 사진 업로드 → OCR → (선택) 번역 → TTS 스트리밍 → MP3 다운로드

실행:  python server.py   →   http://<PC IP>:8000  (폰과 같은 WiFi)
"""
import asyncio
import os
import re
import shutil
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

# 로컬 모델 폴더 (프로젝트 안에 직접 넣어둔 경우 우선 사용 — 자동 다운로드 생략)
# 필수: det/rec 2개 / 선택: 방향 판별 2개 (있으면 네트워크 의존성 완전 제거)
PADDLE_MODEL_DIRS = {
    "det": BASE / "PP-OCRv6_medium_det",
    "rec": BASE / "PP-OCRv6_medium_rec",
    "textline": BASE / "PP-LCNet_x1_0_textline_ori",
    "doc": BASE / "PP-LCNet_x1_0_doc_ori",
}

# PaddleOCR 3.x: 모델 소스 연결 체크 생략 (매 요청 수 초 절약, 캐시된 모델 사용)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _sync_local_models_to_cache() -> None:
    """프로젝트 안의 모델 폴더를 PaddleX 캐시(사용자 폴더/.paddlex/official_models)로
    동기화. PaddleOCR은 model_dir를 지정해도 내부 (name, None, None) 검증에서
    캐시를 참조하는 경우가 있어, 완전 오프라인을 위해 캐시를 미리 채운다.
    - 캐시에 없거나 용량이 작은(미완전 다운로드) 파일만 복사
    - 복사 실패 시 조용히 건너뜀 (기존 동작 유지)"""
    try:
        cache = Path(os.path.expanduser("~")) / ".paddlex" / "official_models"
        cache.mkdir(parents=True, exist_ok=True)
        for name, src in PADDLE_MODEL_DIRS.items():
            if not src.is_dir():
                continue
            dst = cache / src.name
            dst.mkdir(exist_ok=True)
            for f in src.iterdir():
                if f.suffix.lower() not in (".json", ".yml", ".yaml", ".pdiparams"):
                    continue
                target = dst / f.name
                need = (not target.exists()
                        or target.stat().st_size != f.stat().st_size)
                if need:
                    try:
                        shutil.copy2(f, target)
                        print(f"[model-sync] {f.name} → {target}")
                    except OSError as e:
                        print(f"[model-sync] 건너뜀 {f.name}: {e}")
    except Exception as e:
        print(f"[model-sync] 실패(무시하고 계속): {e}")


_sync_local_models_to_cache()

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
if OCR_ENGINE == "paddle":
    try:
        import paddle
        import paddleocr
        print(f"OCR: PaddleOCR {getattr(paddleocr, '__version__', '?')} / "
              f"paddle {getattr(paddle, '__version__', '?')} "
              f"(GPU={paddle.device.is_compiled_with_cuda()})")
    except Exception:
        pass
else:
    print("OCR: Tesseract (PaddleOCR 미설치)")

# Tesseract 언어코드 → PaddleOCR 언어코드
_PADDLE_LANG = {"eng": "en", "kor": "korean", "jpn": "japan",
                "chi_sim": "ch", "rus": "cyrillic", "vie": "latin",
                "spa": "latin", "fra": "latin", "deu": "latin"}


def _ocr_paddle(path: str, lang: str) -> str:
    from paddleocr import PaddleOCR
    engine_name = f"_paddle_{lang}"
    if not hasattr(app.state, engine_name):
        lang_code = _PADDLE_LANG.get(lang, "en")
        # 필수 2개(det/rec)만 있는지 검사 — 방향 모델(textline/doc)은 선택
        local_ok = (PADDLE_MODEL_DIRS["det"].is_dir()
                    and PADDLE_MODEL_DIRS["rec"].is_dir())
        if local_ok:
            # 로컬 PP-OCRv6_medium det/rec 사용 (50개 언어 단일 인식 모델) —
            # 방향 보정 모델(textline/doc_ori)은 비활성화해 네트워크 의존성 제거.
            # 로드 실패(PaddleOCR 구버전 등) 시 아래 기본 경로로 자동 폴백
            try:
                kw = dict(
                    text_detection_model_name="PP-OCRv6_medium_det",
                    text_detection_model_dir=str(PADDLE_MODEL_DIRS["det"]),
                    text_recognition_model_name="PP-OCRv6_medium_rec",
                    text_recognition_model_dir=str(PADDLE_MODEL_DIRS["rec"]),
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    # CPU oneDNN은 PP-OCRv6 일부 연산 미지원
                    # (ConvertPirAttribute2RuntimeAttribute 오류) → plain paddle 모드로 실행
                    enable_mkldnn=False,
                )
                # 방향 판별 모델도 로컬에 있으면 그 위치를 지정 (초기화 시 다운로드 방지)
                if PADDLE_MODEL_DIRS["textline"].is_dir():
                    kw["textline_orientation_model_dir"] = str(PADDLE_MODEL_DIRS["textline"])
                if PADDLE_MODEL_DIRS["doc"].is_dir():
                    kw["doc_orientation_classify_model_dir"] = str(PADDLE_MODEL_DIRS["doc"])
                eng = PaddleOCR(**kw)
                setattr(app.state, engine_name, eng)
                return _ocr_paddle_predict(eng, path)
            except Exception as e:
                print(f"[ocr] 로컬 v6 로드 실패 → 기본 경로로 대체: "
                      f"{type(e).__name__}: {e}")
        try:
            eng = PaddleOCR(use_textline_orientation=True, lang=lang_code,
                            enable_mkldnn=False)
        except TypeError:
            eng = PaddleOCR(use_angle_cls=True, show_log=False, lang=lang_code)
        setattr(app.state, engine_name, eng)
    engine = getattr(app.state, engine_name)
    return _ocr_paddle_predict(engine, path)


def _ocr_paddle_predict(engine, path: str) -> str:
    # 3.x predict API 우선, 실패 시 구버전(2.x) 호환 호출
    try:
        result = engine.predict(path)
        lines = []
        for res in result or []:
            try:
                lines.extend(res["rec_texts"])
            except Exception:
                try:
                    lines.extend(res.json["res"]["rec_texts"])
                except Exception:
                    pass
        return "\n".join(lines)
    except AttributeError:
        result = engine.ocr(path, cls=True)
        lines = []
        for page in result or []:
            for item in page or []:
                try:
                    lines.append(item[1][0])
                except Exception:
                    continue
        return "\n".join(lines)


def _tesseract_cmd():
    """Tesseract 실행파일 탐색: PATH → 표준 설치 위치 (winget/UB-Mannheim)"""
    cands = [shutil.which("tesseract"),
             r"C:\Program Files\Tesseract-OCR\tesseract.exe",
             r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
    for p in cands:
        if p and os.path.isfile(p):
            return p
    return None


def _ocr_tesseract(path: str, lang: str) -> str:
    import pytesseract
    from PIL import Image, ImageOps
    cmd = _tesseract_cmd()
    if not cmd:
        raise RuntimeError("Tesseract 실행파일을 찾을 수 없습니다 "
                           "(설치: winget install --id UB-Mannheim.TesseractOCR)")
    pytesseract.pytesseract.tesseract_cmd = cmd
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("L")      # 회전 보정 + 회색조
    w, h = img.size
    if max(w, h) < 1600:                                  # 저해상도 사진 업스케일
        k = 1600 / max(w, h)
        img = img.resize((int(w * k), int(h * k)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    return pytesseract.image_to_string(img, lang=lang, config="--oem 1 --psm 6")


def ocr_image(path: str, lang: str) -> str:
    paddle_err = None
    if OCR_ENGINE == "paddle":
        try:
            return _ocr_paddle(path, lang).strip()
        except Exception as e:
            paddle_err = f"{type(e).__name__}: {e}"
            print(f"[ocr] PaddleOCR 실패 → Tesseract 대체: {paddle_err}")
    try:
        return _ocr_tesseract(path, lang).strip()
    except Exception as e:
        msg = f"Tesseract 실패: {e}"
        if paddle_err:
            msg += f" | Paddle 원인: {paddle_err}"
        raise RuntimeError(msg)


# ---------------------------------------------------------------- 번역
_TRANS_TARGET = {"kor": "en", "chi_sim": "ko", "jpn": "ko", "rus": "ko",
                 "vie": "ko", "spa": "ko", "fra": "ko", "deu": "ko", "eng": "ko"}
_TRANS_SRC = {"eng": "english", "kor": "korean", "chi_sim": "chinese (simplified)",
              "jpn": "japanese", "rus": "russian", "vie": "vietnamese",
              "spa": "spanish", "fra": "french", "deu": "german"}


def _translate_google(text: str, src: str, tgt: str) -> tuple[str, str]:
    """구글(무료 엔드포인트) — 4,500자 단위 분할 + 긴 백오프 재시도."""
    from deep_translator import GoogleTranslator
    parts, chunk = [], ""
    for para in text.split("\n\n"):
        if len(chunk) + len(para) > 4500 and chunk:
            parts.append(chunk)
            chunk = para
        else:
            chunk = (chunk + "\n\n" + para).strip()
    if chunk:
        parts.append(chunk)
    tr = GoogleTranslator(source=src, target=tgt)
    out = []
    for p in parts:
        last = None
        for wait in (0, 2, 5, 10):        # 백오프: 즉시→2s→5s→10s (최대 ~17s)
            if wait:
                time.sleep(wait)
            try:
                out.append(tr.translate(p) or "")
                last = None
                break
            except Exception as e:
                last = e
        if last is not None:
            raise last
    return "\n\n".join(out), ""


_MEM_CODES = {"eng": "en-GB", "kor": "ko-KR", "jpn": "ja-JP",
              "chi_sim": "zh-CN", "rus": "ru-RU", "vie": "vi-VN",
              "spa": "es-ES", "fra": "fr-FR", "deu": "de-DE"}


def _translate_mymemory(text: str, lang: str) -> tuple[str, str]:
    """대체 엔진 MyMemory — 구글이 차단/제한 중일 때 사용 (450자 단위 분할)."""
    from deep_translator import MyMemoryTranslator
    tr = MyMemoryTranslator(source=_MEM_CODES.get(lang, "en-GB"),
                            target=_MEM_CODES.get("kor") if lang != "kor" else "en-GB")
    # 450자 이하 문장 단위 청크
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > 450 and cur:
            parts.append(cur)
            cur = line
        else:
            cur = (cur + " " + line).strip()
    if cur:
        parts.append(cur)
    out = [tr.translate(p) or "" for p in parts]
    return "\n".join(out), ""


def translate_text(text: str, lang: str) -> tuple[str, str]:
    """(번역문, 오류메시지) — 구글 실패 시 MyMemory로 자동 대체."""
    try:
        tgt = _TRANS_TARGET.get(lang, "ko")
        if tgt == "ko" and lang not in _TRANS_SRC:
            return "", ""
        src = _TRANS_SRC.get(lang, "english")
        if src == tgt:
            return text, ""
        try:
            return _translate_google(text, src, tgt)
        except Exception as e:
            g_err = f"{type(e).__name__}: {e}"
            print(f"[translate] 구글 실패 → MyMemory 대체 ({g_err})")
        try:
            return _translate_mymemory(text, lang)
        except Exception as e:
            return "", f"google({g_err}) → MyMemory({type(e).__name__}: {e})"
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


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


async def _bg_translate(job: dict, text: str, lang: str):
    """번역 백그라운드 실행 — 읽어주기(TTS)를 기다리게 하지 않기 위해 별도 태스크."""
    try:
        t0 = time.time()
        tr, err = await asyncio.get_running_loop().run_in_executor(
            None, translate_text, text, lang)
        job["translation"], job["translation_error"] = tr, err
        job["trans_ms"] = int((time.time() - t0) * 1000)
        if err:
            print(f"[translate] 실패: {err}")
    finally:
        job["translating"] = False


async def _run_job(jid: str, img_path: Path, lang: str, do_translate: bool):
    job = JOBS[jid]
    loop = asyncio.get_running_loop()
    t0 = time.time()
    try:
        # 1) OCR
        job["status"] = "ocr"
        t_ocr = time.time()
        text = await loop.run_in_executor(None, ocr_image, str(img_path), lang)
        job["ocr_ms"] = int((time.time() - t_ocr) * 1000)
        job["text"] = text
        if not text:
            job["status"] = "error"
            job["error"] = "글자를 인식하지 못했습니다. 페이지를 더 밝고 정면으로, 가까이 찍어보세요."
            return

        # 2) 번역을 백그라운드로 시작 → 읽어주기가 번역을 기다리지 않음
        tr_task = (asyncio.create_task(_bg_translate(job, text, lang))
                   if do_translate else None)

        # 3) TTS — edge-tts → SAPI 대체
        job["status"] = "tts"
        t_tts = time.time()
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
        job["tts_ms"] = int((time.time() - t_tts) * 1000)
        if tr_task:
            await tr_task          # 음성은 이미 재생 중 — 번역 완료까지만 폴링 유지
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
           "translation_error": "", "translating": False, "audio": "",
           "audio_ext": "", "error": "", "ms": 0, "ocr_ms": 0, "tts_ms": 0,
           "trans_ms": 0, "image": img_path.name}
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


def _warmup_paddle():
    """서버 시작 직후 OCR 엔진 예열 — GPU 최초 커널 로딩을 사용자 요청과 분리."""
    if OCR_ENGINE != "paddle":
        return
    try:
        from PIL import Image as _Img
        img = UPLOADS / "_warmup.png"
        _Img.new("RGB", (480, 320), "white").save(img)
        t = time.time()
        _ocr_paddle(str(img), "eng")
        img.unlink(missing_ok=True)
        print(f"[warmup] OCR 엔진 예열 완료 ({time.time() - t:.1f}초) — 첫 요청부터 빠릅니다")
    except Exception as e:
        print(f"[warmup] 건너뜀: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=_warmup_paddle, daemon=True).start()
    print("snap2read 서버 시작 → http://0.0.0.0:8000  (폰에서는 PC의 IP 사용)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
