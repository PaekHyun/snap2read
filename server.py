"""snap2read — 사진 찍으면 읽어주는 가정용 AI 리더
폰 웹브라우저 → 사진 업로드 → OCR → 텍스트 확인/수정 → TTS 스트리밍 → MP3 다운로드

실행:  python server.py   →   http://<PC IP>:8000  (폰과 같은 WiFi)
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

# ===== LLM 번역 설정 — 바꿀 값은 이 블록에서만 수정하세요 =====
LLM_MODEL_DIR = BASE / "Qwen3.5-9B-GGUF"   # 로컬 GGUF 폴더 (상대경로 ./Qwen3.5-9B-GGUF)
LLM_MODEL_FILE = ""                        # 비우면 폴더의 .gguf 중 가장 큰 파일 자동 선택
LLM_OLLAMA_NAME = "snap2read-llm"          # Ollama에 등록될 모델 이름
LLM_OLLAMA_PORT = 11434                    # Ollama 포트
LLM_TRANSLATE_TARGET = {"kor": "영어"}      # 책 언어 → 번역 목표 언어 (없으면 한국어)
LLM_KEEP_ALIVE = "12h"                     # 모델 GPU 상주 시간
# ============================================================

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


# ---------------------------------------------------------------- LLM 번역 (Ollama)
LLM_READY = threading.Event()
LLM_REASON = "초기화 중…"
OLLAMA_PROC = None                      # 서버가 직접 기동한 Ollama 프로세스 (종료 시 정리 대상)
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _ollama_url() -> str:
    return f"http://127.0.0.1:{LLM_OLLAMA_PORT}"


def _pick_gguf():
    if LLM_MODEL_FILE:
        p = LLM_MODEL_DIR / LLM_MODEL_FILE
        return p if p.exists() else None
    if not LLM_MODEL_DIR.is_dir():
        return None
    cands = sorted(LLM_MODEL_DIR.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    return cands[0] if cands else None


def _ollama_alive(timeout: float = 3) -> bool:
    try:
        requests.get(f"{_ollama_url()}/api/version", timeout=timeout)
        return True
    except Exception:
        return False


def _ensure_ollama():
    """Ollama 기동 → 로컬 GGUF 등록(필요시) → 워밍업. 어떤 실패에도 서버는 계속 됩니다."""
    global LLM_REASON, OLLAMA_PROC
    try:
        if not _ollama_alive():
            print("[llm] Ollama 기동 중… (ollama serve)")
            OLLAMA_PROC = subprocess.Popen(
                ["ollama", "serve"], creationflags=_CREATE_NO_WINDOW,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(30):                    # 최대 15초
                if _ollama_alive():
                    break
                time.sleep(0.5)
            if not _ollama_alive():
                LLM_REASON = "Ollama를 찾거나 시작할 수 없습니다 — ollama.com 에서 설치하세요"
                print(f"[llm] 비활성: {LLM_REASON}")
                return
        tags = requests.get(f"{_ollama_url()}/api/tags", timeout=5).json().get("models", [])
        if not any(m.get("name", "").split(":")[0] == LLM_OLLAMA_NAME for m in tags):
            gguf = _pick_gguf()
            if not gguf:
                LLM_REASON = f"GGUF 파일이 없습니다: {LLM_MODEL_DIR}"
                print(f"[llm] 비활성: {LLM_REASON}")
                return
            print(f"[llm] 모델 등록 중: {gguf.name} → {LLM_OLLAMA_NAME} (수 분 소요 가능)")
            mf = LLM_MODEL_DIR / "Modelfile.snap2read"
            mf.write_text(f"FROM {gguf}\nPARAMETER num_ctx 4096\n", encoding="utf-8")
            p = subprocess.run(["ollama", "create", LLM_OLLAMA_NAME, "-f", str(mf)],
                               capture_output=True, timeout=1800)
            if p.returncode != 0:
                LLM_REASON = "ollama create 실패: " + p.stderr.decode(errors="replace")[-200:]
                print(f"[llm] 비활성: {LLM_REASON}")
                return
        print("[llm] 모델 워밍업 중… (첫 추론 — GPU 가중치 적재, 수십 초)")
        requests.post(f"{_ollama_url()}/api/generate",
                      json={"model": LLM_OLLAMA_NAME, "prompt": "hi", "stream": False,
                            "keep_alive": LLM_KEEP_ALIVE, "options": {"num_predict": 1}},
                      timeout=600)
        LLM_READY.set()
        LLM_REASON = ""
        print(f"[llm] 준비 완료 → 번역 버튼 사용 가능 ({LLM_OLLAMA_NAME})")
    except Exception as e:
        LLM_REASON = f"{type(e).__name__}: {e}"
        print(f"[llm] 비활성: {LLM_REASON}")


def _release_llm():
    """서버 종료 시 LLM 정리 — 모델을 VRAM에서 즉시 언로드(keep_alive:0),
    서버가 직접 기동한 Ollama 프로세스만 종료 (사용자가 미리 켜둔 것은 유지)."""
    global OLLAMA_PROC
    if not (_ollama_alive() or OLLAMA_PROC):
        return                            # LLM 미사용(--llm false 등) — 할 일 없음
    try:
        loaded = [m.get("name", "") for m in
                  requests.get(f"{_ollama_url()}/api/ps", timeout=5).json().get("models", [])]
        if any(n == LLM_OLLAMA_NAME or n.split(":")[0] == LLM_OLLAMA_NAME for n in loaded):
            requests.post(f"{_ollama_url()}/api/generate",
                          json={"model": LLM_OLLAMA_NAME, "prompt": "", "keep_alive": 0},
                          timeout=15)
            print("[llm] 모델 언로드 완료 — VRAM 해제")
        else:
            print("[llm] 로드된 모델 없음 — 언로드 불필요")
    except Exception:
        pass                              # 정리 실패가 종료를 막지 않게
    if OLLAMA_PROC is not None:
        try:
            OLLAMA_PROC.terminate()
            print("[llm] Ollama(서버가 기동한 프로세스) 종료")
        except Exception:
            pass
        OLLAMA_PROC = None


def _llm_translate_stream(text: str, lang: str):
    target = LLM_TRANSLATE_TARGET.get(lang, "한국어")
    # Qwen3 계열 추론(thinking) 모드 끄기 — 번역은 즉답이 맞음.
    # /no_think: 프롬프트 단 스위치 / think:false: Ollama 공식 옵션 (둘 다 적용)
    prompt = (f"/no_think 다음 텍스트를 {target}으로 번역하세요. "
              f"번역문만 출력하고 설명은 하지 마세요.\n\n{text}")
    req = {"model": LLM_OLLAMA_NAME, "prompt": prompt, "stream": True,
           "think": False,
           "keep_alive": LLM_KEEP_ALIVE,
           "options": {"temperature": 0.2, "num_ctx": 4096}}
    with requests.post(f"{_ollama_url()}/api/generate", json=req,
                       stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("thinking"):
                continue                    # 혹시 남은 사고 토큰은 화면에 안 보이게 폐기
            if data.get("response"):
                yield data["response"]
            if data.get("done"):
                break


# ---------------------------------------------------------------- 파이프라인
async def _run_job(jid: str, img_path: Path, lang: str):
    job = JOBS[jid]
    job["lang"] = lang
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

        # 2) TTS — edge-tts → SAPI 대체
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
    except Exception as exc:  # 절대 조용히 죽지 않게
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- API
@app.post("/api/process")
async def process(image: UploadFile = File(...),
                  lang: str = Form("eng")):
    jid = uuid.uuid4().hex[:12]
    suffix = Path(image.filename or "photo.jpg").suffix or ".jpg"
    img_path = UPLOADS / f"{jid}{suffix}"
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "빈 이미지입니다.")
    img_path.write_bytes(raw)

    job = {"id": jid, "status": "queued", "text": "", "audio": "",
           "audio_ext": "", "error": "", "ms": 0, "ocr_ms": 0, "tts_ms": 0,
           "image": img_path.name}
    with _LOCK:
        JOBS[jid] = job
    task = asyncio.create_task(_run_job(jid, img_path, lang))
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


@app.post("/api/translate/{jid}")
def api_translate(jid: str, text: str = Form(...)):
    """로컬 LLM(Ollama)으로 번역 — 스트리밍 응답 (폰 UI에 실시간 표시)."""
    with _LOCK:
        job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    text = text.strip()
    if not text:
        raise HTTPException(400, "빈 텍스트입니다.")
    if not LLM_READY.is_set():
        raise HTTPException(503, f"LLM 번역 사용 불가: {LLM_REASON or '준비 중입니다 — 1~2분 후 다시 시도'}")
    return StreamingResponse(_llm_translate_stream(text, job.get("lang", "eng")),
                             media_type="text/plain; charset=utf-8")


@app.post("/api/reread/{jid}")
async def reread(jid: str, text: str = Form(...)):
    """사용자가 수정한 텍스트로 음성을 다시 생성."""
    with _LOCK:
        job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    text = text.strip()
    if not text:
        raise HTTPException(400, "빈 텍스트입니다.")
    job["status"] = "tts"
    spoken = _clean_for_tts(text)
    stamp = int(time.time() * 1000)
    audio = UPLOADS / f"{jid}_r{stamp}.mp3"
    if await tts_edge(spoken, VOICES.get(job.get("lang", "eng"), VOICES["eng"]),
                      str(audio)):
        job["audio"], job["audio_ext"] = audio.name, "mp3"
    else:
        wav = UPLOADS / f"{jid}_r{stamp}.wav"
        ok = await asyncio.get_running_loop().run_in_executor(
            None, tts_sapi, spoken, str(wav))
        if not ok:
            job["status"] = "error"
            job["error"] = "음성 생성에 실패했습니다."
            raise HTTPException(500, job["error"])
        job["audio"], job["audio_ext"] = wav.name, "wav"
    job["text"] = text
    job["status"] = "done"
    return {"id": jid, "audio_ext": job["audio_ext"]}


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
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="snap2read 서버")
    ap.add_argument("--llm", default="true", choices=["true", "false"],
                    help="LLM 번역 사용 (기본 true — Ollama 자동 기동·로컬 GGUF 등록·워밍업)")
    args = ap.parse_args()

    threading.Thread(target=_warmup_paddle, daemon=True).start()
    if args.llm == "true":
        threading.Thread(target=_ensure_ollama, daemon=True).start()
    else:
        LLM_REASON = "실행 옵션으로 비활성화됨 (--llm false)"
        print("[llm] 비활성 (--llm false)")

    # HTTPS (자체서명 인증서) — 모바일 브라우저가 http를 https로 강제 전환하는 경우 대응.
    # cert.pem/key.pem이 있으면 https://...:8000으로 접속 가능. 없으면 http로 계속 서비스.
    cert, key = BASE / "cert.pem", BASE / "key.pem"
    ssl_args = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)} if cert.exists() and key.exists() else None
    if ssl_args:
        print("HTTPS 인증서 사용 — 폰에서는 https://<PC IP>:8000 접속 (첫 접속 경고 1회 수락)")
    print(f"snap2read 서버 시작 → {'https' if ssl_args else 'http'}://0.0.0.0:8000  (폰에서는 PC의 IP 사용)")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, **(ssl_args or {}))
    finally:
        _release_llm()   # Ctrl+C 종료 시 — 모델 VRAM 언로드 + 서버가 띄운 Ollama 정리

