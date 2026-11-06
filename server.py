"""snap2read — 사진 찍으면 읽어주는 가정용 AI 리더
폰 웹브라우저 → 사진 업로드 → OCR → 텍스트 확인/수정 → TTS 스트리밍 → MP3 다운로드

실행:  python server.py   →   http(s)://<PC IP>:9000  (폰과 같은 WiFi, 기본 포트 9000)
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Windows cp949 콘솔에서 유니코드 출력(em-dash 등) 크래시 방지 — UTF-8 강제
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

PORT = 9000                                 # 서버 포트 (방화벽 허용 필요)
# ===== LLM 번역 설정 — 바꿀 값은 이 블록에서만 수정하세요 =====
# API 키·무료 API 모델은 config.py에서 관리 (GitHub에 커밋되지 않음)
try:
    from config import GEMINI_API_KEY, GROQ_API_KEY, GEMINI_MODEL, GROQ_MODEL, LLM_ENGINE
except ImportError:
    GEMINI_API_KEY = GROQ_API_KEY = ""
    GEMINI_MODEL = "gemini-2.5-flash"
    GROQ_MODEL = "llama-3.1-8b-instant"
    LLM_ENGINE = ""
    print("[config] config.py 없음 — API 키는 config.py에 넣으세요 (README 참고)")
LLM_MODEL_DIR = BASE / "Qwen3.5-9B-GGUF"   # 로컬 GGUF 폴더 (상대경로 ./Qwen3.5-9B-GGUF)
LLM_MODEL_FILE = ""                        # 비우면 폴더의 .gguf 중 가장 큰 파일 자동 선택
LLM_MODEL_CHOICES = ["free", "qwen3:14b", "qwen3:8b"]  # 1)Free API(Gemini→Groq) 2)14b 3)8b
LLM_OLLAMA_NAME = ""                       # 비우면 시작 시 위 목록에서 선택 (고정하려면 이름 기입 또는 --model 옵션)
                                           # 예: "qwen3:14b" / "qwen3:8b" / "snap2read-llm"(로컬 GGUF)
LLM_OLLAMA_PORT = 11434                    # Ollama 포트
LLM_TRANSLATE_TARGET = {"eng": "한국어", "kor": "영어"}   # 책 언어 → 번역 목표 언어 (없으면 한국어)
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


def _detect_vram_gb():
    """nvidia-smi로 GPU 총 VRAM(GB) 확인 — 실패 시 0."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, timeout=10)
        return max(int(l.strip()) for l in p.stdout.decode("utf-8", "replace").splitlines() if l.strip()) / 1024
    except Exception:
        return 0.0


def _choose_model(have, tags):
    """LLM_MODEL_CHOICES 중 선택 — 콘솔이면 번호로 고르고, 아니면 첫 번째.
    "free"는 API 키가 있으면 선택 가능 (Gemini→Groq 자동 폴백).
    반환: 모델 이름 / 실패 시 "" (사유는 LLM_REASON에 설정)"""
    global LLM_REASON

    def avail(name):
        return any(n == name or n.split(":")[0] == name.split(":")[0] for n in have)

    # "free" 옵션: API 키가 하나라도 있으면 유효
    free_ok = bool(GEMINI_API_KEY or GROQ_API_KEY)

    if LLM_OLLAMA_NAME:                       # --model 옵션 등으로 고정된 경우
        if LLM_OLLAMA_NAME == "free":
            if free_ok:
                print("[llm] 고정 엔진: Free API (Gemini→Groq)")
                return "free"
            LLM_REASON = "Free API 선택됐지만 API 키가 없습니다 — config.py에 GEMINI_API_KEY 또는 GROQ_API_KEY를 넣으세요"
            print(f"[llm] 비활성: {LLM_REASON}")
            return ""
        if avail(LLM_OLLAMA_NAME):
            print(f"[llm] 고정 모델 사용: {LLM_OLLAMA_NAME}")
            return LLM_OLLAMA_NAME
        LLM_REASON = (f"모델 '{LLM_OLLAMA_NAME}' 없음 — "
                      f"'ollama pull {LLM_OLLAMA_NAME}' 실행 후 재시작")
        print(f"[llm] 비활성: {LLM_REASON}")
        return ""
    # 선택 목록 구성: free는 키가 있을 때만 포함
    cands = [c for c in LLM_MODEL_CHOICES if (c == "free" and free_ok) or (c != "free" and avail(c))]
    if not cands:
        LLM_REASON = ("설치된 번역 모델이 없습니다 — 다음 중 하나를 받으세요: "
                      + " / ".join(f"ollama pull {c}" for c in LLM_MODEL_CHOICES))
        print(f"[llm] 비활성: {LLM_REASON}")
        return ""
    if len(cands) == 1:
        print(f"[llm] 모델 자동 선택: {cands[0]}")
        return cands[0]
    if sys.stdin.isatty():
        info = {m.get("name", ""): m for m in tags}
        print("[llm] 번역 엔진 선택 — Free API는 무료(일 1,500건), 16GB GPU면 큰 모델, 8GB GPU면 작은 모델 권장")
        for i, c in enumerate(cands, 1):
            if c == "free":
                srcs = " + ".join(n for n, k in (("Gemini", GEMINI_API_KEY), ("Groq", GROQ_API_KEY)) if k)
                print(f"  {i}) {c:<14} Free API ({srcs}) — 로컬 GPU 불필요")
                continue
            mi = info.get(c) or next((v for k, v in info.items()
                                      if k.split(":")[0] == c.split(":")[0]), {})
            d = mi.get("details", {}) or {}
            print(f"  {i}) {c:<14} {str(d.get('parameter_size', '?')):>7}  "
                  f"{str(d.get('quantization', '?')):<10} {mi.get('size', 0) / 1e9:.1f}GB")
        try:
            pick = input(f"선택 [1-{len(cands)}, Enter=1]: ").strip()
            idx = int(pick) - 1 if pick else 0
        except (ValueError, EOFError, KeyboardInterrupt):
            idx = 0
        name = cands[max(0, min(idx, len(cands) - 1))]
        print(f"[llm] 선택됨: {name}")
        return name
    print("[llm] 비대화형 실행 — 자동 선택…")
    # free 옵션이 목록에 있으면 우선 (API 키가 있다는 뜻 — 로컬 GPU 부담 없음)
    if "free" in cands:
        print("[llm] 자동 선택: free (Free API — Gemini→Groq)")
        return "free"
    vram = _detect_vram_gb()
    info_map = {m.get("name", ""): m for m in tags}

    def size_of(c):
        mi = info_map.get(c) or next((v for k, v in info_map.items()
                                      if k.split(":")[0] == c.split(":")[0]), {})
        return mi.get("size", 0) / 1e9

    # 규칙: 모델(파일크기×1.6 여유 계수)이 VRAM의 ~70% 이하면 GPU 전용 가능
    # OCR+시스템이 VRAM 일부를 쓰므로 안전선 70%로 잡음
    fit = [c for c in cands if size_of(c) * 1.6 <= vram * 0.7] if vram else []
    name = fit[0] if fit else cands[-1]      # 안 맞으면 가장 작은 모델로 폴백
    why = (f"GPU {vram:.0f}GB 기준" if vram else "GPU 감지 실패 — 가장 작은 모델")
    print(f"[llm] 자동 선택: {name} ({why})")
    return name


def _ensure_ollama():
    """LLM 엔진 준비 — free(API)면 Ollama 불필요, ollama면 기동·선택·워밍업.
    어떤 실패에도 서버는 계속 됩니다."""
    global LLM_REASON, OLLAMA_PROC, LLM_OLLAMA_NAME
    try:
        if LLM_ENGINE == "free":
            LLM_OLLAMA_NAME = "free"     # config에서 고정
        if LLM_OLLAMA_NAME == "free":
            if not (GEMINI_API_KEY or GROQ_API_KEY):
                LLM_REASON = "Free API 선택됐지만 API 키가 없습니다 — config.py에 GEMINI_API_KEY 또는 GROQ_API_KEY를 넣으세요"
                print(f"[llm] 비활성: {LLM_REASON}")
                return
            LLM_READY.set()
            LLM_REASON = ""
            keys = " + ".join(n for n, k in (("Gemini", GEMINI_API_KEY), ("Groq", GROQ_API_KEY)) if k)
            print(f"[llm] 준비 완료 → Free API 번역 사용 가능 ({keys})")
            return
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
        have = [m.get("name", "") for m in tags]
        LLM_OLLAMA_NAME = _choose_model(have, tags)
        if not LLM_OLLAMA_NAME:
            return                              # 사유는 _choose_model이 설정
        if LLM_OLLAMA_NAME == "free":
            if not (GEMINI_API_KEY or GROQ_API_KEY):
                LLM_REASON = "Free API 선택됐지만 API 키가 없습니다 — config.py에 GEMINI_API_KEY 또는 GROQ_API_KEY를 넣으세요"
                print(f"[llm] 비활성: {LLM_REASON}")
                return
            LLM_READY.set()
            LLM_REASON = ""
            keys = " + ".join(n for n, k in (("Gemini", GEMINI_API_KEY), ("Groq", GROQ_API_KEY)) if k)
            print(f"[llm] 준비 완료 → Free API 번역 사용 가능 ({keys})")
            return
        if ":" in LLM_OLLAMA_NAME:
            # 공식 모델(qwen3:14b/8b 등) — 로컬 GGUF 등록 건너뛰고 설치 여부만 확인
            if not any(n == LLM_OLLAMA_NAME or n.split(":")[0] == LLM_OLLAMA_NAME.split(":")[0]
                       for n in have):
                LLM_REASON = (f"모델 '{LLM_OLLAMA_NAME}' 없음 — "
                              f"터미널에서 'ollama pull {LLM_OLLAMA_NAME}' 실행 후 재시작")
                print(f"[llm] 비활성: {LLM_REASON}")
                return
            print(f"[llm] 공식 모델 사용: {LLM_OLLAMA_NAME} (think 토글 지원)")
        elif not any(n.split(":")[0] == LLM_OLLAMA_NAME for n in have):
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


def _strip_think_stream(chunks):
    """응답 스트림에서 <think ...>...</think 사고 블록을 제거 (청크 경계 잘림 대응).
    Qwen 계열은 think 옵션이 무시되면 사고를 본문에 <think 태그로 섞어 출력한다."""
    buf = ""
    in_think = False
    need_strip_head = False
    emitted = False
    for chunk in chunks:
        buf += chunk
        out = []
        while buf:
            if in_think:
                i = buf.find("</think")
                if i == -1:
                    buf = buf[-8:]              # 닫는 태그 경계 대비 꼬리만 보존, 나머지 폐기
                    break
                buf = buf[i + 8:]
                in_think = False
                need_strip_head = True          # 사고 직후 잔여 '>'·빈 줄 제거
                continue
            i = buf.find("<think")
            if i == -1:
                if len(buf) > 8:
                    out.append(buf[:-8])
                    buf = buf[-8:]
                break
            out.append(buf[:i])
            rest = buf[i + 6:]
            j = rest.find(">")
            if j == -1:
                in_think = True
                buf = ""
                break
            buf = rest[j + 1:]
            in_think = True
        text = "".join(out)
        if need_strip_head:
            text = text.lstrip(">\n\r\t ")
            if text:
                need_strip_head = False
        if text:
            emitted = True
            yield text
    if buf and not in_think:
        text = buf.lstrip(">\n\r\t ") if need_strip_head else buf
        if text:
            emitted = True
            yield text
    if not emitted:
        yield "(번역 실패: 모델이 사고 모드로만 응답했습니다 — 서버 로그를 확인하세요)"


def _gemini_stream(text: str, lang: str):
    """Gemini API 무료 티어 스트리밍 (SSE). 실패 시 Groq로 폴백은 호출부에서 처리."""
    target = LLM_TRANSLATE_TARGET.get(lang, "한국어")
    prompt = (f"다음 텍스트를 {target}로 번역하세요. "
              f"규칙: 아동 도서 번역체로 자연스럽게. 인물을 지칭할 때는 그, 그녀 대신 이름을 "
              f"사용해도 자연스러우면 이름으로. 번역문만 출력하고 설명은 출력하지 마세요.\n\n"
              f"{text}")
    # 모델 폴백 체인 — 설정 모델이 404(폐기/오타)면 후보를 순서대로 시도
    gemini_candidates = [GEMINI_MODEL, "gemini-3.1-flash-lite", "gemini-flash-lite-latest"]
    seen = set()
    gemini_models = [m for m in gemini_candidates if not (m in seen or seen.add(m))]
    last_gerr = None
    for gmodel in gemini_models:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{gmodel}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}")
        try:
            yield from _gemini_stream_once(url, prompt)
            return
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"[llm] Gemini 모델 '{gmodel}' 없음(404) → 다음 후보 시도")
                last_gerr = e
                continue
            raise
    raise last_gerr or RuntimeError("Gemini 사용 가능 모델 없음")


def _gemini_stream_once(url: str, prompt: str):
    """Gemini 단일 모델 스트리밍 — 무료 티어 부하(503) 재시도 포함."""
    # 무료 티어는 일시적 부하(503)가 흔함 — 최대 3회 백오프 재시도 후 폴백
    r = None
    for attempt in range(3):
        try:
            r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]},
                              stream=True, timeout=600)
            if r.status_code in (429, 500, 503):
                r.close()
                wait = 2 * (attempt + 1)
                print(f"[llm] Gemini {r.status_code} — {wait}초 후 재시도 ({attempt + 1}/3)")
                time.sleep(wait)
                r = None
                continue
            r.raise_for_status()
            break
        except requests.exceptions.RequestException:
            if r is not None:
                r.close()
                r = None
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    if r is None:
        raise RuntimeError("Gemini 재시도 실패")
    with r:
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8", "replace")
            if not line.startswith("data:"):
                continue
            try:
                data = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            for cand in data.get("candidates", []):
                for part in (cand.get("content", {}) or {}).get("parts", []):
                    t = part.get("text")
                    if t:
                        yield t


def _groq_stream(text: str, lang: str):
    """Groq API 무료 티어 스트리밍 (OpenAI 호환 SSE)."""
    target = LLM_TRANSLATE_TARGET.get(lang, "한국어")
    prompt = (f"다음 텍스트를 {target}로 번역하세요. "
              f"규칙: 아동 도서 번역체로 자연스럽게. 인물을 지칭할 때는 그, 그녀 대신 이름을 "
              f"사용해도 자연스러우면 이름으로. 번역문만 출력하고 설명은 출력하지 마세요.\n\n"
              f"{text}")
    with requests.post("https://api.groq.com/openai/v1/chat/completions",
                       headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                       json={"model": GROQ_MODEL,
                             "messages": [{"role": "user", "content": prompt}],
                             "stream": True, "temperature": 0.2},
                       stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8", "replace")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (data.get("choices", [{}])[0].get("delta", {}) or {})
            t = delta.get("content")
            if t:
                yield t


def _llm_translate_stream(text: str, lang: str):
    """엔진 분기: free(Gemini→Groq 폴백) 또는 Ollama. 모두 스트리밍."""
    if LLM_OLLAMA_NAME == "free":
        engines = []
        if GEMINI_API_KEY:
            engines.append(("Gemini", _gemini_stream))
        if GROQ_API_KEY:
            engines.append(("Groq", _groq_stream))
        last_err = None
        for name, gen in engines:
            try:
                it = iter(gen(text, lang))
                first = next(it, None)          # 첫 청크를 미리 받아본다 — 여기서 실패하면 폴백 가능
                if first is None:
                    raise RuntimeError("응답이 비었습니다")
                yield first
                yield from it
                return
            except Exception as e:
                last_err = f"{name}: {type(e).__name__}: {e}"
                print(f"[llm] {name} 실패 → 다음 엔진: {last_err}")
        raise RuntimeError(f"Free API 호출 실패 — {last_err or 'API 키가 없습니다 (config.py 확인)'}")

    target = LLM_TRANSLATE_TARGET.get(lang, "한국어")
    # /no_think: Qwen 계열 프롬프트 스위치 / 마지막 "{target} 번역:" 큐로 출력 언어 고정.
    # 그래도 사고가 본문에 섞여 나오면 _strip_think_stream가 걷어내고,
    # num_predict 상한으로 무한 생성(사고 무한 출력)을 차단한다.
    prompt = (f"/no_think 다음 텍스트를 {target}로 번역하세요. "
              f"규칙: 아동 도서 번역체로 자연스럽게. 인물을 지칭할 때는 그, 그녀 대신 이름을 "
              f"사용해도 자연스러우면 이름으로. 번역문만 출력하고 설명·사고과정은 출력하지 마세요.\n\n"
              f"{text}\n\n{target} 번역:")
    # /api/chat + think:false — Ollama 공식 경로. 다만 모델이 think를 지원하지
    # 않으면 Ollama가 500을 돌려주므로, 실패하면 think 키를 빼고 한 번 재시도.
    req = {"model": LLM_OLLAMA_NAME,
           "messages": [{"role": "user", "content": prompt}],
           "stream": True,
           "think": False,
           "keep_alive": LLM_KEEP_ALIVE,
           "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 1200}}
    def _post(payload):
        return requests.post(f"{_ollama_url()}/api/chat", json=payload,
                             stream=True, timeout=600)
    try:
        r = _post(req)
        if r.status_code == 500:
            r.close()
            print("[llm] /api/chat 500 — think 옵션 미지원 모델로 보임, think 키 제거 재시도")
            req.pop("think", None)
            r = _post(req)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama 호출 실패: {type(e).__name__}: {e}")
    with r:
        def chunks():
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message") or {}
                if data.get("thinking") or msg.get("thinking"):
                    continue                    # thinking 필드로 분리된 사고 토큰 폐기
                content = msg.get("content")
                if content:
                    yield content
                if data.get("done"):
                    break

        yield from _strip_think_stream(chunks())


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
@app.get("/api/latest")
async def latest():
    """가장 최근 작업 ID — teacher 화면이 폰 업로드를 자동 수신하는 데 사용."""
    with _LOCK:
        if not JOBS:
            return {"id": ""}
        jid = max(JOBS, key=lambda k: JOBS[k].get("created", 0))
    return {"id": jid}


@app.get("/api/jobs")
async def jobs():
    """최근 10개 작업 요약 (최신순) — teacher 화면의 목록용."""
    with _LOCK:
        items = sorted(JOBS.values(), key=lambda j: j.get("created", 0), reverse=True)[:10]
    return [{"id": j["id"], "status": j["status"], "error": j.get("error", ""),
             "image": j.get("image", "")} for j in items]


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
           "image": img_path.name, "created": time.time()}
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


@app.get("/teacher")
async def teacher():
    return FileResponse(BASE / "static" / "teacher.html")


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
    ap.add_argument("--model", default="",
                    help="번역 엔진 고정 (free / qwen3:14b / qwen3:8b) — 시작 시 선택 프롬프트 생략")
    args = ap.parse_args()
    if args.model:
        LLM_OLLAMA_NAME = args.model
    if LLM_ENGINE and not args.model:
        LLM_OLLAMA_NAME = LLM_ENGINE   # config.py의 LLM_ENGINE 고정값 적용

    threading.Thread(target=_warmup_paddle, daemon=True).start()
    if args.llm == "true":
        threading.Thread(target=_ensure_ollama, daemon=True).start()
    else:
        LLM_REASON = "실행 옵션으로 비활성화됨 (--llm false)"
        print("[llm] 비활성 (--llm false)")

    # HTTPS (자체서명 인증서) — 모바일 브라우저가 http를 https로 강제 전환하는 경우 대응.
    # cert.pem/key.pem이 있으면 https://...:$PORT(9000)로 접속 가능. 없으면 http로 계속 서비스.
    cert, key = BASE / "cert.pem", BASE / "key.pem"
    ssl_args = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)} if cert.exists() and key.exists() else None
    if ssl_args:
        print(f"HTTPS 인증서 사용 — 폰에서는 https://<PC IP>:{PORT} 접속 (첫 접속 경고 1회 수락)")
    print(f"snap2read 서버 시작 → {'https' if ssl_args else 'http'}://0.0.0.0:{PORT}  (폰에서는 PC의 IP 사용)")
    # Ctrl+C가 즉시 안 먹히던 원인: 번역 스트리밍 등 열린 연결을 uvicorn이 기다림.
    # graceful timeout 5초 — 구버전 uvicorn이면 옵션 생략.
    import inspect
    run_kwargs = {}
    if "timeout_graceful_shutdown" in inspect.signature(uvicorn.run).parameters:
        run_kwargs["timeout_graceful_shutdown"] = 5
    try:
        uvicorn.run(app, host="0.0.0.0", port=PORT, **run_kwargs, **(ssl_args or {}))
    except KeyboardInterrupt:
        pass
    finally:
        _release_llm()   # Ctrl+C 종료 시 — 모델 VRAM 언로드 + 서버가 띄운 Ollama 정리

