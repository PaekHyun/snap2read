# snap2read 📸→🔊

**책 페이지를 사진 찍으면 PC가 읽어주는 가정용 AI 리더.**
Snap. Translate. Read aloud.

폰 웹브라우저 → 사진 업로드 → PC(OCR) → 원문 표시 → (선택) 번역 표시 → 자연 음성 **스트리밍 재생** → MP3/WAV 다운로드

## 빠른 시작

```bash
pip install -r requirements.txt
python server.py
```

- 방화벽 프롬프트가 뜨면 **사설 네트워크 허용**
- 폰 브라우저에서 `http://<PC의 IP>:8000` 접속 (예: `http://192.168.0.12:8000`)
- PC IP 확인: `ipconfig` → IPv4 주소
- iOS Safari: 공유 → **홈 화면에 추가**하면 앱처럼 사용 가능

## OCR 엔진 (성능 순)

| 엔진 | 성능 | 설치 | 요건 |
|------|------|------|------|
| **PaddleOCR (GPU)** ⭐추천 | 최고 — 정확도·속도 최상급, 한글 지원 | 아래 "GPU 설치" 참조 | CUDA GPU (3070 Ti 완벽 지원) |
| PaddleOCR (CPU) | 우수 — CPU만으로도 Tesseract보다 정확 | `pip install paddleocr paddlepaddle` | CPU만 |
| Tesseract | 보통 — 간단하지만 복잡 레이아웃에 약함 | `winget install --id UB-Mannheim.TesseractOCR` 후 `pip install pytesseract` | Windows 설치 프로그램 + 언어팩 |

**서버는 자동 선택합니다** — PaddleOCR이 import되면 GPU/CPU 판단해 사용하고, 아니면 Tesseract. 아무것도 없으면 시작 안 됨.

### GPU(3070 Ti)용 PaddleOCR 설치 — 권장 구성

PaddlePaddle CUDA 빌드는 **CUDA 12.6 + cuDNN 9.x** 조합이 필요합니다. 먼저 설치된 것을 확인하세요:

```bash
nvidia-smi        # 드라이버·CUDA 지원 확인
nvcc --version    # CUDA 툴킷 버전
```

```bash
# 1) Python 3.9~3.12 권장 (3.13/3.14는 아직 미지원)
# 2) 의존성 설치
pip install -r requirements.txt

# 3) PaddlePaddle GPU (CUDA 12.6 빌드)
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 4) PaddleOCR
python -m pip install paddleocr

# 5) 동작 확인 (GPU 인식되는지)
python -c "import paddle; paddle.utils.run_check(); print(paddle.device.cuda.device_count())"
```

- `run_check()`가 GPU를 잡으면 서버는 자동으로 PaddleOCR을 씁니다.
- CUDA 툴킷이 없거나 버전이 안 맞으면 CPU 빌드(`pip install paddlepaddle`)로 시작해도 OCR 정확도는 동일합니다(속도만 차이).
- 컴퓨터 비전 의존성(opencv 등)은 PaddleOCR 설치 시 함께 깔립니다.

### Tesseract만 쓸 경우 (GPU 없는 보조 PC용)

```bash
winget install --id UB-Mannheim.TesseractOCR
pip install pytesseract
```

- 설치 후 **언어팩**: 한국어는 `kor`, 간체 중국어는 `chi_sim` (설치 프로그램에서 언어 선택 또는 [tessdata](https://github.com/tesseract-ocr/tessdata)에서 다운로드해 설치 폴더의 `tessdata`에 복사)
- 서버는 Tesseract가 한 개 언어만 있어도 동작합니다 (`eng` 하나면 영어는 OK).

## Windows 부팅 시 자동 실행

저장소의 `register_autostart.ps1`이 로그온 시 서버를 자동 실행하는 예약 작업을 등록·제거합니다 (작업 스케줄러 사용).

```powershell
# 등록 (이 폴더에서 실행 — server.py 위치를 자동으로 잡음)
powershell -ExecutionPolicy Bypass -File .\register_autostart.ps1

# 즉시 시작 (재부팅 없이 테스트)
Start-ScheduledTask -TaskName snap2read

# 제거
powershell -ExecutionPolicy Bypass -File .\register_autostart.ps1 -Remove
```

동작 방식:

- 작업 이름 `snap2read`, **로그온 시** 트리거로 `python server.py` 실행 (작업 폴더 = 스크립트가 있는 폴더)
- 노트북 배터리에서도 실행, 실행 시간 제한 없음, **중복 실행 방지**, 실패 시 1분 간격 최대 3회 재시작
- 등록은 멱등(idempotent) — 다시 실행하면 기존 작업을 갱신
- 콘솔 창을 숨기려면 작업 스케줄러에서 프로그램을 `pythonw.exe`로 변경

> 💡 최초 1회는 수동으로 서버를 실행해 방화벽 "사설 네트워크 허용"을 미리 승인해 두세요. 자동 실행 프로세스는 방화벽 프롬프트를 놓치기 쉽습니다.

## 구성

| 파일 | 역할 |
|------|------|
| `server.py` | FastAPI 서버 — OCR → 번역 → TTS → 스트리밍/다운로드 |
| `static/index.html` | 폰용 웹 UI (카메라 촬영 → 진행상황 → 자동 재생 → 저장) |
| `register_autostart.ps1` | Windows 로그온 자동실행 등록/제거 스크립트 |
| `uploads/` | 업로드 사진·생성 음성 캐시 |

## 파이프라인

1. **OCR**: PaddleOCR(GPU 우선) → 없으면 Tesseract
2. **번역**: deep-translator(Google, 무료·키 불필요) — "번역도 함께 보기" 체크 시
3. **TTS**: edge-tts(온라인, 자연 음성 MP3) → 실패 시 **Windows SAPI(오프라인 WAV)** 자동 대체

## API

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /api/process` | 사진 업로드 (`image`, `lang`, `translate`) → `{id}` |
| `GET /api/status/{id}` | 진행상황/원문/번역/소요시간 |
| `GET /api/audio/{id}` | 음성 스트리밍 + 다운로드 |
| `GET /api/image/{id}` | 업로드된 원본 사진 |

## 참고

- 서드파티 API 키 불필요. 오프라인 폴백(OCR·SAPI) 동작.
- 생성 음성·사진은 `uploads/`에 남습니다 — 공용 PC라면 주기적으로 정리하세요.
