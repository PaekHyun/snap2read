# snap2read 📸→🔊

**책 페이지를 사진 찍으면 PC(GPU)가 읽어주는 가정용 AI 리더.**
Snap. Translate. Read aloud.

폰 웹브라우저 → 사진 업로드 → **PC GPU OCR(PP-OCRv6)** → 원문 표시·**수정 가능** → 자연 음성 **스트리밍 재생** → **로컬 LLM 한국어 번역** → MP3/WAV 다운로드

- 정확도: PaddleOCR **PP-OCRv6 medium** (50개 언어 단일 인식 모델, 그림 배경 오인식 최소화)
- 속도: GPU에서 페이지 1장 **0.3~1초** (서버 시작 시 자동 워밍업)
- **OCR 결과 수정**: 인식된 텍스트를 화면에서 직접 고치고 `🔁 다시 읽기` → 수정된 문장으로 음성 재생성
- **LLM 번역**: 로컬 Ollama(qwen3:14b/8b)로 오프라인·무제한 한국어 번역 — 사고과정(thinking) 자동 제거
- **Teacher 모드**: 폰으로 찍으면 태블릿(`/teacher`)에 자동 수신 — 태블릿에서 수정·재생·번역·저장
- 음성: edge-tts(자연 음성 MP3) → 실패 시 Windows SAPI(오프라인 WAV) 자동 대체

---

## 빠른 시작 (GPU 기본)

> 요구: NVIDIA GPU + Windows + **Python 3.9~3.12** (3.13/3.14는 Paddle 미지원 — conda 가상환경 권장)
> + **Ollama** (번역용 — https://ollama.com 설치, 1회)

```bash
# 0) 가상환경 준비 (예: conda)
conda create -n snap2read python=3.12 -y
conda activate snap2read

# 1) 기본 의존성
pip install -r requirements.txt

# 2) PaddlePaddle GPU 빌드  ← GPU 소유자는 이것 하나만!
#    GPU 세대에 따라 인덱스가 다릅니다 (아래 표)
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# 3) PaddleOCR
pip install paddleocr==3.7.0

# 4) GPU 인식 확인 — "PaddlePaddle works well on 1 GPU" 필수
python -c "import paddle; paddle.utils.run_check()"

# 5) 번역 모델 받기 — Ollama가 설치돼 있어야 합니다 (둘 다 받으면 시작 시 선택)
ollama pull qwen3:14b     # 16GB GPU 권장 (≈9GB)
ollama pull qwen3:8b      # 8GB GPU 권장 (≈5GB)

# 6) OCR 모델 준비 (아래 "모델 파일" 섹션) 후 서버 실행
python server.py
```

시작 배너에서 GPU 여부를 즉시 확인할 수 있습니다:

```
OCR: PaddleOCR 3.7.0 / paddle 3.2.1 (GPU=True)   ← 정상 (GPU)
OCR: PaddleOCR 3.7.0 / paddle 3.3.1 (GPU=False)  ← CPU 빌드가 설치된 상태
```

### GPU 빌드 선택표 (GPU 세대별)

| GPU | Compute Capability | 설치 명령 |
|-----|--------------------|-----------|
| **RTX 50 시리즈** (5090/5080/5070Ti… Blackwell) | sm_120 | `pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/` |
| **RTX 30/40 시리즈** (3070Ti/3080/4090… Ada·Ampere) | sm_86/89 | `pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/` |

- Paddle 3.x 휠은 **CUDA 런타임·cuDNN을 자체 포함** — 시스템 CUDA Toolkit/cuDNN 설치는 불필요 (직접 컴파일할 때만 필요).
- RTX 50 시리즈에 cu126/cu118 빌드를 쓰면 **커널이 아예 로드되지 않습니다** (`compute capability 120 not supported` 경고). 반드시 cu129.
- paddle.org.cn 접속이 안 되는 네트워크라면 임시로 CPU 빌드: `pip install paddlepaddle==3.3.1` (정확도 동일, 속도만 차이)

### CPU만 쓸 경우

```bash
pip install paddlepaddle==3.3.1 paddleocr==3.7.0
```

- 정확도는 GPU와 동일. 페이지당 2~8초.
- **주의**: CPU 경로는 oneDNN이 PP-OCRv6 일부 연산을 지원하지 못해 크래시할 수 있습니다. 서버가 이미 `enable_mkldnn=False`로 우회 처리해 둡니다.

---

## 폰에서 접속

1. 서버 실행 후 폰을 **같은 WiFi**에 연결
2. 폰 브라우저에서 `http://<PC IP>:9000` 접속 (PC IP: `ipconfig` → IPv4)
3. **방화벽 수동 등록 (권장)** — 다른 기기(폰·태블릿)에서 접속이 안 되면 대부분 방화벽이 원인입니다. PowerShell(관리자)에서 한 번 실행:

```powershell
New-NetFirewallRule -DisplayName "snap2read 9000" -Direction Inbound -Protocol TCP -LocalPort 9000 -Action Allow -Profile Private
```

- 기존 8000용 규칙이 있다면 제거: `Remove-NetFirewallRule -DisplayName "snap2read 8000"`
- 등록 확인: `Get-NetFirewallRule -DisplayName "snap2read 9000"`
3. **📷 사진 찍기/선택** → 폰에서는 **카메라 앱이 바로 열립니다** (HTTP여도 카메라 동작 — `getUserMedia` 미사용)
4. 촬영 → 자동 처리 → **음성 자동 재생**, 원문/번역 표시, `⬇ MP3/WAV 저장` 버튼
5. iOS: 공유 → **홈 화면에 추가**하면 앱처럼 사용

### HTTPS (자동)

- 서버에 `cert.pem`/`key.pem`이 있으면 자동으로 **HTTPS**로 뜹니다 (`https://<PC IP>:9000`) — 모바일 브라우저가 http→https 강제 전환 현상 대응.
- 자체서명 인증서라 처음 접속 시 "안전하지 않음" 경고가 뜹니다: **고급 → 안전하지 않음으로 이동(계속)** 1회 누르면 정상 사용 (기기별 1회).
- 인증서 재생성이 필요하면(예: PC IP가 바뀐 경우) `python generate_cert.py` 실행.
- `cert.pem`/`key.pem`이 없으면 이전처럼 HTTP로 서비스 (둘 다 지원).
- **`WARNING: Invalid HTTP request received.` 반복** — 서버가 HTTPS로 뜨고 있는데 폰이 **http**로 접속하려다 실패한 흔적입니다 (폰 브라우저가 http→https 강제 전환 중). 서버 오류가 아니라 무해하며, **폰에서 `https://`로 접속하면 사라집니다**. 반대로 서버가 HTTP인데 폰이 https로 시도하면 "사이트에 연결할 수 없음"이 뜹니다 — 이때는 인증서 생성(`python generate_cert.py`) 후 재시작하세요.

> IP가 자주 바뀌면 공유기에서 PC에 고정 IP 할당을 권장합니다.

---

## OCR 모델 파일 (필수 — 로컬 배치 방식)

서버는 프로젝트 폴더 안의 모델을 **직접 로드**합니다 (네트워크 다운로드 불필요, 완전 오프라인):

```
snap2read/
  PP-OCRv6_medium_det/     ← 글자 위치 검출   (inference.json/.pdiparams/.yml)
  PP-OCRv6_medium_rec/     ← 글자 인식(50개국) (inference.json/.pdiparams/.yml)
  PP-LCNet_x1_0_textline_ori/  ← 줄 방향 (선택)
  PP-LCNet_x1_0_doc_ori/       ← 문서 방향 (선택)
```

받는 방법 (둘 중 편한 것):

1. **노트북 자동 다운로더** (권장): `download_page.ipynb` 실행 — 4개 repo를 순회하며 이어받기(진행바·중단 재개)
2. 브라우저 수동: `https://huggingface.co/PaddlePaddle/<모델명>/resolve/main/<파일명>` — **safetensors 저장소가 아니라 순수 repo에서 받을 것** (`inference.pdiparams`가 있는 쪽)

> 방향 모델 2개가 없어도 서버는 동작합니다 (해당 기능 자동 비활성).
> 서버 시작 시 로컬 폴더를 PaddleX 캐시(`~\.paddlex\official_models`)로 자동 동기화(model-sync)하므로, 캐시를 신경 쓸 필요 없습니다.

### 파일 구성 참고

| 저장소 유형 | 파일 | 용도 |
|------------|------|------|
| `PP-OCRv6_medium_det` (순수) | inference.json, **inference.pdiparams**, inference.yml | ✅ PaddleOCR용 |
| `PP-OCRv6_medium_det_safetensors` | model.safetensors, config.json | ❌ transformers용 — PaddleOCR이 로드 불가 |

---

## LLM 번역 (3가지 엔진)

번역은 시작 시 선택합니다 — **1) Free API (Gemini→Groq 자동 폴백) 2) qwen3:14b 3) qwen3:8b**.

### Free API (무료, 로컬 GPU 불필요) ⭐추천

1. `config.example.py`를 `config.py`로 복사
2. [https://aistudio.google.com](https://aistudio.google.com) → "Get API key" → 키를 `GEMINI_API_KEY`에 붙여넣기 (무료, 카드 불필요 — 약 1,500요청/일)
3. (선택) [https://console.groq.com](https://console.groq.com) → API Key 생성 → `GROQ_API_KEY`에 붙여넣기 (무료, 카드 불필요 — 14,400요청/일)
4. `python server.py` → 선택 목록에서 `1) free` 선택

- **Gemini 한도 소진 시 Groq로 자동 전환** — 두 엔진 모두 무료
- 키는 `config.py`에만 저장 (`.gitignore` 등록 — GitHub에 커밋되지 않음)
- 다른 PC에서는 `config.py`를 직접 복사해야 합니다

### 로컬 Ollama (오프라인, 무제한)

```bash
# 1) Ollama 설치: https://ollama.com (1회)
# 2) 번역 모델 받기
ollama pull qwen3:14b     # 16GB GPU 권장 (≈9GB)
ollama pull qwen3:8b      # 8GB GPU 권장 (≈5GB)
# 3) (선택) 로컬 GGUF: ./Qwen3.5-9B-GGUF/에 .gguf 넣으면 자동 등록
```

- 고정: `python server.py --model qwen3:8b` (또는 `--model free`)
- config.py의 `LLM_ENGINE = "free"`로 고정도 가능
- 자동실행(비대화형) 시: free 옵션이 있으면 우선, 없으면 GPU VRAM 기준 자동 선택
- Qwen3 사고과정(thinking)은 서버에서 자동 제거 — 한국어 번역만 스트리밍 표시
- **Ctrl+C 종료 시** 모델을 VRAM에서 즉시 언로드 + 서버가 기동한 Ollama 정리

## Teacher 모드 (태블릿)

폰으로 찍으면 태블릿에 자동으로 나타나는 교사용 화면:

- 접속: `https://<PC IP>:9000/teacher` (또는 메인 화면 우측 상단 "교사용 화면 →")
- **폰 업로드 자동 수신** (1.5초 내) — 태블릿에서 읽어주기·수정·번역·MP3 저장 전부 가능
- 태블릿에서 직접 사진 촬영도 가능
- 최근 10개 작업 목록에서 다시 열기

## Windows 부팅 시 자동 실행 / 바로가기 실행

```powershell
# 등록 (이 폴더에서 실행 — miniforge envs\tf\python.exe를 자동 탐색, 없으면 PATH의 python)
powershell -ExecutionPolicy Bypass -File .\register_autostart.ps1

# 즉시 시작 (재부팅 없이 테스트)
Start-ScheduledTask -TaskName snap2read

# 제거
powershell -ExecutionPolicy Bypass -File .\register_autostart.ps1 -Remove
```

- 작업 이름 `snap2read`, **로그온 시** `python server.py` 실행, 중복 실행 방지, 실패 시 재시작
- **miniforge (tf) 환경 자동 탐색**: `%USERPROFILE%\miniforge3\envs\tf\python.exe` 등 표준 위치를 찾아 사용 — 환경 이름이 다르면 스크립트 상단 `$CondaEnv` 수정
- 등록은 멱등 — 다시 실행하면 갱신
- 콘솔 창 숨기려면 작업 스케줄러에서 프로그램을 `pythonw.exe`로 변경
- 💡 최초 1회는 수동 실행으로 방화벽 "사설 네트워크 허용"을 미리 승인

**바로가기 한 번 클릭으로 실행**: `start_server.bat` 더블클릭 — miniforge(tf) 환경으로 서버 시작.
바탕화면에 이 bat의 바로가기를 만들어 두면 클릭 한 번으로 실행됩니다. (창을 닫으면 서버 종료)

---

## 파이프라인 & 내부 동작

1. **OCR**: 로컬 PP-OCRv6 medium (GPU 자동) → 실패 시 기본 경로 → 실패 시 Tesseract 폴백
   - 서버 시작 시 **워밍업**(가짜 이미지 1회 처리)으로 GPU 첫 구동 지연을 사용자 요청과 분리
   - CPU에서는 oneDNN 비활성화(`enable_mkldnn=False`) — PP-OCRv6 호환성
2. **LLM 번역** (🌐 버튼 누를 때만 — Ollama):
   - 시작 시 설치된 모델 선택(콘솔) 또는 VRAM 감지 자동 선택(비대화형)
   - `/api/chat` + `think:false` → 500 시 think 키 제거 재시도
   - 스트림에서 `<think…</think` 사고 블록 실시간 제거 + `num_predict` 상한(무한 생성 차단)
   - 스트리밍으로 UI에 실시간 표시
3. **수정·다시 읽기**: 완료 후 텍스트 박스에서 직접 수정 → `🔁 다시 읽기`로 새 음성 생성 (캐시 우회 URL로 재생)
4. **TTS**: edge-tts(MP3) → 실패 시 Windows SAPI(WAV) 자동 대체

## API

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /api/process` | 사진 업로드 (`image`, `lang`) → `{id}` |
| `POST /api/reread/{id}` | 수정된 텍스트로 음성 재생성 (`text`) |
| `POST /api/translate/{id}` | 로컬 LLM 번역 — 스트리밍 응답 (`text`) |
| `GET /api/status/{id}` | 진행상황/원문/번역/단계별 소요시간 |
| `GET /api/audio/{id}` | 음성 스트리밍 + 다운로드 |
| `GET /api/image/{id}` | 업로드된 원본 사진 |
| `GET /api/latest` | 최신 작업 ID (teacher 자동 수신용) |
| `GET /api/jobs` | 최근 10개 작업 목록 (teacher 목록용) |
| `GET /teacher` | 교사용 화면 (태블릿) |

## 트러블슈팅 (실전에서 확인된 것들)

| 증상 | 원인 | 해결 |
|------|------|------|
| `partially initialized module 'paddle'` | **paddlepaddle-gpu와 paddlepaddle이 같은 site-packages에 겹침** | `pip uninstall -y paddlepaddle-gpu paddlepaddle` → 하나만 재설치 |
| `cudnn64_8.dll not configured (error 126)` | 구버전(cu118) 빌드가 시스템 cuDNN 8을 찾음 | 50시리즈면 cu129 빌드로 재설치 |
| `compute capability 120 not supported` | RTX 50 시리즈에 구형 CUDA 빌드 | cu129 빌드 사용 |
| `ConvertPirAttribute2RuntimeAttribute not support` (onednn) | CPU oneDNN × PP-OCRv6 비호환 | 서버가 자동 우회(`enable_mkldnn=False`) — paddle을 3.x로 유지 |
| `set_optimization_level` AttributeError | paddleocr 3.7 ↔ paddle 2.x 조합 | paddle을 3.x로 (`pip show paddlepaddle*`로 확인) |
| `TesseractNotFoundError` | Paddle 경로 실패 후 폴백에서 Tesseract도 없음 | 서버가 Paddle 원인을 함께 출력 — 그 원인부터 해결 |
| 첫 요청이 느림 (10~20초) | GPU 첫 구동(커널 로딩) | 서버가 시작 시 자동 워밍업 — 완료 후 첫 요청부터 빠름 |
| `TooManyRequests` | — (번역 기능은 제거됨) | — |
| 포트 9000 충돌 (`error 10048`) | 이전 서버 프로세스 생존 | 이전 창 `Ctrl+C` 또는 작업 관리자에서 python 종료 |
| `WARNING: Invalid HTTP request received.` 반복 | 서버는 HTTPS인데 폰이 http로 접속 시도 (https 강제 전환 실패 흔적) | 무해 — 폰에서 `https://<PC IP>:9000`으로 접속. 서버가 HTTP인데 이 경고가 나오면 반대로 폰이 https로 시도 중 → 인증서 생성(`python generate_cert.py`) 후 재시작 |

## 구성

| 파일 | 역할 |
|------|------|
| `server.py` | FastAPI 서버 — OCR → LLM 번역 → TTS → 스트리밍/다운로드 (모델 자동 동기화·선택·언로드 포함) |
| `static/index.html` | 폰용 웹 UI (촬영 → 진행상황 → 자동 재생 → 저장) |
| `static/teacher.html` | 교사용 화면 (폰 업로드 자동 수신 + 태블릿 촬영 + 편집/번역/저장) |
| `start_server.bat` | 원클릭 실행기 (miniforge tf 환경 자동 탐색 — 시작 시 번역 엔진 선택 프롬프트) |
| `start_server_free.bat` | 원클릭 실행기 — **Free API 고정** (Gemini→Groq 자동 폴백, config.py에 키 필요) |
| `start_server_14b.bat` | 원클릭 실행기 — **qwen3:14b 고정** (16GB GPU용, 모델 선택 없음) |
| `start_server_8b.bat` | 원클릭 실행기 — **qwen3:8b 고정** (8GB GPU용, 모델 선택 없음) |
| `generate_cert.py` | HTTPS 자체서명 인증서 생성 (SAN에 로컬 IP 자동 포함) |
| `download_page.ipynb` | HuggingFace 모델 4종 이어받기 노트북 (진행바·중단 재개) |
| `download_models.py` | 모델 다운로더 스크립트판 (`--legacy` v5 / `--all` 언어별) |
| `register_autostart.ps1` | Windows 로그온 자동실행 등록/제거 |
| `uploads/` | 업로드 사진·생성 음성 캐시 (커밋 제외) |

## 참고

- API 키 불필요. TTS는 온라인(실패 시 SAPI 대체), OCR은 오프라인 동작.
- 모델 폴더(총 ~150MB)는 `.gitignore` 제외 — repo에 커밋하지 않음.
- 생성 음성·사진은 `uploads/`에 남습니다 — 공용 PC라면 주기적으로 정리하세요.
