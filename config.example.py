"""snap2read 설정 파일 예시 — 이 파일을 config.py로 복사해서 사용하세요.

⚠️ config.py는 .gitignore에 등록돼 있어 GitHub에 올라가지 않습니다 (API 키 보호).
   다른 PC에서는 이 파일을 config.py로 복사한 뒤 키를 넣으세요.

## API 키 발급 (둘 다 무료, 카드 불필요)
- Gemini: https://aistudio.google.com → "Get API key" → 키 복사
- Groq:   https://console.groq.com → "API Keys" → Create API Key
"""

GEMINI_API_KEY = ""     # 여기에 Gemini 키 붙여넣기
GROQ_API_KEY = ""       # 여기에 Groq 키 붙여넣기

GEMINI_MODEL = "gemini-3.8-flash"    # 2026-11 기준 (2.5-flash는 신규 사용자 제공 중단)
GROQ_MODEL = "openai/gpt-oss-120b"   # 무료 티어 (2026-11 기준: llama 계열은 유료 전환됨)

LLM_ENGINE = ""    # "free" | "ollama" | "" (시작 시 선택 프롬프트)
