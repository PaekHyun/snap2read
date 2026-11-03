"""PaddleOCR 모델 수동 다운로드 — 모델 자동 다운로드가 막힌/느린 네트워크용.

HuggingFace(huggingface.co/PaddlePaddle)에서 모델 파일을 받아
PaddleOCR 캐시 폴더(%%USERPROFILE%%.paddlex.official_models.<모델명>)에 넣습니다.
캐시가 채워지면 PaddleOCR은 자동 다운로드 없이 바로 실행됩니다.

실행:  python download_models.py          # PP-OCRv6 medium (권장, 50개 언어 단일 인식 모델)
       python download_models.py --legacy # PP-OCRv5 구성 (영어 위주)
       python download_models.py --all    # 구성 + 한국어·일본어·중국어·러시아어 등 언어별 모델
"""
import os
import sys
import urllib.error
import urllib.request

HF = "https://huggingface.co/PaddlePaddle/{model}/resolve/main/{file}"

# 파이프라인 공통: 방향 판별 모델 2종 + 인식/검출 모델
ORI_MODELS = [
    "PP-LCNet_x1_0_textline_ori",  # 텍스트 줄 방향
    "PP-LCNet_x1_0_doc_ori",       # 문서 방향
]
MODELS_V6 = ["PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"] + ORI_MODELS
MODELS_V5 = [
    "PP-OCRv5_server_det", "PP-OCRv5_server_rec",
    "PP-LCNet_x1_0_textline_ori", "PP-LCNet_x1_0_doc_ori",
]
LANG_MODELS = [
    "korean_PP-OCRv5_mobile_rec",      # 한국어
    "japan_PP-OCRv3_mobile_rec",       # 일본어 (v5 미배포)
    "chinese_cht_PP-OCRv3_mobile_rec", # 중국어 번체 (v5 미배포)
    "latin_PP-OCRv5_mobile_rec",       # 라틴 문자 확장
    "cyrillic_PP-OCRv5_mobile_rec",    # 키릴 문자
]
FILES = ["config.json", "inference.json", "inference.pdiparams", "inference.yml"]

CACHE = os.path.join(os.path.expanduser("~"), ".paddlex", "official_models")

def main():
    models = MODELS_V6
    if "--legacy" in sys.argv:
        models = MODELS_V5
    if "--all" in sys.argv:
        models = models + LANG_MODELS
    os.makedirs(CACHE, exist_ok=True)
    print("저장 위치:", CACHE)

    for model in models:
        d = os.path.join(CACHE, model)
        os.makedirs(d, exist_ok=True)
        print(f"\n[{model}]")
        for f in FILES:
            dst = os.path.join(d, f)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                print(f"  스킵(이미 있음): {f} ({os.path.getsize(dst):,} B)")
                continue
            url = HF.format(model=model, file=f)
            try:
                urllib.request.urlretrieve(url, dst)
                print(f"  OK ({os.path.getsize(dst):,} B): {f}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"  이 모델에는 없는 파일 — 건너뜀: {f}")
                    continue
                print(f"  실패: {e}")
            except Exception as e:
                print(f"  실패: {e}")

    print("\n완료. 서버 재시작 시 캐시가 사용됩니다.")

if __name__ == "__main__":
    main()
