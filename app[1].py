"""AI 콘텐츠 판별기 — Streamlit Community Cloud 배포용.

이미지: capcheck/ai-image-detection
텍스트: rasbt/ai-text-detector-gpt2-fixed (공식 fixed-position readout)
영상: 최대 20개 추출 프레임의 이미지 모델 분류 점수 참고용 평균

모든 AI 탐지 결과는 참고용이며 작성/생성 출처의 증거가 아닙니다.
"""
from __future__ import annotations

import gc
import json
import os
import tempfile
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# 대형 AI 라이브러리는 실제 분석 버튼을 누르기 전까지 로드하지 않습니다.
IMAGE_MODEL = "capcheck/ai-image-detection"
TEXT_MODEL = "rasbt/ai-text-detector-gpt2-fixed"
MAX_VIDEO_FRAMES = 20
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024

st.set_page_config(page_title="AI 콘텐츠 판별기", page_icon="🔎", layout="centered")


class DetectorCache:
    """무료 RAM을 아끼기 위해 한 시점에 모델 하나만 메모리에 유지."""

    def __init__(self):
        self.lock = RLock()
        self.kind = None
        self.model = None

    def load(self, kind):
        # 상위 호출부가 lock을 보유한 상태에서만 호출
        if self.kind == kind and self.model is not None:
            return self.model
        self.model = None
        self.kind = None
        gc.collect()
        import torch
        torch.set_num_threads(2)
        if kind == "image":
            from transformers import pipeline
            model_obj = pipeline("image-classification", model=IMAGE_MODEL, device=-1)
        elif kind == "text":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from huggingface_hub import hf_hub_download
            meta_path = hf_hub_download(repo_id=TEXT_MODEL, filename="detector-config.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL, dtype=torch.float32)
            model.eval()
            model_obj = (tokenizer, model, metadata)
        else:
            raise ValueError("알 수 없는 모델 종류")
        self.model = model_obj
        self.kind = kind
        return model_obj


@st.cache_resource
# 모든 방문자가 하나의 모델 저장소와 잠금장치를 공유합니다.
def get_cache() -> DetectorCache:
    return DetectorCache()


def judge(score):
    if score >= 70:
        return "AI 생성으로 분류하는 점수가 높습니다."
    if score >= 40:
        return "이 점수만으로는 판단하기 어렵습니다."
    return "실제 이미지/사람 작성으로 분류하는 점수가 높습니다."


def image_score(detector, image):
    results = detector(image.convert("RGB"), top_k=None)
    for result in results:
        label = str(result["label"]).strip().upper()
        if label in {"FAKE", "AI", "AI-GENERATED"}:
            return float(result["score"]) * 100
    raise RuntimeError("이미지 모델 결과에서 FAKE/AI 레이블을 찾지 못했습니다.")


def text_score(detector, text):
    import torch
    tokenizer, model, metadata = detector
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    max_text_length = int(metadata["max_text_length"])
    truncated = len(token_ids) > max_text_length
    token_ids = token_ids[:max_text_length]
    if metadata["readout_position"] == "fixed":
        pad_length = int(metadata["context_length"]) - len(token_ids) - 1
        if pad_length < 0:
            raise ValueError("분석할 텍스트가 입력 한도를 초과했습니다.")
        input_ids = token_ids + [tokenizer.pad_token_id] * pad_length + [tokenizer.eos_token_id]
        attention_mask = [1] * len(token_ids) + [0] * pad_length + [1]
    else:
        input_ids = token_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
    with torch.inference_mode():
        outputs = model(
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            attention_mask=torch.tensor([attention_mask], dtype=torch.long),
        )
        probabilities = (outputs.logits / float(metadata["temperature"])).float().softmax(dim=-1)[0]
        ai = float(probabilities[int(metadata["label_mapping"]["ai"])]) * 100
        human = float(probabilities[int(metadata["label_mapping"]["human"])]) * 100
    return ai, human, truncated


def video_scores(detector, uploaded_video, progress):
    # OpenCV는 실제 파일 경로가 필요합니다. 분석 후 즉시 삭제합니다.
    suffix = Path(uploaded_video.name).suffix.lower()
    with tempfile.NamedTemporaryFile(prefix="ai_detector_", suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_video.getvalue())
        video_path = tmp.name
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("영상 파일을 열 수 없습니다. MP4 파일로 다시 시도해 주세요.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total <= 0 or fps <= 0:
            raise ValueError("영상의 프레임 수 또는 FPS를 읽을 수 없습니다.")
        n = min(total, MAX_VIDEO_FRAMES)
        positions = [0] if n == 1 else [round(i * (total - 1) / (n - 1)) for i in range(n)]
        scores = []
        for i, pos in enumerate(positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            success, frame = cap.read()
            if success:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                scores.append(image_score(detector, img))
            progress.progress((i + 1) / n, text=f"프레임 분석 중: {i + 1}/{n}")
        if not scores:
            raise ValueError("영상에서 프레임을 추출하지 못했습니다.")
        return scores, total / fps
    finally:
        if cap is not None:
            cap.release()
        try:
            os.remove(video_path)
        except OSError:
            pass


st.title("🔎 AI 콘텐츠 판별기")
st.write("사진 · 텍스트 · 영상을 분석합니다. PC와 스마트폰 모두 사용할 수 있어요.")
st.info("⚠️ 점수는 모델의 분류 점수이며 실제 AI 생성 확률이 아닙니다. 사람이나 콘텐츠의 출처를 확정하는 증거로 사용하지 마세요.")

tab_image, tab_text, tab_video = st.tabs(["🖼️ 이미지", "📝 텍스트", "🎞️ 영상"])

with tab_image:
    image_upload = st.file_uploader("사진 선택", type=["jpg", "jpeg", "png", "webp", "bmp"], key="image")
    if image_upload is not None:
        st.image(image_upload, caption=image_upload.name, use_container_width=True)
    if st.button("이미지 분석", type="primary", key="analyze_image"):
        if image_upload is None:
            st.warning("먼저 이미지를 선택해 주세요.")
        elif image_upload.size > MAX_IMAGE_BYTES:
            st.warning("이미지는 15MB 이하만 분석할 수 있습니다.")
        else:
            try:
                with st.spinner("이미지 분석 중입니다. 첫 요청에는 모델 다운로드가 필요할 수 있습니다."):
                    image = Image.open(image_upload).convert("RGB")
                    cache = get_cache()
                    with cache.lock:
                        detector = cache.load("image")
                        score = image_score(detector, image)
                st.subheader("이미지 분석 결과")
                st.metric("AI 이미지 분류 점수", f"{score:.2f}%")
                st.write(f"실제 이미지 분류 점수: **{100-score:.2f}%**")
                st.write(f"판정 참고: **{judge(score)}**")
                st.caption("CIFAKE 기반으로 학습된 모델로, 새로운 AI 이미지나 편집·압축 이미지에는 오류가 있을 수 있습니다.")
            except Exception as exc:
                st.error(f"이미지 분석 실패: {exc}")

with tab_text:
    st.caption("영어 글 권장 · 한국어 분석 결과는 신뢰하기 어렵습니다.")
    input_text = st.text_area("분석할 글", height=240, placeholder="영어 글을 100자 이상 입력해 주세요.")
    if st.button("텍스트 분석", type="primary", key="analyze_text"):
        text = input_text.strip()
        if len(text) < 100:
            st.warning("100자 이상의 텍스트를 입력해 주세요.")
        elif len(text) > 30000:
            st.warning("30,000자 이하만 분석할 수 있습니다.")
        else:
            try:
                with st.spinner("텍스트 분석 중입니다. 첫 요청에는 모델 다운로드가 필요할 수 있습니다."):
                    cache = get_cache()
                    with cache.lock:
                        detector = cache.load("text")
                        ai, human, truncated = text_score(detector, text)
                st.subheader("텍스트 분석 결과")
                st.metric("AI 작성 분류 점수", f"{ai:.2f}%")
                st.write(f"사람 작성 분류 점수: **{human:.2f}%**")
                st.write(f"판정 참고: **{judge(ai)}**")
                if truncated:
                    st.warning("텍스트가 모델 입력 한도를 초과해 뒷부분은 분석에서 제외됐습니다.")
                st.caption("이 모델은 영어 데이터 기반입니다. 문체·번역·교정 등으로 결과가 달라집니다.")
            except Exception as exc:
                st.error(f"텍스트 분석 실패: {exc}")

with tab_video:
    st.caption(f"영상에서 최대 {MAX_VIDEO_FRAMES}개 프레임만 뽑아 이미지 모델로 분석합니다. 영상 전용 판별이 아닙니다.")
    video_upload = st.file_uploader("영상 선택 (최대 50MB)", type=["mp4", "avi", "mov", "mkv", "wmv"], key="video")
    if video_upload is not None:
        st.video(video_upload)
    if st.button("영상 분석", type="primary", key="analyze_video"):
        if video_upload is None:
            st.warning("먼저 영상을 선택해 주세요.")
        elif video_upload.size > MAX_VIDEO_BYTES:
            st.warning("영상은 50MB 이하만 분석할 수 있습니다.")
        else:
            try:
                with st.spinner("영상 프레임 분석 중입니다. CPU 환경에서는 느릴 수 있습니다."):
                    progress = st.progress(0, text="프레임 준비 중")
                    cache = get_cache()
                    with cache.lock:
                        detector = cache.load("image")
                        scores, duration = video_scores(detector, video_upload, progress)
                    progress.empty()
                avg = float(np.mean(scores))
                st.subheader("영상 분석 결과")
                st.write(f"영상 길이: **{duration:.1f}초** · 분석 프레임: **{len(scores)}개**")
                st.write(f"70점 이상 프레임: **{sum(score >= 70 for score in scores)}개**")
                st.metric("프레임 평균 AI 분류 점수", f"{avg:.2f}%")
                st.write(f"판정 참고: **{judge(avg)}**")
                st.line_chart({"AI 분류 점수 (%)": scores}, y_label="점수 (%)", x_label="추출 프레임 순서")
                st.caption("움직임·음성·프레임 간 일관성을 직접 판단하지 않습니다. 평균 점수는 영상의 AI 생성 확률이 아닙니다.")
            except Exception as exc:
                st.error(f"영상 분석 실패: {exc}")

st.divider()
st.caption("개인정보나 비공개 파일은 업로드하지 마세요. 공개 웹앱에서는 제3자 접근을 제한할 수 없습니다.")
