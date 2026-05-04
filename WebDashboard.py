import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from mylib import myTextAnalyzer as ta
from konlpy.tag import Okt
import re

# 1. 페이지 설정
st.set_page_config(page_title="단어 빈도수 시각화 대시보드", layout="wide")

# 🔥 캐싱
@st.cache_data
def load_data(file_path, column_name):
    df = pd.read_csv(file_path)
    if column_name not in df.columns:
        return None
    return df[column_name].dropna().astype(str).tolist()

@st.cache_data
def preprocess_text(corpus):
    cleaned = []
    for text in corpus:
        text = re.sub(r'[^가-힣\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 1:
            cleaned.append(text)
    return cleaned

# 🔥 핵심: chunk 기반 Okt 처리 (OutOfMemory 방지)
@st.cache_data
def tokenize_chunked(corpus, chunk_size=300):
    okt = Okt()
    tokens = []

    for i in range(0, len(corpus), chunk_size):
        chunk = corpus[i:i + chunk_size]

        # 너무 큰 문자열 방지
        joined_text = " ".join(chunk)

        for word, tag in okt.pos(joined_text):
            if tag in ['Noun', 'Verb', 'Adjective']:
                tokens.append(word)

    return tokens

# 2. 사이드바
with st.sidebar:
    st.title("📂 파일 선택")
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])

    column_name = st.text_input("데이터 컬럼명", value="review")

    # 미리보기
    if st.button("데이터 미리보기"):
        if uploaded_file is not None:
            df_preview = pd.read_csv(uploaded_file)
            st.dataframe(df_preview.head(5))
            uploaded_file.seek(0)
        else:
            st.warning("파일 업로드 먼저")

    st.markdown("---")
    st.title("⚙️ 옵션")

    show_bar_graph = st.checkbox("막대 그래프", value=True)
    num_words_bar = st.slider("단어 수", 5, 50, 20)

    show_wordcloud = st.checkbox("워드클라우드", value=False)
    num_words_wc = st.slider("워드 수", 10, 200, 50)

    start_btn = st.button("분석 시작", type="primary")

# 3. 메인
st.title("단어 빈도수 시각화")

if start_btn:
    if uploaded_file is None:
        st.error("CSV 파일 업로드 필요")
    else:
        try:
            if not os.path.exists('data'):
                os.makedirs('data')

            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("분석 중..."):

                # 1. 데이터 로드
                corpus = load_data(file_path, column_name)
                if corpus is None:
                    st.error(f"'{column_name}' 컬럼 없음")
                    st.stop()

                # 2. 전처리
                corpus = preprocess_text(corpus)

                if len(corpus) == 0:
                    st.error("유효한 데이터 없음")
                    st.stop()

                # 3. 토큰화 (🔥 핵심 개선)
                tokens = tokenize_chunked(corpus)

                if len(tokens) == 0:
                    st.error("토큰 없음")
                    st.stop()

                # 4. 빈도 분석
                counter = ta.analyze_word_freq(tokens)

                st.success(f"완료: {len(corpus)}개 문장 / {len(tokens)}개 단어")

                # 폰트
                font_path = "c:/Windows/Fonts/malgun.ttf"
                if not os.path.exists(font_path):
                    font_path = None

                col1, col2 = st.columns(2)

                # 막대 그래프
                if show_bar_graph:
                    with col1:
                        fig = plt.figure(figsize=(8, 6))
                        ta.visualize_barhgraph(counter, num_words_bar, font_path=font_path)
                        st.pyplot(fig)
                        plt.close()

                # 워드클라우드
                if show_wordcloud:
                    with col2:
                        fig = plt.figure(figsize=(8, 8))
                        ta.visualize_wordcloud(counter, num_words_wc, font_path)
                        st.pyplot(fig)
                        plt.close()

        except Exception as e:
            st.error(str(e))