import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import platform

# 한글 폰트 설정 (OS별로 다름)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
    font_path = 'C:/Windows/Fonts/malgun.ttf'
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
    font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
else: # Linux
    plt.rc('font', family='NanumGothic')
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 페이지 기본 설정
st.set_page_config(page_title="단어 빈도수 시각화", layout="wide")

def main():
    st.title("단어 빈도수 시각화")

    # 사이드바 설정
    with st.sidebar:
        st.header("파일 선택")
        uploaded_file = st.file_uploader("Drag and drop file here", type=['csv'])
        
        st.caption("데이터가 있는 컬럼명")
        col_name = st.text_input("", value="review", label_visibility="collapsed")
        check_data_btn = st.button("데이터 파일 확인")
        
        st.divider()
        
        st.header("설정")
        
        # 빈도수 그래프 설정
        show_bar_chart = st.checkbox("빈도수 그래프", value=True)
        if show_bar_chart:
            bar_word_count = st.slider("단어 수", min_value=10, max_value=50, value=20, key='bar_slider')
        
        # 워드클라우드 설정
        show_wordcloud = st.checkbox("워드클라우드", value=False)
        if show_wordcloud:
            wc_word_count = st.slider("단어 수", min_value=20, max_value=500, value=50, key='wc_slider')
            
        start_btn = st.button("분석 시작", type="primary")

    # 데이터 파일 확인 버튼 동작
    if uploaded_file and check_data_btn:
        df = pd.read_csv(uploaded_file)
        st.write("데이터 미리보기:")
        st.dataframe(df.head())

    # 분석 시작 버튼 동작
    if uploaded_file and start_btn:
        df = pd.read_csv(uploaded_file)
        
        if col_name in df.columns:
            # 결측치 제거 및 텍스트 데이터 추출
            text_data = df[col_name].dropna().astype(str).tolist()
            text = " ".join(text_data)
            
            # --- 텍스트 토큰화 ---
            # 주의: 여기서는 띄어쓰기 기준으로 단순 분리합니다.
            # 이미지처럼 '연기', '감동' 등 의미 있는 형태소만 추출하려면 KoNLPy 등의 형태소 분석기 사용을 권장합니다.
            words = text.split() 
            
            # 빈도수 계산
            counter = Counter(words)
            
            # 성공 메시지
            st.success(f"분석이 완료되었습니다 ({len(text_data):,}개의 리뷰, {len(words):,}개의 단어)")

            # 결과 출력을 위한 컬럼 나누기 (둘 다 체크 시 병렬 배치 등 조정 가능하지만, 세로 배치가 기본)
            
            # 1. 빈도수 그래프 출력
            if show_bar_chart:
                st.subheader("단어 빈도수 그래프")
                top_words = counter.most_common(bar_word_count)
                
                if top_words:
                    words_list, counts = zip(*top_words)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # 빈도수가 높은 것이 위로 가도록 역순 정렬
                    ax.barh(words_list[::-1], counts[::-1])
                    ax.set_xlabel("빈도수")
                    st.pyplot(fig)

            # 2. 워드클라우드 출력
            if show_wordcloud:
                st.subheader("워드클라우드")
                wc = WordCloud(
                    font_path=font_path,
                    background_color="white",
                    max_words=wc_word_count,
                    width=800, 
                    height=600
                )
                wc.generate_from_frequencies(counter)
                
                fig_wc, ax_wc = plt.subplots(figsize=(10, 8))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off") # 축 숨기기
                st.pyplot(fig_wc)
                
        else:
            st.error(f"업로드한 파일에 '{col_name}' 컬럼이 존재하지 않습니다. 컬럼명을 다시 확인해 주세요.")

if __name__ == "__main__":
    main()