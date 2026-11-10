import base64
import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PROMPT_TEXT_KOR = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
요약할 표 또는 텍스트: {element}"""

PROMPT_IMG_KOR = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""


def generate_text_summaries(texts, tables, summarize_texts=False):
    """텍스트/표 요약 생성. 표 요약이 []이면 hi_res 누락 또는 PDF에 표 객체 없음."""
    prompt = ChatPromptTemplate.from_template(PROMPT_TEXT_KOR)
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    cfg = {"max_concurrency": 5}  # 올리면 빨라지지만 API 요청 제한 위험

    text_summaries = chain.batch(texts, cfg) if texts and summarize_texts else list(texts)
    table_summaries = chain.batch(tables, cfg) if tables else []
    return text_summaries, table_summaries


def encode_image(image_path):
    """이미지를 base64 문자열로 변환."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_summarize(img_base64, prompt=PROMPT_IMG_KOR):
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    msg = chat.invoke([HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
    ])])
    return msg.content


def generate_img_summaries(path):
    """폴더 내 이미지 전체의 (base64 리스트, 요약 리스트) 반환."""
    img_base64_list, image_summaries = [], []
    for img_file in sorted(os.listdir(path)):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            b64 = encode_image(os.path.join(path, img_file))
            img_base64_list.append(b64)
            image_summaries.append(image_summarize(b64))
    return img_base64_list, image_summaries
