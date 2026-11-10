"""PDF -> 추출 -> 요약 -> 다중 벡터 검색기 -> 질의응답 / Before-After 비교.

사용:
    python main.py --pdf data/내문서.pdf              # 처음부터 (hi_res 파싱 2~3분 + 요약 API 비용)
    python main.py --reuse                            # data/processed/*.pkl 재사용 (추출·요약 생략)
    python main.py --reuse --questions data/questions.txt --out qa_results.md
"""
import argparse
import pickle
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.extract import categorize_elements, extract_pdf_elements, split_texts
from src.retriever import add_documents, build_chain, create_retriever
from src.summarize import generate_img_summaries, generate_text_summaries

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
IMG_RE = re.compile(r"^[A-Za-z0-9+/]{100,}={0,2}$")  # base64 이미지 판별


def prepare(pdf, reuse):
    """추출 + 요약 결과를 만들거나(pkl 저장) 저장본을 불러온다."""
    ext_pkl, sum_pkl = PROCESSED / "extracted.pkl", PROCESSED / "summaries.pkl"
    if reuse:
        return pickle.loads(ext_pkl.read_bytes()), pickle.loads(sum_pkl.read_bytes())

    figures = DATA / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    raw = extract_pdf_elements(pdf.parent, pdf.name, figures)
    texts, tables = categorize_elements(raw)
    ext = {"texts": texts, "tables": tables, "texts_2k_token": split_texts(texts)}

    text_sum, table_sum = generate_text_summaries(ext["texts_2k_token"], tables, summarize_texts=True)
    img_b64, img_sum = generate_img_summaries(str(figures))
    summ = {"text_summaries": text_sum, "table_summaries": table_sum,
            "image_summaries": img_sum, "img_base64_list": img_b64}

    PROCESSED.mkdir(parents=True, exist_ok=True)
    ext_pkl.write_bytes(pickle.dumps(ext))
    sum_pkl.write_bytes(pickle.dumps(summ))
    return ext, summ


def build_text_only_chain(texts):
    """비교용: 이미지·표 없이 텍스트만 색인한 RAG."""
    vs = Chroma(collection_name="text_only_rag",
                embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    vs.add_texts(texts)
    prompt = ChatPromptTemplate.from_template(
        "아래 자료를 근거로 질문에 한국어로 답하세요.\n자료에 없는 내용은 모른다고 답하세요.\n\n"
        "질문: {question}\n\n참고 텍스트:\n{context}")
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)
    join = lambda docs: "\n".join(d.page_content for d in docs)
    return ({"context": vs.as_retriever(search_kwargs={"k": 4}) | join,
             "question": RunnablePassthrough()} | prompt | model | StrOutputParser())


def source_kinds(docs, tables):
    """검색된 원본의 종류(이미지/표/텍스트) 요약."""
    kinds = ["이미지" if IMG_RE.match(d) else "표" if d in tables else "텍스트" for d in docs]
    return ", ".join(f"{k} {kinds.count(k)}" for k in dict.fromkeys(kinds))


def cell(s):
    return s.strip().replace("|", "\\|").replace("\n", "<br>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=DATA / "sample_text.pdf")
    ap.add_argument("--reuse", action="store_true", help="data/processed/*.pkl 재사용")
    ap.add_argument("--questions", type=Path, default=DATA / "questions.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "qa_results.md")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    ext, summ = prepare(args.pdf, args.reuse)
    questions = [q.strip() for q in args.questions.read_text(encoding="utf-8").splitlines() if q.strip()]

    retriever = create_retriever()
    add_documents(retriever, summ["text_summaries"], ext["texts_2k_token"])
    add_documents(retriever, summ["table_summaries"], ext["tables"])
    add_documents(retriever, summ["image_summaries"], summ["img_base64_list"])
    chain, text_chain = build_chain(retriever), build_text_only_chain(ext["texts_2k_token"])

    rows = ["| # | 질문 | 답변 | 근거 원본 종류 | 소요(초) |", "|---|---|---|---|---|"]
    ba = []
    for i, q in enumerate(questions, 1):
        t = time.time()
        after = chain.invoke(q)
        rows.append(f"| {i} | {cell(q)} | {cell(after)} | "
                    f"{source_kinds(retriever.invoke(q), ext['tables'])} | {time.time() - t:.1f} |")
        ba.append((q, text_chain.invoke(q), after))

    out = ["## 질의응답", "", *rows, "", "## Before / After (텍스트 RAG vs 멀티모달 RAG)"]
    for q, before, after in ba:
        out += ["", f"### Q. {q}", "", f"**[BEFORE] 텍스트 전용**\n\n{before}", "",
                f"**[AFTER] 멀티모달**\n\n{after}"]
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
