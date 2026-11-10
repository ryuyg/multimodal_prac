import re
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ID_KEY = "doc_id"


def create_retriever():
    vectorstore = Chroma(collection_name="multimodal_rag", embedding_function=OpenAIEmbeddings())
    return MultiVectorRetriever(vectorstore=vectorstore, docstore=InMemoryStore(), id_key=ID_KEY)


def add_documents(retriever, doc_summaries, doc_contents):
    """요약은 벡터 DB, 원본은 docstore에 같은 id로 저장."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [Document(page_content=s, metadata={ID_KEY: doc_ids[i]})
                    for i, s in enumerate(doc_summaries)]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


def split_image_text_types(docs):
    """검색 결과를 base64 이미지와 텍스트로 분리."""
    b64_images, texts = [], []
    for doc in docs:
        doc = doc.page_content if isinstance(doc, Document) else doc
        if re.match(r"^[A-Za-z0-9+/]{100,}={0,2}$", doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    messages = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                for img in data_dict["context"]["images"]]
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({"type": "text", "text": (
        "아래 자료를 근거로 질문에 한국어로 답하세요.\n"
        "자료에 없는 내용은 모른다고 답하세요.\n\n"
        f"질문: {data_dict['question']}\n\n"
        f"참고 텍스트:\n{formatted_texts}")})
    return [HumanMessage(content=messages)]


def build_chain(retriever):
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)
    return (
        {"context": retriever | RunnableLambda(split_image_text_types),
         "question": RunnablePassthrough()}
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )
