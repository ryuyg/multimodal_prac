from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(path, fname, image_dir):
    """PDF에서 이미지(image_dir에 저장), 테이블, 텍스트 블록을 추출한다."""
    return partition_pdf(
        filename=str(path / fname),
        strategy="hi_res",  # 없으면 표/이미지 추출 정확도 급락
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],  # 그래프가 Table로 분류되는 PDF 대비
        languages=["kor", "eng"],  # 미지정 시 영어로 OCR되어 한글이 깨짐
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=str(image_dir),  # image_output_dir_path는 무시됨(루트 figures/에 저장)
    )


def categorize_elements(raw_pdf_elements):
    """추출 요소를 (texts, tables)로 분류한다."""
    tables, texts = [], []
    for element in raw_pdf_elements:
        t = str(type(element))
        if "unstructured.documents.elements.Table" in t:
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in t:
            texts.append(str(element))
    return texts, tables


def split_texts(texts, chunk_size=2000, chunk_overlap=200):
    """텍스트를 합친 뒤 토큰 기준으로 겹치게 분할한다."""
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(" ".join(texts))
