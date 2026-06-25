from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from qdrant_client.models import PointStruct
import requests
from docx import Document
from io import BytesIO
import re
import tempfile
from collections import Counter
from docx.table import Table
from docx.text.paragraph import Paragraph
import uuid
import time

app = FastAPI()
load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL")
)

from qdrant_client.models import Distance, VectorParams

try:
    client.get_collection("orbit_docs")
    print("orbit_docs 이미 존재")
except:
    client.create_collection(
        collection_name="orbit_docs",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print("orbit_docs 생성 완료")

print("AI 임베딩 모델(bge-m3)을 로드하는 중입니다. 잠시만 기다려주세요...")
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("모델 로드 완료! 서버를 시작합니다.")

# bge-m3 최대 입력 토큰 수 (초과분은 임베딩에 반영되지 않음)
EMBED_MAX_TOKENS = 8192

class SearchRequest(BaseModel):
    query: str
    limit: int = 3

class DocumentChunkRequest(BaseModel):
    fileName: str
    signedUrl: str
    mimeType: str

class ChunkEmbedItem(BaseModel):
    chunk_seq: int
    rag_doc_seq: int
    file_name: str
    chunk_text: str

class ChunkEmbedRequest(BaseModel):
    chunks: list[ChunkEmbedItem]

class DeletePointRequest(BaseModel):
    point_ids: list[str]

# 검색용 임베딩 함수
def create_embedding(text: str):
    encoded_input = tokenizer(
        text, padding=True, truncation=True,
        return_tensors='pt', max_length=8192
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
        emb = model_output[0][:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.tolist()[0]

# 업로드 문서 배치 임베딩 함수
def create_embeddings(texts: list[str]):
    encoded_input = tokenizer(
        texts, padding=True, truncation=True,
        return_tensors='pt', max_length=8192
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
        emb = model_output[0][:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.tolist()

_STYLE_LEVEL = {
    "Heading 1": 1, "Heading 2": 2, "Heading 3": 3,
    "제목 1": 1,    "제목 2": 2,    "제목 3": 3,
}

_PATTERN_LEVEL = [
    (r'^\d+\.\d+\.\d+\s+\S', 3),
    (r'^\d+\.\d+\s+\S',      2), 
    (r'^\d+\.\d+\.\s+\S',    2),
    (r'^\d+\.\s+\S',         1), 
    (r'^제\s*\d+\s*조',       1),
    (r'^[가-힣]\.\s',         2),
    (r'^[①-⑳]\s',            2), 
    (r'^【.+】',              1),
]

_MAX_HEADER_LEN = 60

def detect_heading_level(text: str, style: str = '') -> int:
    lvl = _STYLE_LEVEL.get(style)
    if lvl:
        return lvl

    stripped = text.strip()

    if '\n' in stripped:
        return 0

    if len(stripped) > _MAX_HEADER_LEN:
        return 0

    for pattern, level in _PATTERN_LEVEL:
        if re.match(pattern, stripped):
            return level

    return 0

_MD_SEPARATOR_RE = re.compile(r'^\|(\s*-{3,}\s*\|)+$')


def _iter_unique_row_cells(row):
    seen = set()
    for cell in row.cells:
        cid = id(cell._tc)
        if cid in seen:
            continue
        seen.add(cid)
        yield cell


def _cell_text(cell) -> str:
    parts = []
    for child in cell._tc.iterchildren():
        if child.tag.endswith('}p'):
            t = Paragraph(child, cell).text.strip()
            if t:
                parts.append(t)
        elif child.tag.endswith('}tbl'):
            nested = Table(child, cell)
            for row in nested.rows:
                row_txt = ' '.join(
                    t for t in (
                        _cell_text(c) for c in _iter_unique_row_cells(row)
                    ) if t
                )
                if row_txt:
                    parts.append(row_txt)
    return ' '.join(parts)


def _direct_nested_tables(cell) -> list[Table]:
    return [
        Table(ch, cell)
        for ch in cell._tc.iterchildren()
        if ch.tag.endswith('}tbl')
    ]


def table_to_markdown(table: Table) -> str:
    rows = list(table.rows)

    # ── 1x1 래퍼 표 해제 ──
    if len(rows) == 1:
        cells = list(_iter_unique_row_cells(rows[0]))
        if len(cells) == 1:
            own_text = ' '.join(
                Paragraph(ch, cells[0]).text.strip()
                for ch in cells[0]._tc.iterchildren()
                if ch.tag.endswith('}p')
            ).strip()
            nested = _direct_nested_tables(cells[0])
            if nested and not own_text:
                return '\n'.join(
                    md for md in (table_to_markdown(t) for t in nested)
                    if md.strip()
                )

    md_rows = []
    header_emitted = False
    for row in rows:
        cells = [
            _cell_text(c).replace('\n', ' ').replace('|', '\\|')
            for c in _iter_unique_row_cells(row)
        ]
        if not any(c.strip() for c in cells):   # 빈 행 생략
            continue
        md_rows.append('| ' + ' | '.join(cells) + ' |')
        if not header_emitted:                   # 헤더 구분선
            md_rows.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')
            header_emitted = True
    return '\n'.join(md_rows)


def _merge_table_md(prev_md: str, new_md: str) -> str:
    prev_lines = [ln for ln in prev_md.split('\n') if ln.strip()]
    new_lines = [
        ln for ln in new_md.split('\n')
        if ln.strip() and not _MD_SEPARATOR_RE.match(ln.strip())
    ]
    if new_lines and prev_lines and new_lines[0].strip() == prev_lines[0].strip():
        new_lines = new_lines[1:]
    if not new_lines:
        return prev_md
    return prev_md + '\n' + '\n'.join(new_lines)


def iter_block_items(parent):
    for child in parent.element.body.iterchildren():
        if child.tag.endswith('}p'):
            yield Paragraph(child, parent)
        elif child.tag.endswith('}tbl'):
            yield Table(child, parent)


def extract_docx_blocks(file_bytes: bytes) -> list[dict]:
    doc = Document(BytesIO(file_bytes))
    blocks = []
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = block.style.name if block.style else ''
            blocks.append({
                'text': text,
                'level': detect_heading_level(text, style),
                'is_table': False,
            })
        elif isinstance(block, Table):
            md = table_to_markdown(block)
            if md.strip():
                blocks.append({'text': md, 'level': 0, 'is_table': True})
    return blocks


def _normalize_pdf_line(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'([A-Z]{2,10}-)\s+(\d+)', r'\1\2', text)   # FR- 001 → FR-001
    text = re.sub(r'Page\s+\d+\s*/\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^©\s*\d{4}.*$', '', text)                 # 줄 시작 ©만 푸터로 간주
    return text.strip()


def _remove_repeated_lines(lines: list[str], min_repeat: int = 3) -> list[str]:
    counter = Counter(lines)
    repeated = set()
    for ln, cnt in counter.items():
        if cnt < min_repeat:
            continue
        if not (5 < len(ln) <= 80):
            continue
        if detect_heading_level(ln) > 0:
            continue
        repeated.add(ln)
    return [ln for ln in lines if ln not in repeated]


def extract_pdf_blocks(file_bytes: bytes) -> list[dict]:
    try:
        from pdf2docx import Converter
    except ImportError:
        raise ImportError(
            "pdf2docx 가 설치되지 않았습니다.\n"
            "pip install pdf2docx 를 실행해 주세요."
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_tmp:
        pdf_tmp.write(file_bytes)
        pdf_path = pdf_tmp.name

    docx_path = pdf_path.replace(".pdf", "_converted.docx")

    try:
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)
        cv.close()

        with open(docx_path, "rb") as f:
            docx_bytes = f.read()

        blocks = extract_docx_blocks(docx_bytes)

    finally:
        for path in (pdf_path, docx_path):
            try:
                os.remove(path)
            except OSError:
                pass

    all_lines = []
    for block in blocks:
        if not block['is_table']:
            cleaned = _normalize_pdf_line(block['text'])
            if cleaned:
                all_lines.append(cleaned)

    clean_lines = _remove_repeated_lines(all_lines)
    clean_set = set(clean_lines)

    result = []
    for block in blocks:
        if block['is_table']:
            result.append(block)
        else:
            cleaned = _normalize_pdf_line(block['text'])
            if cleaned and cleaned in clean_set:
                result.append({
                    'text':     cleaned,
                    'level':    block['level'],
                    'is_table': False,
                })

    return result


def extract_blocks(file_name: str, file_bytes: bytes) -> list[dict]:
    lower = file_name.lower()
    if lower.endswith('.docx'):
        return extract_docx_blocks(file_bytes)
    if lower.endswith('.pdf'):
        return extract_pdf_blocks(file_bytes)
    raise ValueError(f"지원하지 않는 파일 형식: {file_name}")


MAX_CHUNK_LEN = 3000
OVERLAP_LEN   = 500
MIN_CHUNK_LEN = 30
TABLE_CONTEXT_MAX = 500


def _split_by_window(text: str,
                     chunk_size: int = MAX_CHUNK_LEN,
                     overlap: int = OVERLAP_LEN) -> list[str]:
    if not text:
        return []

    parts = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            last_nl = text.rfind('\n', start, end)
            if last_nl > start + overlap:
                end = last_nl

        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)

        next_start = end - overlap
        if next_start <= start:
            next_start = end

        start = next_start

    return parts


def _split_table_by_rows(table_md: str,
                         chunk_size: int = MAX_CHUNK_LEN) -> list[str]:
   
    lines = [ln for ln in table_md.split('\n') if ln.strip()]
    if not lines:
        return []

    header_cnt = 0
    if len(lines) >= 2 and _MD_SEPARATOR_RE.match(lines[1].strip()):
        header_cnt = 2

    header = '\n'.join(lines[:header_cnt])
    body_lines = lines[header_cnt:]

    if not body_lines:
        return [table_md]

    parts: list[str] = []
    cur: list[str] = []
    cur_len = len(header)

    for ln in body_lines:
        if cur and cur_len + len(ln) + 1 > chunk_size:
            parts.append('\n'.join(([header] if header else []) + cur))
            cur = []
            cur_len = len(header)
        cur.append(ln)
        cur_len += len(ln) + 1

    if cur:
        parts.append('\n'.join(([header] if header else []) + cur))

    return parts or [table_md]


def _build_prefix(breadcrumb: list[str], already_in_body: str) -> str:
    parts = []
    for crumb in breadcrumb:
        if crumb and crumb not in already_in_body:
            parts.append(crumb)
    return ' > '.join(parts)


def hierarchical_context_chunk(blocks: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    breadcrumb = ['', '']      # [H1, H2]
    pending: list[str] = []    # 아직 확정되지 않은 줄 누적

    def emit(text: str, ctype: str):
        chunks.append({
            'text': text,
            'chunk_type': ctype,
            'h1': breadcrumb[0],
            'h2': breadcrumb[1],
        })

    def flush(force: bool = False):
        nonlocal pending
        if not pending:
            return
        body = '\n'.join(pending).strip()
        if not body:
            pending = []
            return
        if len(body) < MIN_CHUNK_LEN and not force:
            return

        prefix = _build_prefix(breadcrumb, body)
        full_text = (prefix + '\n' + body).strip() if prefix else body

        if len(full_text) <= MAX_CHUNK_LEN:
            emit(full_text, 'text')
        else:
            for part in _split_by_window(full_text):
                emit(part, 'text')
        pending = []

    def flush_table(table_md: str):
        nonlocal pending
        context = '\n'.join(pending).strip()
        ref_body = (context + '\n' + table_md) if context else table_md
        prefix = _build_prefix(breadcrumb, ref_body)

        ctx = '\n'.join(p for p in (prefix, context) if p).strip()
        full = (ctx + '\n' + table_md).strip() if ctx else table_md

        if len(full) <= MAX_CHUNK_LEN:
            emit(full, 'table')
        else:
            budget = max(MAX_CHUNK_LEN - len(ctx) - 1, MAX_CHUNK_LEN // 2)
            for part in _split_table_by_rows(table_md, budget):
                emit((ctx + '\n' + part).strip() if ctx else part, 'table')

        pending = []

    def _pending_is_only_headers() -> bool:
        return all(
            detect_heading_level(ln) > 0
            for ln in pending
            if ln.strip()
        )

    for block in blocks:
        text    = block['text']
        level   = block['level']
        is_tbl  = block['is_table']

        if level == 1:
            flush()
            breadcrumb[0] = text
            breadcrumb[1] = ''
            pending.append(text)

        elif level == 2:
            if _pending_is_only_headers():
                breadcrumb[1] = text
                pending.append(text)
            else:
                flush()
                breadcrumb[1] = text
                pending.append(text)

        elif level == 3:
            if _pending_is_only_headers():
                pending.append(text)
            else:
                flush()
                pending.append(text)

        elif is_tbl:
            if (
                pending
                and not _pending_is_only_headers()
                and sum(len(ln) for ln in pending) > TABLE_CONTEXT_MAX
            ):
                flush()
            flush_table(text)

        else:
            pending.append(text)
            if sum(len(ln) for ln in pending) > MAX_CHUNK_LEN:
                flush()

    flush(force=True)

    merged: list[dict] = []
    carry: dict | None = None

    for chunk in chunks:
        if carry:
            chunk['text'] = carry['text'] + '\n' + chunk['text']
            if not chunk['h1']:
                chunk['h1'] = carry['h1']
            if not chunk['h2']:
                chunk['h2'] = carry['h2']
            carry = None

        if len(chunk['text'].strip()) < MIN_CHUNK_LEN:
            carry = chunk
        else:
            merged.append(chunk)

    if carry:
        merged.append(carry)

    return merged


def chunk_document(file_name: str, file_bytes: bytes) -> list[dict]:
    blocks = extract_blocks(file_name, file_bytes)
    merged_blocks = []

    for block in blocks:
        if (
            merged_blocks
            and block["is_table"]
            and merged_blocks[-1]["is_table"]
        ):
            merged_blocks[-1]["text"] = _merge_table_md(
                merged_blocks[-1]["text"],
                block["text"]
            )
        else:
            merged_blocks.append(block)

    return hierarchical_context_chunk(merged_blocks)

def extract_text(file_name: str, file_bytes: bytes) -> str:

    blocks = extract_blocks(
        file_name,
        file_bytes
    )

    for i, block in enumerate(blocks):
        print(f"===== BLOCK {i} =====")
        print(block["text"])

    return "\n".join(
        block["text"]
        for block in blocks
    )

@app.post("/search")
def search(request: SearchRequest):
    query_vector = create_embedding(request.query)
    results = client.query_points(
        collection_name="orbit_docs",
        query=query_vector,
        limit=request.limit
    )
    return [
        {
            "chunk_seq":   p.payload.get("chunk_seq"),
            "rag_doc_seq": p.payload.get("rag_doc_seq"),
            "file_name":   p.payload.get("file_name"),
            "text":        p.payload.get("chunk_text"),
            "score":       p.score,
        }
        for p in results.points
    ]

@app.post("/document/chunk")
def document_chunk(request: DocumentChunkRequest):

    response = requests.get(request.signedUrl)
    response.raise_for_status()

    start = time.time()

    raw_text = extract_text(
        request.fileName,
        response.content
    )

    chunks = chunk_document(
        request.fileName,
        response.content
    )

    print(f"청킹 시간 = {time.time() - start:.2f}초")
    return {
        "raw_text": raw_text,
        "chunks": [
            {
                "chunk_index": idx,
                "chunk_text": chunk["text"]
            }
            for idx, chunk in enumerate(chunks)
        ]
    }

@app.post("/embed/chunks")
def embed_chunks(request: ChunkEmbedRequest):

    points = []
    save_points = []

    texts = [
        chunk.chunk_text
        for chunk in request.chunks
    ]

    embed_start = time.time()

    vectors = create_embeddings(texts)
    print("청크 수 = ", len(texts))
    print(f"임베딩 시간 = {time.time() - embed_start:.2f}초")

    for i, text in enumerate(texts):
        token_count = len(
            tokenizer.encode(
                text,
                add_special_tokens=True
            )
        )

        if token_count > EMBED_MAX_TOKENS:
            print(
                f"[경고][CHUNK {i+1}] tokens = {token_count} "
                f"> {EMBED_MAX_TOKENS} → 초과분은 임베딩에서 잘립니다. "
                "청킹 단계 분할 기준(MAX_CHUNK_LEN)을 확인하세요."
            )
        else:
            print(f"[CHUNK {i+1}] tokens = {token_count}")

    for chunk, vector in zip(request.chunks, vectors):

        point_id = str(uuid.uuid4())

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_seq": chunk.chunk_seq,
                    "rag_doc_seq": chunk.rag_doc_seq,
                    "file_name": chunk.file_name,
                    "chunk_text": chunk.chunk_text
                }
            )
        )

        save_points.append({
            "chunk_seq": chunk.chunk_seq,
            "point_id": point_id
        })
    upsert_start = time.time()
    client.upsert(
        collection_name="orbit_docs",
        points=points
    )
    print(f"Qdrant 저장 시간 = {time.time() - upsert_start:.2f}초")
    return {
        "success": True,
        "points" : save_points
    }

@app.post("/delete/points")
def delete_points(req: DeletePointRequest):

    client.delete(
        collection_name="orbit_docs",
        points_selector=req.point_ids
    )

    print(f"Qdrant 삭제 완료 = {len(req.point_ids)}개")
    return {
        "success": True
    }

print("QDRANT_URL =", os.getenv("QDRANT_URL"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)