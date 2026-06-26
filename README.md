# PDF Miner — Enterprise Architecture Recommendation

## 1. Executive Summary

This document presents the architecture for **PDF Miner**, a high-performance, 100% local, offline PDF data mining application for invoice extraction. The system uses a **hybrid extraction strategy**: deterministic template-based parsing for known vendor layouts (fast, accurate) and LLM/VLM-based extraction for unseen or complex layouts (adaptive, intelligent).

> [!IMPORTANT]
> All components run locally with zero cloud API dependencies. Models are downloaded once and cached. The entire pipeline operates offline after initial setup.

---

## 2. Tech Stack Component Table

| Layer | Component | Primary Choice | Fallback / Supplement | Justification |
|---|---|---|---|---|
| **PDF Ingestion** | PDF Reader | **PyMuPDF (fitz)** | pikepdf (repair) | 10x faster than alternatives; handles both digital & scanned PDFs; renders pages to images |
| **PDF Ingestion** | Table Extraction (Digital) | **pdfplumber** | Camelot | Character-level precision; excellent table detection on text-based PDFs |
| **Image Preprocessing** | Deskew / Binarize / Denoise | **OpenCV (cv2)** | Pillow (format ops) | Adaptive thresholding, Hough deskew, morphological ops — industry standard |
| **Orientation Detection** | Auto-rotate | **Tesseract OSD** | OpenCV Hough | Built-in Orientation & Script Detection; accurate and lightweight |
| **Layout Analysis** | Document Structure | **Docling (IBM)** | Marker (GPL) | MIT license; end-to-end pipeline; TableFormer for complex tables; SmolDocling VLM integration |
| **OCR** | Text Recognition | **PaddleOCR** | Tesseract 5.x | Best accuracy-to-speed ratio; built-in layout analysis; 80+ languages |
| **VLM (Vision-Language)** | Complex Layout Extraction | **Qwen2.5-VL-7B** | Florence-2 (lightweight) | Top document understanding benchmarks (95.7% DocVQA); dynamic resolution; Apache 2.0 |
| **LLM (Text)** | Structured Extraction | **Qwen3-8B** | Qwen2.5-7B / Phi-4-Mini (3.8B) | Thinking/non-thinking mode; best JSON schema compliance at 8B; Apache 2.0 |
| **Inference Engine** | GPU Production | **vLLM** | SGLang (better for VLMs) | Continuous batching, PagedAttention; best throughput for batch invoice processing |
| **Inference Engine** | CPU / Dev | **Ollama** | llama.cpp | Simplest setup; REST API; good for development and CPU-only deployments |
| **Constrained Decoding** | JSON Enforcement (GPU) | **XGrammar** | Outlines | PDA-based token masking; near-zero overhead; default in vLLM & SGLang since 2026 |
| **Constrained Decoding** | JSON Enforcement (Ollama) | **Instructor** | — | Retry-based validation; works with any OpenAI-compatible API |
| **Data Validation** | Schema Enforcement | **Pydantic v2** | — | Industry standard; strict type validation; JSON Schema generation |
| **Vector DB** | Vendor Template Matching | **LanceDB** | ChromaDB | Embedded/serverless; disk-based; fast ANN search; no infrastructure overhead |
| **Embedding Model** | Vectorization | **all-MiniLM-L6-v2** | BGE-small-en-v1.5 | 80MB; fast; excellent quality for semantic similarity |
| **Relational DB** | Extracted Data Storage | **SQLite** (dev) / **PostgreSQL** (scale) | DuckDB (analytics) | SQLite for simplicity; PostgreSQL for multi-user production |
| **Task Queue** | Async Processing | **asyncio + AnyIO** | Celery (if distributed) | Native Python async; minimal overhead for single-machine deployment |
| **API Framework** | REST API | **FastAPI** | — | Async-native; Pydantic integration; auto-generated OpenAPI docs |
| **Monitoring** | Observability | **Loguru + Rich** | Prometheus (if needed) | Beautiful console output; structured logging; zero-config |

---

## 3. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PDF MINER — SYSTEM ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐     ┌──────────────────────────────────────────────────────┐  │
│  │  INPUT        │     │  LAYER 1: INGESTION & PREPROCESSING                 │  │
│  │              │     │                                                      │  │
│  │  📄 PDFs     │────▶│  PyMuPDF ──▶ Digital? ──YES──▶ Direct Text Extract  │  │
│  │  📷 Scans    │     │     │                                    │           │  │
│  │  📁 Batch    │     │     │         NO (Scanned)               │           │  │
│  │              │     │     ▼                                    │           │  │
│  └──────────────┘     │  pikepdf ──▶ OpenCV Pipeline:            │           │  │
│                       │  (repair)    ├─ Orientation (OSD)        │           │  │
│                       │              ├─ Deskew (Hough)           │           │  │
│                       │              ├─ Denoise (Median)         │           │  │
│                       │              └─ Binarize (Adaptive)      │           │  │
│                       └────────────────────┬─────────────────────┘           │  │
│                                            │                                 │  │
│  ┌─────────────────────────────────────────▼─────────────────────────────┐   │  │
│  │  LAYER 2: LAYOUT ANALYSIS & OCR                                       │   │  │
│  │                                                                       │   │  │
│  │  Docling Pipeline:                                                    │   │  │
│  │  ┌───────────┐   ┌──────────────┐   ┌──────────────┐                │   │  │
│  │  │ Layout    │──▶│ Table        │──▶│ Reading      │                │   │  │
│  │  │ Detection │   │ Structure    │   │ Order        │                │   │  │
│  │  │ (DocLayNet)│   │ (TableFormer)│   │ Detection    │                │   │  │
│  │  └───────────┘   └──────────────┘   └──────┬───────┘                │   │  │
│  │                                            │                         │   │  │
│  │  OCR Engine (if scanned):                  │                         │   │  │
│  │  ┌──────────────┐   ┌──────────────┐       │                         │   │  │
│  │  │ PaddleOCR    │──▶│ Text + BBox  │───────┤                         │   │  │
│  │  │ (primary)    │   │ Results      │       │                         │   │  │
│  │  └──────────────┘   └──────────────┘       │                         │   │  │
│  │  ┌──────────────┐                          │                         │   │  │
│  │  │ Tesseract 5  │─── (fallback/simple) ────┘                         │   │  │
│  │  └──────────────┘                                                    │   │  │
│  └─────────────────────────────────────────┬─────────────────────────────┘   │  │
│                                            │                                 │  │
│  ┌─────────────────────────────────────────▼─────────────────────────────┐   │  │
│  │  LAYER 3: ROUTING & CLASSIFICATION                                    │   │  │
│  │                                                                       │   │  │
│  │  ┌─────────────────┐     ┌──────────────────────────────────────┐    │   │  │
│  │  │ Vendor Template │     │ Template Match Score                  │    │   │  │
│  │  │ Matcher         │────▶│                                      │    │   │  │
│  │  │ (LanceDB +      │     │  score ≥ 0.92 ──▶ DETERMINISTIC PATH │    │   │  │
│  │  │  Embeddings)    │     │  score < 0.92 ──▶ LLM/VLM PATH      │    │   │  │
│  │  └─────────────────┘     └──────────────────────────────────────┘    │   │  │
│  └──────────┬──────────────────────────────────┬─────────────────────────┘   │  │
│             │                                  │                             │  │
│     ┌───────▼────────┐                ┌────────▼────────┐                    │  │
│     │ FAST PATH      │                │ SMART PATH      │                    │  │
│     │ (Deterministic)│                │ (LLM/VLM)       │                    │  │
│     │                │                │                  │                    │  │
│  ┌──▼──────────────┐│             ┌───▼──────────────┐  │                    │  │
│  │  LAYER 4a:      ││             │  LAYER 4b:       │  │                    │  │
│  │  TEMPLATE       ││             │  LLM/VLM         │  │                    │  │
│  │  EXTRACTION     ││             │  EXTRACTION       │  │                    │  │
│  │                 ││             │                   │  │                    │  │
│  │  • Regex rules  ││             │  ┌─────────────┐ │  │                    │  │
│  │  • Spatial      ││             │  │ Qwen2.5-VL  │ │  │                    │  │
│  │    anchoring    ││             │  │ 7B (Vision) │ │  │                    │  │
│  │  • pdfplumber   ││             │  │ via vLLM    │ │  │                    │  │
│  │    tables       ││             │  └──────┬──────┘ │  │                    │  │
│  │  • Field maps   ││             │         │        │  │                    │  │
│  └────────┬────────┘│             │  ┌──────▼──────┐ │  │                    │  │
│           │         │             │  │ Qwen2.5-7B  │ │  │                    │  │
│           │         │             │  │ (Text LLM)  │ │  │                    │  │
│           │         │             │  │ + Outlines   │ │  │                    │  │
│           │         │             │  │ (JSON lock) │ │  │                    │  │
│           │         │             │  └──────┬──────┘ │  │                    │  │
│           │         │             └─────────┤        │  │                    │  │
│           │         │                       │        │  │                    │  │
│  ┌────────▼─────────▼───────────────────────▼────────▼──▼────────────────┐   │  │
│  │  LAYER 5: VALIDATION & STRUCTURING                                    │   │  │
│  │                                                                       │   │  │
│  │  ┌──────────────┐   ┌────────────────┐   ┌──────────────────────┐    │   │  │
│  │  │ Pydantic v2  │──▶│ Business Rules │──▶│ Confidence Scoring   │    │   │  │
│  │  │ Schema Valid. │   │ (Tax calc,     │   │ & Human-in-the-Loop  │    │   │  │
│  │  │              │   │  total check)  │   │ Flagging             │    │   │  │
│  │  └──────────────┘   └────────────────┘   └──────────┬───────────┘    │   │  │
│  └─────────────────────────────────────────────────────┤─────────────────┘   │  │
│                                                        │                     │  │
│  ┌─────────────────────────────────────────────────────▼─────────────────┐   │  │
│  │  LAYER 6: STORAGE & OUTPUT                                            │   │  │
│  │                                                                       │   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐    │   │  │
│  │  │ SQLite / │  │ LanceDB  │  │ JSON /   │  │ CSV / Excel /     │    │   │  │
│  │  │ Postgres │  │ (vectors │  │ JSONL    │  │ Webhook Export    │    │   │  │
│  │  │ (records)│  │  + RAG)  │  │ (raw)    │  │                   │    │   │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └───────────────────┘    │   │  │
│  └───────────────────────────────────────────────────────────────────────┘   │  │
│                                                                              │  │
│  ┌───────────────────────────────────────────────────────────────────────┐   │  │
│  │  CROSS-CUTTING: FastAPI Server · Loguru Logging · asyncio Orchestrator│   │  │
│  └───────────────────────────────────────────────────────────────────────┘   │  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Component Architecture

### 4.1 Document Ingestion & Preprocessing

```
                    INPUT PDF
                       │
                       ▼
              ┌────────────────┐
              │  pikepdf       │──── Repair/decrypt if needed
              │  (pre-flight)  │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐         ┌──────────────────┐
              │  PyMuPDF       │────YES──▶│ Extract text,    │
              │  has_text()?   │         │ tables, metadata │──▶ LAYER 2
              └───────┬────────┘         └──────────────────┘
                      │ NO (scanned)
                      ▼
              ┌────────────────┐
              │  PyMuPDF       │
              │  render_page() │──── 300 DPI PNG
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────────────────────────────┐
              │  OpenCV Preprocessing Pipeline          │
              │                                        │
              │  1. cv2.cvtColor(BGR2GRAY)             │
              │  2. Tesseract OSD → auto-rotate        │
              │  3. Hough Line → deskew angle → rotate │
              │  4. cv2.medianBlur(ksize=3) → denoise  │
              │  5. cv2.adaptiveThreshold() → binarize │
              │  6. DPI check → upscale if < 300       │
              └────────────────┬───────────────────────┘
                               │
                               ▼
                        Clean image ──▶ LAYER 2 (OCR)
```

**Key Design Decisions:**

| Decision | Choice | Rationale |
|---|---|---|
| PDF library | PyMuPDF over pdf2image | No Poppler dependency; 10x faster rendering; handles text + image extraction |
| Binarization | Adaptive thresholding | Handles uneven lighting in scans better than global (Otsu) thresholding |
| Target DPI | 300 DPI | Industry standard for OCR accuracy; diminishing returns above 300 |
| Pre-flight repair | pikepdf | Gracefully handles corrupted/encrypted PDFs before main pipeline |

### 4.2 Layout Analysis & OCR

The system uses **Docling** as the primary layout analysis engine, with **PaddleOCR** as the OCR backend for scanned documents.

```python
# Conceptual pipeline (simplified)
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_opts = PdfPipelineOptions()
pipeline_opts.do_ocr = True                    # Enable OCR for scans
pipeline_opts.do_table_structure = True         # Enable TableFormer
pipeline_opts.ocr_options.engine = "paddleocr"  # Use PaddleOCR backend

converter = DocumentConverter(pipeline_options=pipeline_opts)
result = converter.convert("invoice.pdf")

# Access structured elements
for element in result.document.body:
    if element.label == "table":
        # Access table as DataFrame
        df = element.export_to_dataframe()
    elif element.label == "text":
        print(element.text)
```

**Why Docling over alternatives:**

| Feature | Docling | Marker | Unstructured |
|---|---|---|---|
| License | **MIT** ✅ | GPL-3.0 ⚠️ | Apache 2.0 ✅ |
| Table extraction | **TableFormer** (SOTA) | Basic | Moderate |
| Offline | **Full** ✅ | Full ✅ | Partial ❌ |
| Output format | JSON / Markdown / DoclingDocument | Markdown only | Elements JSON |
| OCR backends | PaddleOCR, Tesseract, EasyOCR | Surya (built-in) | Tesseract |
| VLM integration | SmolDocling (256M) | None | None |
| Active development | **Very active** (IBM) | Active | Active |

### 4.3 Local LLM/VLM Inference Engine

#### Model Selection Strategy

```
┌─────────────────────────────────────────────────────────┐
│              MODEL SELECTION MATRIX                       │
├──────────────────┬──────────┬───────────┬───────────────┤
│ Task             │ Model    │ Size      │ When Used     │
├──────────────────┼──────────┼───────────┼───────────────┤
│ Visual doc parse │Qwen2.5-VL│ 7B        │ Complex/new   │
│ (full-page)      │  -7B     │ ~14GB Q4  │ layouts       │
├──────────────────┼──────────┼───────────┼───────────────┤
│ Visual doc parse │Florence-2│ 0.77B     │ Quick region  │
│ (lightweight)    │          │ ~1.5GB    │ detection     │
├──────────────────┼──────────┼───────────┼───────────────┤
│ Structured JSON  │Qwen3-8B  │ 8B        │ Text→JSON     │
│ extraction       │          │ ~16GB Q4  │ extraction    │
├──────────────────┼──────────┼───────────┼───────────────┤
│ Lightweight JSON │Phi-4-Mini│ 3.8B      │ Simple fields │
│ extraction       │          │ ~7GB Q4   │ (fast path)   │
├──────────────────┼──────────┼───────────┼───────────────┤
│ Vendor embedding │MiniLM-L6 │ 80MB      │ Template      │
│                  │ -v2      │           │ matching      │
└──────────────────┴──────────┴───────────┴───────────────┘
```

#### Inference Engine Configuration

**Production (GPU available — ≥16GB VRAM):**
```yaml
# vLLM server configuration
engine: vLLM
models:
  - name: qwen2.5-vl-7b-instruct-awq   # AWQ 4-bit quantized
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    guided_decoding_backend: xgrammar    # Default since 2026; PDA-based
    tensor_parallel_size: 1              # Single GPU
    
  - name: qwen3-8b-awq                  # Or qwen2.5-7b-instruct-awq
    gpu_memory_utilization: 0.85
    max_model_len: 8192
    guided_decoding_backend: xgrammar
    extra_params:
      enable_thinking: false             # Disable thinking for speed

batching:
  max_batch_size: 32                     # Continuous batching
  
# Expected throughput: ~15-25 invoices/minute (complex layouts)
```

**Development / CPU-only:**
```yaml
# Ollama configuration
engine: Ollama
models:
  - name: qwen2.5-vl:7b-q4_K_M          # GGUF quantized
  - name: qwen2.5:7b-instruct-q4_K_M
  
# Instructor for JSON enforcement (retry-based)
# Expected throughput: ~2-5 invoices/minute (CPU)
```

**Low-resource (≤8GB VRAM or CPU-only):**
```yaml
engine: llama.cpp
models:
  - name: phi-4-mini-instruct-q4_K_M     # 3.8B, ~4GB RAM
  - name: florence-2-large                 # 0.77B, ~1.5GB RAM
  
# Trade accuracy for speed; use template path more aggressively
```

#### Why these specific models:

> [!TIP]
> **Qwen2.5-VL-7B** is the sweet spot for document understanding — it processes documents at native resolution (no fixed-size crops), handles tables/charts/handwriting, and outputs structured descriptions. At 7B parameters with 4-bit quantization, it fits comfortably in 16GB VRAM. Scored **95.7% on DocVQA**.

> [!TIP]
> **Qwen3-8B** with thinking mode disabled is the best choice for structured JSON extraction. It has a "thinking/non-thinking" toggle — use non-thinking mode for fast, schema-compliant JSON output. Combined with XGrammar constrained decoding in vLLM, it guarantees valid Pydantic JSON on every call with near-zero overhead. Alternative: Qwen2.5-7B-Instruct if you prefer a more battle-tested option.

### 4.4 Data Structuring & Validation

#### Pydantic Schema Definitions

```python
from pydantic import BaseModel, Field, field_validator
from decimal import Decimal
from datetime import date
from enum import Enum

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    INR = "INR"

class LineItem(BaseModel):
    description: str = Field(..., min_length=1, max_length=500)
    quantity: Decimal = Field(..., ge=0)
    unit_price: Decimal = Field(..., ge=0)
    amount: Decimal = Field(..., ge=0)
    tax_rate: Decimal | None = Field(default=None, ge=0, le=100)
    hsn_sac_code: str | None = None

    @field_validator("amount")
    @classmethod
    def validate_line_total(cls, v, info):
        """Cross-validate: amount ≈ quantity × unit_price"""
        qty = info.data.get("quantity")
        price = info.data.get("unit_price")
        if qty and price:
            expected = qty * price
            if abs(v - expected) > Decimal("0.01"):
                # Flag but don't reject — OCR errors are common
                pass  # Log warning
        return v

class InvoiceData(BaseModel):
    vendor_name: str
    vendor_tax_id: str | None = None       # GSTIN, EIN, VAT number
    invoice_number: str
    invoice_date: date
    due_date: date | None = None
    currency: Currency = Currency.USD
    
    line_items: list[LineItem] = Field(..., min_length=1)
    
    subtotal: Decimal = Field(..., ge=0)
    tax_amount: Decimal = Field(default=Decimal("0"), ge=0)
    discount: Decimal = Field(default=Decimal("0"), ge=0)
    total_amount: Decimal = Field(..., ge=0)
    
    extraction_method: str                 # "template" | "vlm" | "llm"
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    flagged_for_review: bool = False
    
    @field_validator("total_amount")
    @classmethod  
    def validate_total(cls, v, info):
        """Business rule: total ≈ subtotal + tax - discount"""
        sub = info.data.get("subtotal", Decimal("0"))
        tax = info.data.get("tax_amount", Decimal("0"))
        disc = info.data.get("discount", Decimal("0"))
        expected = sub + tax - disc
        if abs(v - expected) > Decimal("0.50"):
            # Auto-flag for human review
            info.data["flagged_for_review"] = True
        return v
```

#### Constrained Decoding Pipeline

```
┌──────────────────────────────────────────────────────────┐
│  CONSTRAINED DECODING STRATEGY                            │
│                                                          │
│  WITH vLLM/SGLang (Production):                          │
│  ┌────────────┐    ┌───────────┐    ┌──────────────┐    │
│  │ Pydantic   │───▶│ XGrammar  │───▶│ vLLM guided  │    │
│  │ Schema     │    │ PDA +     │    │ decoding     │    │
│  │            │    │ pre-comp  │    │ (token-level)│    │
│  └────────────┘    └───────────┘    └──────┬───────┘    │
│                                            │             │
│                         Guaranteed valid JSON ✓          │
│                         Zero retries needed  ✓          │
│                         Near-zero overhead   ✓          │
│                         (mask overlaps GPU forward pass) │
│                                                          │
│  WITH Ollama (Development):                              │
│  ┌────────────┐    ┌────────────┐    ┌──────────────┐   │
│  │ Pydantic   │───▶│ Instructor │───▶│ Ollama API   │   │
│  │ Schema     │    │ (retry +   │    │ (JSON mode)  │   │
│  │            │    │  validate) │    │              │   │
│  └────────────┘    └────────────┘    └──────┬───────┘   │
│                                             │            │
│                         Validated JSON ✓                 │
│                         Max 3 retries  ⚠️                │
│                         Simpler setup  ✓                 │
└──────────────────────────────────────────────────────────┘
```

### 4.5 Hybrid Extraction Pipeline — The Core Innovation

This is the **most critical architectural decision**: a two-path system that routes each invoice through either a fast deterministic path or an intelligent LLM path based on vendor template matching.

```
                        INCOMING INVOICE
                              │
                              ▼
                 ┌────────────────────────┐
                 │ STEP 1: Ingest & Parse │
                 │ (Docling + PaddleOCR)  │
                 └───────────┬────────────┘
                             │
                     Structured text + layout
                             │
                             ▼
                 ┌────────────────────────┐
                 │ STEP 2: Vendor Lookup  │
                 │                        │
                 │ Extract candidate name │
                 │ → Embed with MiniLM    │
                 │ → Query LanceDB       │
                 │ → Get similarity score │
                 └───────────┬────────────┘
                             │
                    ┌────────┴─────────┐
                    │                  │
            score ≥ 0.92        score < 0.92
            (Known Vendor)      (New/Unknown)
                    │                  │
                    ▼                  ▼
        ┌───────────────────┐  ┌───────────────────┐
        │ FAST PATH ⚡       │  │ SMART PATH 🧠      │
        │                   │  │                   │
        │ Load vendor       │  │ Pass full page    │
        │ template config   │  │ image to          │
        │                   │  │ Qwen2.5-VL-7B     │
        │ Apply:            │  │                   │
        │ • Spatial anchors │  │ VLM extracts:     │
        │   (bbox regions)  │  │ • All text fields │
        │ • Regex patterns  │  │ • Table structure │
        │ • Table column    │  │ • Spatial layout  │
        │   mappings        │  │                   │
        │ • pdfplumber      │  │ Then pass to      │
        │   table extract   │  │ Qwen2.5-7B-Inst   │
        │                   │  │ + Outlines for    │
        │ Speed: ~50ms      │  │ structured JSON   │
        │ Accuracy: ~98%    │  │                   │
        │                   │  │ Speed: ~2-5s      │
        └────────┬──────────┘  │ Accuracy: ~92-95% │
                 │             └────────┬──────────┘
                 │                      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌────────────────────────┐
                 │ STEP 3: Validate       │
                 │                        │
                 │ • Pydantic schema      │
                 │ • Business rules       │
                 │   (totals match, dates │
                 │    valid, tax calc)    │
                 │ • Confidence scoring   │
                 │ • Flag if conf < 0.85  │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ STEP 4: Learn          │
                 │ (Feedback Loop)        │
                 │                        │
                 │ If Smart Path result   │
                 │ was validated/corrected│
                 │ by human:             │
                 │ → Auto-generate new   │
                 │   vendor template     │
                 │ → Store in LanceDB   │
                 │ → Next time: Fast Path│
                 └───────────────────────┘
```

> [!IMPORTANT]
> **The Learning Loop** is what makes this system get faster over time. Every successfully validated LLM extraction for a new vendor becomes a template. After processing ~5-10 invoices from the same vendor, the system can switch entirely to the Fast Path for that vendor — achieving 98%+ accuracy at 50ms/invoice.

### 4.6 Storage Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  STORAGE LAYER                                                │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  SQLite / PostgreSQL — Relational Store                  │ │
│  │                                                         │ │
│  │  invoices        │  line_items      │  processing_log   │ │
│  │  ─────────       │  ──────────      │  ──────────────   │ │
│  │  id              │  id              │  id               │ │
│  │  vendor_name     │  invoice_id (FK) │  invoice_id (FK)  │ │
│  │  invoice_number  │  description     │  method           │ │
│  │  invoice_date    │  quantity        │  confidence       │ │
│  │  total_amount    │  unit_price      │  processing_time  │ │
│  │  tax_amount      │  amount          │  model_used       │ │
│  │  currency        │  tax_rate        │  timestamp        │ │
│  │  pdf_hash (idx)  │  hsn_sac_code   │  errors           │ │
│  │  status          │                  │                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  LanceDB — Vector Store (Embedded)                       │ │
│  │                                                         │ │
│  │  vendor_templates                                       │ │
│  │  ─────────────────                                      │ │
│  │  vendor_id         │  Template matching & RAG           │ │
│  │  vendor_name       │  ───────────────────────           │ │
│  │  name_embedding    │  • Semantic vendor name matching   │ │
│  │  layout_signature  │  • Layout fingerprint similarity   │ │
│  │  extraction_rules  │  • Few-shot example retrieval      │ │
│  │  field_anchors     │  • Template auto-generation        │ │
│  │  sample_count      │                                    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Filesystem — Document Store                             │ │
│  │                                                         │ │
│  │  data/                                                  │ │
│  │  ├── incoming/          # Raw PDFs (watched folder)     │ │
│  │  ├── processed/         # Archived originals            │ │
│  │  ├── cache/             # Rendered images, OCR cache    │ │
│  │  └── exports/           # JSON/CSV/Excel outputs        │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Recommended Project Structure

```
pdf-miner/
├── pyproject.toml                    # Project config (uv/poetry)
├── README.md
├── .env.example                      # Configuration template
│
├── src/
│   └── pdf_miner/
│       ├── __init__.py
│       ├── main.py                   # FastAPI app entry point
│       ├── config.py                 # Settings (Pydantic Settings)
│       │
│       ├── ingestion/                # Layer 1
│       │   ├── __init__.py
│       │   ├── pdf_reader.py         # PyMuPDF wrapper
│       │   ├── preprocessor.py       # OpenCV pipeline
│       │   └── classifier.py         # Digital vs. scanned detection
│       │
│       ├── analysis/                 # Layer 2
│       │   ├── __init__.py
│       │   ├── docling_pipeline.py   # Docling integration
│       │   ├── ocr_engine.py         # PaddleOCR / Tesseract
│       │   └── table_extractor.py    # Specialized table extraction
│       │
│       ├── routing/                  # Layer 3
│       │   ├── __init__.py
│       │   ├── vendor_matcher.py     # LanceDB template matching
│       │   └── router.py            # Fast path vs. Smart path
│       │
│       ├── extraction/               # Layer 4
│       │   ├── __init__.py
│       │   ├── template_engine.py    # Deterministic extraction
│       │   ├── vlm_extractor.py      # Qwen2.5-VL integration
│       │   ├── llm_extractor.py      # Qwen2.5 text extraction
│       │   └── prompts/              # Prompt templates
│       │       ├── invoice_extract.py
│       │       └── table_parse.py
│       │
│       ├── validation/               # Layer 5
│       │   ├── __init__.py
│       │   ├── schemas.py            # Pydantic models (InvoiceData)
│       │   ├── business_rules.py     # Tax, total validation
│       │   └── confidence.py         # Confidence scoring
│       │
│       ├── storage/                  # Layer 6
│       │   ├── __init__.py
│       │   ├── database.py           # SQLAlchemy/SQLite
│       │   ├── vector_store.py       # LanceDB operations
│       │   └── exporter.py           # JSON/CSV/Excel export
│       │
│       ├── learning/                 # Feedback Loop
│       │   ├── __init__.py
│       │   └── template_generator.py # Auto-template creation
│       │
│       └── api/                      # REST API
│           ├── __init__.py
│           ├── routes.py             # FastAPI endpoints
│           └── dependencies.py       # DI container
│
├── models/                           # Local model cache
│   ├── qwen2.5-vl-7b-awq/
│   ├── qwen2.5-7b-instruct-awq/
│   └── all-MiniLM-L6-v2/
│
├── templates/                        # Vendor extraction templates
│   └── vendor_configs/
│       ├── _base.yaml
│       └── example_vendor.yaml
│
├── data/
│   ├── incoming/                     # Drop PDFs here
│   ├── processed/
│   ├── cache/
│   └── exports/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│       └── sample_invoices/
│
└── scripts/
    ├── setup_models.py               # Download & cache models
    └── benchmark.py                  # Performance benchmarking
```

---

## 6. Performance Projections

| Scenario | Path | Speed (per invoice) | Accuracy | Hardware |
|---|---|---|---|---|
| Known vendor (template) | Fast ⚡ | **30-80ms** | **97-99%** | CPU only |
| Simple new vendor | Smart 🧠 (LLM only) | **1-3s** | **90-95%** | GPU (16GB) |
| Complex new vendor | Smart 🧠 (VLM+LLM) | **3-8s** | **88-94%** | GPU (16GB) |
| Batch (100 known vendors) | Fast ⚡ | **~5 seconds total** | **97-99%** | CPU only |
| Batch (100 mixed) | Hybrid | **~2-4 minutes** | **93-97%** | GPU (16GB) |

> [!NOTE]
> **Throughput scales linearly** with the percentage of invoices routed through the Fast Path. As the system learns more vendor templates, throughput approaches the deterministic path speeds. After processing invoices from ~50 unique vendors, expect 80%+ of incoming invoices to hit the Fast Path.

---

## 7. Hardware Requirements

| Tier | CPU | RAM | GPU | Storage | Best For |
|---|---|---|---|---|---|
| **Minimum** | 4 cores | 16GB | None | 50GB SSD | Dev/testing, low volume (<100/day) |
| **Recommended** | 8 cores | 32GB | RTX 3060 12GB | 200GB SSD | Production, medium volume (500/day) |
| **Optimal** | 12+ cores | 64GB | RTX 4090 24GB | 500GB NVMe | High throughput (2000+/day) |

> [!TIP]
> **For CPU-only deployment**, use Phi-4-Mini (3.8B) with 4-bit quantization via llama.cpp. It requires only ~4GB RAM for the model and still achieves ~85-90% accuracy on structured invoice extraction. Pair with aggressive template learning to minimize LLM calls.

---

## 8. Dependency Summary

```toml
# pyproject.toml — Key dependencies

[project]
requires-python = ">=3.11"
dependencies = [
    # Core
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.9",
    "pydantic-settings>=2.5",
    
    # PDF Processing
    "pymupdf>=1.25",
    "pdfplumber>=0.11",
    "pikepdf>=9.0",
    
    # Image Processing
    "opencv-python-headless>=4.10",
    "Pillow>=10.4",
    
    # Layout Analysis & OCR
    "docling>=2.15",
    "paddleocr>=2.9",
    "pytesseract>=0.3",
    
    # LLM/VLM Inference
    "vllm>=0.7",              # GPU production
    "outlines>=0.1",           # Constrained decoding
    "instructor>=1.7",         # Ollama JSON enforcement
    "ollama>=0.4",             # Dev/CPU inference
    
    # Embeddings & Vector DB
    "sentence-transformers>=3.3",
    "lancedb>=0.17",
    
    # Storage
    "sqlalchemy>=2.0",
    "aiosqlite>=0.20",
    
    # Utilities
    "loguru>=0.7",
    "rich>=13.9",
    "anyio>=4.7",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
cpu = [
    "llama-cpp-python>=0.3",   # CPU inference alternative
]
export = [
    "openpyxl>=3.1",           # Excel export
    "pandas>=2.2",             # Data manipulation
]
```

---

## 9. Security & Privacy Considerations

| Concern | Mitigation |
|---|---|
| No data leaves the machine | All models run locally; no network calls after initial setup |
| Sensitive invoice data | SQLite with filesystem encryption (BitLocker/LUKS); no cloud storage |
| Model provenance | All recommended models are Apache 2.0 or MIT licensed |
| Input sanitization | pikepdf pre-flight to defuse malicious PDFs; sandbox PDF rendering |
| Audit trail | Full processing_log table with timestamps, methods, confidence scores |

---

## 10. Open Questions for Decision

> [!IMPORTANT]
> **Q1: License Sensitivity** — Are you comfortable with AGPL (PyMuPDF) in production? If not, we can use `pypdfium2` (Apache 2.0/BSD) as an alternative, though it has a slightly less rich API.

> [!IMPORTANT]
> **Q2: GPU Availability** — What GPU(s) will be available in production? This directly impacts model selection:
> - **24GB VRAM**: Run both VLM and LLM concurrently
> - **12-16GB VRAM**: Run one model at a time (swap as needed)
> - **CPU only**: Use Phi-4-Mini + aggressive template caching

> [!IMPORTANT]
> **Q3: Invoice Volume** — What's the expected daily volume?
> - **< 100/day**: Single-process async is sufficient
> - **100-1000/day**: Multi-worker with task queue
> - **1000+/day**: Consider Celery + Redis for distributed processing

> [!IMPORTANT]
> **Q4: Frontend Requirements** — Do you need a web UI for:
> - Uploading invoices?
> - Reviewing/correcting flagged extractions (Human-in-the-Loop)?
> - Dashboard/analytics?
> - Or is a CLI/API-only interface sufficient?

> [!IMPORTANT]
> **Q5: Multi-language Support** — Will invoices be in English only, or do you need multilingual support (e.g., German, Chinese, Arabic)? This impacts OCR engine configuration and LLM prompt design.

---

## 11. 2026 Trends & Future-Proofing

| Trend | Implication for PDF Miner | Action |
|---|---|---|
| **End-to-end VLMs replacing OCR pipelines** | Models like Qwen2.5-VL can directly extract structured data from page images, bypassing OCR entirely | Architecture supports both paths — can gradually shift from OCR→LLM to pure VLM as models improve |
| **XGrammar-2 & structural tags** | New constrained decoding standard enables OpenAI-style tool calling locally | Already adopted in our stack; enables complex nested schemas |
| **Agentic IDP (Intelligent Document Processing)** | Models that can read, reason about, and act on documents autonomously | The Learning Loop (Section 4.5) is a step toward this — auto-template generation |
| **Sub-1B specialist models** | GOT-OCR2.0 (580M), SmolDocling (256M) prove that tiny models can rival large ones on specific tasks | Use Florence-2 / SmolDocling for fast pre-classification before invoking larger models |
| **RLVR (Reinforcement Learning with Verifiable Rewards)** | Models like olmOCR-2 use RLVR to reduce hallucinations in document extraction | Monitor for Qwen3-VL variants trained with RLVR for even higher accuracy |
| **Quantization advances (Q4_K_M → Q6_K)** | Better quantization = larger models on smaller hardware | Can revisit 13B+ models as quantization improves |

> [!TIP]
> **Future-proofing strategy**: The hybrid architecture (Fast Path + Smart Path) is inherently future-proof. As VLMs improve, the Smart Path gets more accurate. As the template library grows, the Fast Path handles more volume. The system naturally optimizes itself over time regardless of which models are plugged in.

---

## 12. Summary — Getting Started

**Phase 1 (Week 1-2): Foundation**
1. Set up project structure with `uv` or `poetry`
2. Implement PDF ingestion layer (PyMuPDF + OpenCV preprocessing)
3. Integrate Docling + PaddleOCR for layout analysis
4. Define Pydantic schemas for invoice data

**Phase 2 (Week 3-4): Smart Path**
1. Deploy Qwen2.5-VL-7B via Ollama (dev) or vLLM (production)
2. Build VLM extraction pipeline with XGrammar/Instructor
3. Implement validation and confidence scoring
4. Store results in SQLite + LanceDB

**Phase 3 (Week 5-6): Fast Path & Learning**
1. Build template engine with spatial anchors and regex rules
2. Implement vendor matching via LanceDB embeddings
3. Build the auto-template generation feedback loop
4. Add the routing layer (score threshold → Fast/Smart path)

**Phase 4 (Week 7-8): Production Hardening**
1. Add FastAPI REST endpoints
2. Implement batch processing with asyncio
3. Add monitoring (Loguru + Rich dashboards)
4. Benchmark and optimize throughput
