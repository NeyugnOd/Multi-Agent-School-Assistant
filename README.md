# Trợ Lý Tài Liệu cho Trường Học Multi Agent — Multi-Agent School Assistant

Trợ lý học đường sử dụng kiến trúc đa agent kết hợp RAG, chạy hoàn toàn trên máy local. Được phát triển dựa trên kiến trúc của [Akshay Pachaar](https://lightning.ai) và điều chỉnh lại cho lĩnh vực giáo dục — lưu trữ vector bằng Milvus trên Docker, suy luận bằng Ollama + Gemma 3 1B nhẹ và nhanh.

*A RAG-powered school assistant using multi-agent architecture with local LLM inference. Adapted for education use cases with Milvus on Docker and Ollama + Gemma 3 1B.*

---

## Tính Năng | Features

- **Trả lời có trích dẫn nguồn** — câu trả lời luôn kèm theo tài liệu tham chiếu
  *Citations-first retrieval — answers always backed by source documents*
- **Đa agent chuyên biệt** — mỗi agent đảm nhận một vai trò riêng: truy xuất, soạn thảo, đánh giá, tổng hợp
  *Multi-agent orchestration — specialized agents for retrieval, drafting, evaluation, and synthesis*
- **Tự động tìm web khi cần** — fallback sang Firecrawl nếu tài liệu nội bộ chưa đủ
  *Web fallback — automatically searches the web when internal knowledge is insufficient*
- **Binary Quantization (BQ)** — tìm kiếm vector độ trễ thấp (<30ms trên 50M+ vectors)
  *Low-latency vector search across large document sets*
- **Chạy hoàn toàn local** — không cần API LLM bên ngoài, bảo mật dữ liệu
  *Fully local — runs on your machine via Ollama, no external LLM API needed*

---

## Kiến Trúc | Architecture

![Architecture Diagram](./architecture.svg)

---

## Công Nghệ Sử Dụng | Tech Stack

| Thành phần | Công cụ |
|---|---|
| LLM (local) | [Ollama](https://ollama.com) + `gemma3:1b` |
| Vector Database | [Milvus](https://milvus.io) (Docker) + Binary Quantization |
| Điều phối Agent | [CrewAI](https://crewai.com) |
| Tìm kiếm Web | [Firecrawl](https://firecrawl.dev) |
| Embedding | `BAAI/bge-base-en-v1.5` |

---

## 🚀 Hướng Dẫn Chạy | Quick Start

### 1. Clone repo

```bash
git clone https://github.com/NeyugnOd/Multi-Agent-School-Assistant.git
cd Multi-Agent-School-Assistant
```

### 2. Cài đặt biến môi trường | Setup environment

```bash
cp .env.example .env
# Điền FIRECRAWL_API_KEY vào file .env
# Fill in your FIRECRAWL_API_KEY in .env
```

### 3. Cài thư viện | Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Khởi động Milvus | Start Milvus (vector database)

```bash
docker compose up -d
```

### 5. Tải model LLM | Pull the LLM model

```bash
ollama pull gemma3:1b
```

### 6. Chạy ứng dụng | Run the assistant

```bash
python app_new.py
```

---

## Cấu Hình | Configuration

Toàn bộ cài đặt nằm trong `config/settings.py`, có thể override qua `.env`:

*All settings are in `config/settings.py` and can be overridden via `.env`:*

| Biến | Mặc định | Mô tả |
|---|---|---|
| `FIRECRAWL_API_KEY` | — | Bắt buộc để dùng tìm kiếm web |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL Ollama server |
| `LLM_MODEL` | `gemma3:1b` | Model LLM local |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Model embedding HuggingFace |
| `TOP_K` | `3` | Số chunk truy xuất mỗi lần |

---

## Cấu Trúc Thư Mục | Project Structure

```
├── app_new.py              # Điểm khởi chạy chính | Main entry point
├── src/                    # Source code
│   ├── agents/             # Định nghĩa các agent
│   ├── rag/                # RAG pipeline, chunking, ingestion
│   └── utils/              # Helpers
├── config/
│   └── settings.py         # Cấu hình Pydantic
├── docker-compose.yml      # Milvus stack (etcd + MinIO + Milvus)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Yêu Cầu | Requirements

- Python 3.10+
- Docker & Docker Compose
- [Ollama](https://ollama.com) cài trên máy local
- Firecrawl API key (có free tier tại [firecrawl.dev](https://firecrawl.dev))

---

## Credits

Project này được phát triển dựa trên kiến trúc **Multi-Agent Legal Assistant** của [Akshay Pachaar](https://lightning.ai) — một hệ thống RAG ban đầu xây dựng cho tra cứu văn bản pháp luật. Mình đã điều chỉnh lại toàn bộ cho lĩnh vực giáo dục: thay thế knowledge base pháp luật bằng tài liệu học đường, dùng Gemma 3 1B qua Ollama cho nhẹ hơn, và chạy Milvus trên Docker thay vì cloud.

*This project is adapted from Akshay Pachaar's Multi-Agent Legal Assistant on Lightning AI Studio. The architecture was repurposed for school/education use cases with a lighter local LLM (Gemma 3 1B via Ollama) and self-hosted Milvus on Docker.*
