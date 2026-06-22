# Qwen3-VL Business Card Lead Retrieval API Gateway

A production-ready, Dockerized API gateway designed to extract and normalize contact information (leads) from single- or double-sided business card images. The service utilizes a backend vLLM engine hosting the **Baidu Qianfan-OCR** Vision-Language Model (VLM), fronted by a FastAPI gateway, Nginx reverse proxy, and optional Ngrok tunneling.

---

## 🏗️ System Architecture

The gateway is architecture-hardened to prevent slow-loris attacks, enforce payload limits, and streamline asynchronous inferencing on a GPU.

```mermaid
graph TD
    Client[HTTP Client / External App] -->|HTTPS Tunnel| Ngrok[Ngrok Container]
    Ngrok -->|HTTP Proxy| Nginx[Nginx Reverse Proxy]
    Nginx -->|Route forwarding & limit checks| FastAPI[FastAPI Gateway app_new.py]
    FastAPI -->|OpenAI SDK Async calls| vLLM[vLLM Inference Server]
    vLLM -->|Model Execution| GPU[NVIDIA GPU - baidu/Qianfan-OCR]
```

### Infrastructure Components
1. **vLLM Inference Server**: Hosts the `baidu/Qianfan-OCR` model, accelerated by NVIDIA GPUs. It exposes an OpenAI-compatible completion API.
2. **FastAPI Gateway (`app_new.py`)**: Implements strict input validation, image preprocessing (resizing and base64 encoding), asynchronous model querying, and extensive post-processing rules.
3. **Nginx Reverse Proxy**: Enforces a strict `10MB` payload size limit, disables proxy buffering for raw connections, and sets standard 60-second read/connect timeouts.
4. **Ngrok Tunneling**: Automatically exposes the local Nginx endpoint to a static public domain for external webhooks or mobile integrations.

---

## 📂 Project Directory Structure

```
qwen_docker_nginx/
├── app_new.py             # FastAPI entrypoint containing validation, OCR, & normalization logic
├── Dockerfile             # Python environment container builder
├── docker-compose.yml     # Orchestrates vllm, gateway, nginx, and ngrok services
├── nginx.conf             # Nginx reverse proxy configuration with payload size security bounds
├── countries.json         # ISO-like country names, aliases mapping, and calling codes
├── prompt.txt             # Structured prompt template for single-side card extraction
├── prompt_double.txt      # Structured prompt template for double-side card extraction
├── requirements.txt       # Python application package dependencies
├── .env                   # Local environment variables configuration (ignored by Git)
└── .gitignore             # Git ignored files configuration
```

---

## ✨ Features & Normalization Engine

To guarantee consistent output formatting suitable for CRM intake, `app_new.py` runs a multi-stage post-processing normalization pipeline:

*   **Multi-lingual Transliteration**: Non-Latin characters (including CJK and Vietnamese) are transliterated to ASCII using `anyascii` and diacritics are removed (e.g., converting `đ`/`Đ` to `d`/`D`).
*   **Latin Text Preference**: The VLM prompt instructs the model that when a card contains both native script (e.g. Chinese characters) and Latin letters, the Latin version **must** be preferred.
*   **Chinese Name Standardization**: Multi-syllable spaced Pinyin names are automatically merged into single strings (e.g., `Wang Hao Xuan` ➔ `Wang Haoxuan`).
*   **Pinyin Company Guard**: Heuristic check to nullify company names that fail to translate and get output as space-separated Pinyin syllables.
*   **Phone Number Standardization**: Strips non-digits, resolves leading zeroes using `countries.json` calling codes, maps missing prefixes to a default (e.g., `+84` for Vietnam), and verifies that the total length resides between 7 and 15 digits.
*   **Email Validation**: Rejects invalid structural email strings using regex.
*   **Hallucination Filters**: Detects and voids dummy values resulting from few-shot prompting leaks (e.g., resetting `Kim Min Soo` or `John Smith` entries to `null` on empty cards).
*   **Local Image Preprocessing**: Downscales large images to a maximum dimension of `768px` before converting them to base64, reducing network transmission overhead to the vLLM backend.

---

## 🚀 Quick Start

### 📋 Prerequisites
*   An NVIDIA GPU with CUDA compatibility.
*   NVIDIA Container Toolkit installed on the host.
*   Docker & Docker Compose.
*   An active Ngrok Authtoken.

### ⚙️ Environment Variables Setup
Create a `.env` file in the root of the `qwen_docker_nginx/` directory with the following structure:

```env
# Tunneling configurations
NGROK_AUTHTOKEN=your-ngrok-authtoken-here

# Gateway authorization token
API_KEY=your-highly-secure-api-key-here

# Image size restrictions
MAX_FILE_SIZE_MB=10
```

### 🚢 Deployment
Run the following command to build and launch all containers in detached mode:

```bash
docker compose up -d --build
```

Monitor the startup logs of the gateway and backend models:
```bash
docker compose logs -f gateway
docker compose logs -f vllm
```

---

## 🔌 API Reference

All requests must include the authorization header:
`API-Key: <your_configured_api_key>`

### Output Schema (`BusinessCardData`)
Every extraction endpoint returns the following standard JSON payload structure:
```json
{
  "full_name": "string" or null,
  "company": "string" or null,
  "email": "string" or null,
  "country": "string" or null,
  "phone_number": "string" or null
}
```

---

### Endpoints

#### 1. System Health Check
*   **Endpoint**: `GET /api/v1/health`
*   **Headers**: None
*   **Description**: Validates connection and model listing status with the vLLM engine.
*   **Response**:
    ```json
    {
      "status": "healthy",
      "vllm_connected": true
    }
    ```

#### 2. Extract Lead from Path or URL (Single Card)
*   **Endpoint**: `POST /api/v1/extract`
*   **Headers**: `API-Key: <key>`
*   **Payload**:
    ```json
    {
      "image_path": "https://example.com/card.jpg"
    }
    ```
    *(Note: `image_path` can also be a local file path inside the gateway container, e.g., `/media/uploads/card.jpg`)*

#### 3. Extract Lead from File Upload (Single Card)
*   **Endpoint**: `POST /api/v1/extract-file`
*   **Headers**: `API-Key: <key>`
*   **Request Type**: `multipart/form-data`
*   **Form Param**: `file` (binary payload, supports `.png`, `.jpg`, `.jpeg`, `.heic`)

#### 4. Extract Lead from Path or URL (Double-sided Card)
*   **Endpoint**: `POST /api/v1/extract-double`
*   **Headers**: `API-Key: <key>`
*   **Payload**:
    ```json
    {
      "front_image_path": "https://example.com/card_front.jpg",
      "back_image_path": "https://example.com/card_back.jpg"
    }
    ```

#### 5. Extract Lead from File Upload (Double-sided Card)
*   **Endpoint**: `POST /api/v1/extract-double-file`
*   **Headers**: `API-Key: <key>`
*   **Request Type**: `multipart/form-data`
*   **Form Params**:
    *   `front_image` (binary payload)
    *   `back_image` (binary payload)

---

## 🐍 Client Integration Examples (Python)

Below are usage examples illustrating how to query the gateway endpoints from an external application.

### 1. Processing Single Cards

```python
import os
import requests

api_url = "https://steadier-grit-seldom.ngrok-free.dev"  # Replace with your URL
api_key = os.getenv("API_KEY", "your-secure-api-key")
headers = {"API-Key": api_key}

# --- Option A: JSON Request using image URL ---
payload = {"image_path": "https://example.com/card_image.jpg"}
response = requests.post(f"{api_url}/api/v1/extract", json=payload, headers=headers)
print("URL Extract:", response.json())

# --- Option B: Multipart Upload using local file ---
with open("local_card.png", "rb") as f:
    files = {"file": ("local_card.png", f, "image/png")}
    response = requests.post(f"{api_url}/api/v1/extract-file", files=files, headers=headers)
print("File Upload Extract:", response.json())
```

### 2. Processing Double-Sided Cards

```python
import os
import requests

api_url = "https://steadier-grit-seldom.ngrok-free.dev"
api_key = os.getenv("API_KEY", "your-secure-api-key")
headers = {"API-Key": api_key}

# --- Option A: JSON Request using front/back URLs ---
payload = {
    "front_image_path": "https://example.com/front.jpg",
    "back_image_path": "https://example.com/back.jpg"
}
response = requests.post(f"{api_url}/api/v1/extract-double", json=payload, headers=headers)
print("Double URL Extract:", response.json())

# --- Option B: Multipart Upload using local front/back files ---
with open("front.heic", "rb") as front_file, open("back.png", "rb") as back_file:
    files = {
        "front_image": ("front.heic", front_file, "image/heic"),
        "back_image": ("back.png", back_file, "image/png")
    }
    response = requests.post(f"{api_url}/api/v1/extract-double-file", files=files, headers=headers)
print("Double File Upload Extract:", response.json())
```

---

## 🛠️ Logging and Request Correlation

Every request traversing the gateway is assigned a unique `X-Request-ID` header (either parsed from incoming headers or automatically generated as a UUID). All log lines generated during that request include this ID, enabling seamless tracking and debugging across synchronous requests.

Example log output:
```log
2026-06-22 17:05:12,123 [INFO] [ReqID: 9a785dfa-b12e-4ab8-8422-92ab8d960012] qwen_gateway: Received extract-file request
2026-06-22 17:05:14,456 [INFO] [ReqID: 9a785dfa-b12e-4ab8-8422-92ab8d960012] qwen_gateway: File extraction successful. Name: Nguyen Van A, Company: FPT Software
```
