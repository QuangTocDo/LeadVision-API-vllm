import os
from dotenv import load_dotenv
load_dotenv()

import json
import re
import time
import base64
import asyncio
import logging
import unicodedata
import uuid
import contextvars
from io import BytesIO
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, File, UploadFile, APIRouter, Depends, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from pydantic import BaseModel, model_validator

import uvicorn
from PIL import Image
import pillow_heif
import httpx
from openai import AsyncOpenAI
# pyrefly: ignore [missing-import]
from anyascii import anyascii


# Context variable to hold the request ID for the current request context
request_id_ctx_var = contextvars.ContextVar("request_id", default="")

# File size limit configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 768 * 768

# Custom formatter to safely inject request_id if it's missing on the record
class RequestIDFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "request_id"):
            req_id = request_id_ctx_var.get()
            record.request_id = f"ReqID: {req_id}" if req_id else "System"
        return super().format(record)

# Setup root handler with our formatter
handler = logging.StreamHandler()
handler.setFormatter(RequestIDFormatter(
    "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s"
))

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Clear any existing handlers to prevent duplicate logging
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)
root_logger.addHandler(handler)

logger = logging.getLogger("qwen_gateway")


# Register HEIF opener to support HEIC files in Pillow
pillow_heif.register_heif_opener()

app = FastAPI(
    title="Qwen3-VL Lead Retrieval API (vLLM Gateway)",
    description="API gateway to extract lead details from business cards or photos using a backend vLLM Qwen3-VL instance.",
    version="1.0.0"
)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Retrieve Request-ID from request headers, or generate a new UUID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        token = request_id_ctx_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_ctx_var.reset(token)

class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and (
            request.url.path.endswith("/extract-file") or
            request.url.path.endswith("/extract-double-file")
        ):
            content_length = request.headers.get("content-length")
            if content_length:
                # For double-file uploads, allow up to 2x the single file limit
                limit = MAX_FILE_SIZE_BYTES * 2 if request.url.path.endswith("/extract-double-file") else MAX_FILE_SIZE_BYTES
                if int(content_length) > limit:
                    # pyrefly: ignore [missing-import]
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB * 2 if request.url.path.endswith('/extract-double-file') else MAX_FILE_SIZE_MB}MB."}
                    )
        return await call_next(request)

app.add_middleware(LimitUploadSizeMiddleware)
app.add_middleware(RequestIDMiddleware)

router = APIRouter(prefix="/api/v1")



async def validate_and_get_file_bytes(file: UploadFile = File(...)) -> bytes:
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in [".png", ".jpg", ".jpeg", ".heic"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Supported formats are: png, jpg, jpeg, heic."
        )
    # Perform early content-size check on actual upload
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
        )
    return image_bytes


async def _validate_single_file(file: UploadFile, label: str) -> bytes:
    """Validate a single uploaded file. for double-image endpoints."""
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in [".png", ".jpg", ".jpeg", ".heic"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}' for {label}. Supported formats are: png, jpg, jpeg, heic."
        )
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{label} is too large. Maximum allowed size per image is {MAX_FILE_SIZE_MB}MB."
        )
    return image_bytes


# Connect to the backend vLLM server (default port 8002)
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8002/v1")
model_id = os.getenv("VLLM_MODEL_ID", "baidu/Qianfan-OCR")

openai_client = AsyncOpenAI(
    base_url=VLLM_API_URL,
    api_key="EMPTY", 
    timeout=20.0
)

# API Key security configuration
API_KEY = os.getenv("API_KEY", "your-super-secret-api-key")
api_key_header = APIKeyHeader(name="API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials. Invalid API Key."
        )


# Load country data from countries.json
COUNTRIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "countries.json")
country_data = {"aliases": {}, "codes": {}}
if os.path.exists(COUNTRIES_FILE):
    try:
        with open(COUNTRIES_FILE, "r", encoding="utf-8") as f:
            country_data = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load countries.json: {e}")

# Load prompt from prompt.txt
PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.txt")
prompt_text = ""
if os.path.exists(PROMPT_FILE):
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except Exception as e:
        logger.error(f"Could not load prompt.txt: {e}")

# Load double-image prompt from prompt_double.txt
PROMPT_DOUBLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_double.txt")
prompt_double_text = ""
if os.path.exists(PROMPT_DOUBLE_FILE):
    try:
        with open(PROMPT_DOUBLE_FILE, "r", encoding="utf-8") as f:
            prompt_double_text = f.read()
    except Exception as e:
        logger.error(f"Could not load prompt_double.txt: {e}")


class ExtractionRequest(BaseModel):
    image_path: str


class DoubleExtractionRequest(BaseModel):
    """Request model for front + back business card extraction."""
    front_image_path: str
    back_image_path: str


# ---------------------------------------------------------------------------
# Pinyin post-processing helpers
# ---------------------------------------------------------------------------

# Pinyin syllables that are diagnostic of Chinese company name transliterations.
# These are the romanized forms of characters commonly found in Chinese company
# names: 有限公司 (You Xian Gong Si), 科技 (Ke Ji), 集团 (Ji Tuan), etc.
_PINYIN_COMPANY_MARKERS: set[str] = {
    "Gong", "Si", "You", "Xian", "Gu", "Fen", "Ze", "Ren",
    "Ji", "Tuan", "Mao", "Yi", "Ke", "Zhi", "Chuang", "Huan",
    "Qiu", "Hua", "Xin", "Fa", "Zhan", "Ye", "Jing", "Ying",
    "Guan", "Zi", "Xun", "Shi", "Jie", "Dian", "Neng", "Lian",
    "He", "Shang", "Wu", "Pin", "Tai", "Luo", "Yun", "Shu",
    "Ju", "Wen", "Bo", "Chen", "Chu", "Quan", "Qian", "Zong",
    "Meng", "Lang", "Peng", "Bin", "Feng", "Guang", "Cheng",
}


def _is_spaced_pinyin(text: str, min_words: int = 4) -> bool:
    """
    Return True if *text* looks like a sequence of space-separated Chinese
    pinyin syllables (each syllable corresponding to one Chinese character).

    Heuristic rules (ALL must pass):
    - At least `min_words` space-separated tokens.
    - ≥80 % of tokens are short (≤5 chars — typical single-syllable pinyin).
    - At least 2 tokens match known Chinese company/name pinyin markers.
    """
    if not text:
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    short_words = [w for w in words if len(w.rstrip(".,")) <= 5]
    if len(short_words) / len(words) < 0.80:
        return False
    marker_hits = sum(1 for w in words if w in _PINYIN_COMPANY_MARKERS)
    return marker_hits >= 2


def _normalize_chinese_person_name(name: str) -> str:
    """
    Convert a space-separated Chinese pinyin person name to the standard
    international format where surname and given-name are kept as two tokens
    but all given-name syllables are merged into one word.

    Examples:
      'Wang Hao Xuan'  ->  'Wang Haoxuan'
      'Chen Jun Xuan'  ->  'Chen Junxuan'
      'Zhang Jing Yi'  ->  'Zhang Jingyi'
      'Li Wei'         ->  'Li Wei'   (2 words — unchanged)
    """
    words = name.split()
    if len(words) < 3:
        return name
    # Only treat as pinyin when every token is a short syllable (≤5 chars)
    if not all(len(w) <= 5 for w in words):
        return name
    surname = words[0]
    # First given-name syllable keeps its capitalisation; the rest are lowercased
    # so the result is e.g. "Haoxuan" not "HaoXuan".
    given = words[1] + "".join(w.lower() for w in words[2:])
    return f"{surname} {given}"


# ---------------------------------------------------------------------------

class BusinessCardData(BaseModel):
    full_name: str | None = None
    company: str | None = None
    email: str | None = None
    country: str | None = None
    phone_number: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_and_normalize(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data

        # Capture a copy of raw fields before normalization for formatting check
        raw_fields = dict(data)

        def strip_diacritics(text: str | None) -> str | None:
            if not text:
                return None
            s = str(text).replace('đ', 'd').replace('Đ', 'D')
            nfd = unicodedata.normalize('NFD', s)
            filtered = "".join([c for c in nfd if unicodedata.category(c) != 'Mn'])
            return unicodedata.normalize('NFC', filtered)

        def transliterate_text(text: str | None) -> str | None:
            if not text:
                return None
            
            def is_cjk_or_asian(c: str) -> bool:
                return (
                    '\u4e00' <= c <= '\u9fff' or     
                    '\uac00' <= c <= '\ud7a3' or     
                    '\u3040' <= c <= '\u309f' or     
                    '\u30a0' <= c <= '\u30ff'        
                )

            text_str = str(text)
            spaced_chars = []
            for i, char in enumerate(text_str):
                spaced_chars.append(char)
                if is_cjk_or_asian(char):
                    if i + 1 < len(text_str) and is_cjk_or_asian(text_str[i+1]):
                        spaced_chars.append(' ')
            
            processed_text = "".join(spaced_chars)
            latin_text = anyascii(processed_text)
            cleaned_latin = re.sub(r"\s+", " ", latin_text).strip()
            return strip_diacritics(cleaned_latin)

        # 1. Normalize Country
        country = data.get("country")
        normalized_country = None
        if country:
            c_lower = str(country).strip().lower()
            # Lookup in aliases from countries.json, fallback to title case
            normalized_country = country_data.get("aliases", {}).get(c_lower)
            if not normalized_country:
                normalized_country = str(country).strip().title()
            data["country"] = transliterate_text(normalized_country)

        # 2. Normalize Phone Number
        phone = data.get("phone_number")
        if phone:
            phone_str = str(phone).strip()
            has_plus = phone_str.startswith("+")
            cleaned = re.sub(r"\D", "", phone_str)
            
            if has_plus:
                pass
            else:
                code = country_data.get("codes", {}).get(normalized_country) if normalized_country else None
                if cleaned.startswith("0"):
                    if code:
                        cleaned = code + cleaned[1:]
                    else:
                        cleaned = "84" + cleaned[1:]  # Fallback to Vietnam
                elif len(cleaned) == 9 and cleaned.startswith("9"):
                    if normalized_country == "Vietnam" or normalized_country is None:
                        cleaned = "84" + cleaned
                elif len(cleaned) == 10:
                    if normalized_country == "United States":
                        cleaned = "1" + cleaned
                    elif normalized_country == "United Kingdom" and cleaned.startswith("7"):
                        cleaned = "44" + cleaned
            
            # Validate length
            if 7 <= len(cleaned) <= 15:
                data["phone_number"] = cleaned
            else:
                data["phone_number"] = None

        # 3. Normalize Email
        email = data.get("email")
        if email:
            email_str = str(email).strip().lower()
            email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
            if re.match(email_regex, email_str):
                data["email"] = email_str
            else:
                data["email"] = None

        # 4. Normalize Name
        name = data.get("full_name")
        if name:
            name_str = str(name).strip()
            name_str = re.sub(r"^(Mr\.|Ms\.|Mrs\.|Dr\.|Engr\.)\s+", "", name_str, flags=re.IGNORECASE)
            name_str = re.sub(r"\s+", " ", name_str)
            name_str = transliterate_text(name_str.title())

            # Chinese name fix: merge given-name syllables when in a China context.
            # e.g. "Chen Jun Xuan" → "Chen Junxuan"
            is_chinese_context = normalized_country and "china" in normalized_country.lower()
            if name_str and is_chinese_context:
                name_str = _normalize_chinese_person_name(name_str)

            data["full_name"] = name_str

        # 5. Normalize Company
        company = data.get("company", data.get("organization"))
        if company:
            comp_str = str(company).strip()
            comp_str = re.sub(r"\s+", " ", comp_str)
            comp_str = transliterate_text(comp_str)

            # Pinyin company guard: if the company string looks like a
            # space-separated Chinese pinyin transliteration (e.g.
            # "Huan Qiu Zhi Chuang Ke Ji You Xian Gong Si"), the model
            # failed to read the Latin company name from the card.
            # Return null — a pinyin syllable chain is not a valid
            # international company name.
            if comp_str and _is_spaced_pinyin(comp_str, min_words=4):
                logger.info(
                    f"Company '{comp_str}' detected as spaced-pinyin → null"
                )
                comp_str = None

            data["company"] = comp_str
            if "organization" in data:
                del data["organization"]

        # Check for invalid formats in full_name, company, and email
        is_invalid = False

        # 1. Check Email format (if present)
        email_val = raw_fields.get("email")
        if email_val:
            email_str = str(email_val).strip().lower()
            email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
            if not re.match(email_regex, email_str):
                is_invalid = True

        # 2. Check Full Name format (if present)
        name_val = raw_fields.get("full_name")
        if name_val:
            name_str = str(name_val).strip().lower()
            # If name contains '@', 'www.', 'http://', 'https://' or ends with common domains like .com, .vn, .net
            if ("@" in name_str or 
                "www." in name_str or 
                "http://" in name_str or 
                "https://" in name_str or
                any(name_str.endswith(ext) for ext in [".com", ".vn", ".net", ".org", ".co.jp"])):
                is_invalid = True

        # 3. Check Company format (if present)
        company_val = raw_fields.get("company")
        if company_val:
            company_str = str(company_val).strip().lower()
            if "@" in company_str:
                is_invalid = True

        if is_invalid:
            logger.warning(
                f"Format validation failed for raw fields: {raw_fields}. Resetting all fields to null."
            )
            data["full_name"] = None
            data["company"] = None
            data["email"] = None
            data["country"] = None
            data["phone_number"] = None

        return data


def process_local_image(resolved_path: str) -> str:
    img = Image.open(resolved_path)
    img = img.convert("RGB")
    MAX_SIZE = 768
    if max(img.size) > MAX_SIZE:
        img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_image_bytes(image_bytes: bytes) -> str:
    img = Image.open(BytesIO(image_bytes))
    img = img.convert("RGB")
    MAX_SIZE = 768
    if max(img.size) > MAX_SIZE:
        img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@router.get("/health")
async def health_check():
    try:
        await openai_client.with_options(timeout=2.0).models.list()
        return {"status": "healthy", "vllm_connected": True}
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(
            status_code=503,
            detail=f"vLLM backend unreachable: {str(e)}"
        )


async def _extract_business_card_from_base64(img_base64: str) -> BusinessCardData:
    if not prompt_text:
        raise HTTPException(status_code=500, detail="prompt.txt file is missing or empty inside container.")

    data_url = f"data:image/jpeg;base64,{img_base64}"
    
    MAX_RETRIES = 3
    retry_delay = 1.0
    response = None
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a business card OCR system. "
                            "CRITICAL RULE: When a business card contains BOTH native-script text "
                            "(Chinese, Japanese, Korean, etc.) AND Latin-alphabet text (A-Z), "
                            "you MUST read and return the Latin-alphabet version for full_name and company. "
                            "NEVER transliterate or romanize native-script text if a Latin version exists anywhere on the card. "
                            "If no Latin version exists for a field, return null for that field."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url}
                            },
                            {
                                "type": "text",
                                "text": prompt_text
                            }
                        ]
                    }
                ],
                extra_body={
                    "guided_json": BusinessCardData.model_json_schema()
                },
                max_tokens=512,
                top_p=0.8,
                temperature=0.0,
            )
            break
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"vLLM inference failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2.0
            else:
                logger.exception(f"vLLM inference failed permanently after {MAX_RETRIES} attempts.")
                
    if response is None:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to retrieve data from vLLM backend after multiple attempts. Error: {str(last_error)}"
        )
        
    output_text = response.choices[0].message.content.strip()

    # Clean up JSON formatting from the output if needed (robust search)
    clean_text = output_text
    json_match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
    if json_match:
        clean_text = json_match.group(1).strip()
    else:
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            card_data = BusinessCardData(**parsed)
            
            # Post-parsing check to filter out few-shot dummy data hallucinations (e.g. blank images)
            if (card_data.full_name in ["Kim Min Soo", "John Smith"] or 
                card_data.email in ["kim.minsoo@gmail.com", "john.smith@gmail.com"]):
                logger.warning("Few-shot dummy hallucination detected. Resetting all fields to null.")
                card_data.full_name = None
                card_data.company = None
                card_data.email = None
                card_data.country = None
                card_data.phone_number = None
                
            return card_data
        else:
            raise ValueError("Parsed JSON is not a dictionary.")
    except (json.JSONDecodeError, ValueError) as e:
        logger.exception(f"Failed to parse model output as JSON: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse model output as JSON: {str(e)}"
        )


@router.post("/extract", response_model=BusinessCardData, dependencies=[Depends(verify_api_key)])
async def extract_details(request: ExtractionRequest):
    logger.info(f"Received extract request for image: {request.image_path}")
    
    # 1. Image validation
    is_url = request.image_path.startswith(("http://", "https://"))
    resolved_path = request.image_path
    
    if not is_url:
        # Resolve relative "media/..." to "/media/..." if mounted inside the container
        if not os.path.exists(resolved_path) and resolved_path.startswith("media/"):
            alternative_path = "/" + resolved_path
            if os.path.exists(alternative_path):
                resolved_path = alternative_path
        
        # Check size of local file
        if os.path.exists(resolved_path):
            if os.path.getsize(resolved_path) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Local file is too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
                )

    try:
        # 2. Input preparation
        if is_url:
            async with httpx.AsyncClient(timeout=15.0) as client:
                try:
                    async with client.stream("GET", resolved_path) as response:
                        response.raise_for_status()
                        
                        # Validate Content-Type early
                        content_type = response.headers.get("Content-Type", "")
                        if content_type and not content_type.startswith("image/"):
                            raise HTTPException(
                                status_code=400,
                                detail=f"Unsupported downloaded content type '{content_type}'. Must be an image."
                            )
                        
                        # Validate Content-Length early
                        content_length = response.headers.get("Content-Length")
                        if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Downloaded image is too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
                            )
                        
                        image_bytes = await response.aread()
                        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Downloaded image is too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
                            )
                except Exception as e:
                    if isinstance(e, HTTPException):
                        raise e
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to fetch image from URL: {str(e)}"
                    )
            img_base64 = await asyncio.to_thread(process_image_bytes, image_bytes)
        else:
            img_base64 = await asyncio.to_thread(process_local_image, resolved_path)

        # 3. Model inference and post-processing
        card_data = await _extract_business_card_from_base64(img_base64)
        logger.info(f"Extraction successful. Name: {card_data.full_name}, Company: {card_data.company}")
        return card_data

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.exception("Extraction error occurred")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )



@router.post("/extract-file", response_model=BusinessCardData, dependencies=[Depends(verify_api_key)])
async def extract_details_file(
    image_bytes: bytes = Depends(validate_and_get_file_bytes)
):
    logger.info("Received extract-file request")
    try:
        # Process image bytes in a thread pool
        img_base64 = await asyncio.to_thread(process_image_bytes, image_bytes)
        
        # Model inference and post-processing
        card_data = await _extract_business_card_from_base64(img_base64)
        logger.info(f"File extraction successful. Name: {card_data.full_name}, Company: {card_data.company}")
        return card_data

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.exception("Extraction error occurred")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


async def _extract_business_card_double(front_base64: str, back_base64: str) -> BusinessCardData:
    """Send both front and back images in a single Qwen3-VL request and extract unified lead data."""
    if not prompt_double_text:
        raise HTTPException(
            status_code=500,
            detail="prompt_double.txt file is missing or empty inside container."
        )

    front_url = f"data:image/jpeg;base64,{front_base64}"
    back_url = f"data:image/jpeg;base64,{back_base64}"

    MAX_RETRIES = 3
    retry_delay = 1.0
    response = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = await openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a business card OCR system. "
                            "CRITICAL RULE: When a business card contains BOTH native-script text "
                            "(Chinese, Japanese, Korean, etc.) AND Latin-alphabet text (A-Z), "
                            "you MUST read and return the Latin-alphabet version for full_name and company. "
                            "NEVER transliterate or romanize native-script text if a Latin version exists anywhere on the card. "
                            "If no Latin version exists for a field, return null for that field."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            # Front image first
                            {
                                "type": "image_url",
                                "image_url": {"url": front_url}
                            },
                            # Back image second
                            {
                                "type": "image_url",
                                "image_url": {"url": back_url}
                            },
                            # Combined extraction prompt
                            {
                                "type": "text",
                                "text": prompt_double_text
                            }
                        ]
                    }
                ],
                extra_body={
                    "guided_json": BusinessCardData.model_json_schema()
                },
                max_tokens=512,
                temperature=0.0
            )
            break
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"vLLM double inference failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2.0
            else:
                logger.exception(f"vLLM double inference failed permanently after {MAX_RETRIES} attempts.")

    if response is None:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to retrieve data from vLLM backend after multiple attempts. Error: {str(last_error)}"
        )

    output_text = response.choices[0].message.content.strip()

    # Clean up JSON formatting from the output if needed (robust search)
    clean_text = output_text
    json_match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
    if json_match:
        clean_text = json_match.group(1).strip()
    else:
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            card_data = BusinessCardData(**parsed)

            # Post-parsing check to filter out few-shot dummy data hallucinations
            if (card_data.full_name in ["Kim Min Soo", "John Smith"] or
                    card_data.email in ["kim.minsoo@gmail.com", "john.smith@gmail.com"]):
                logger.warning("Few-shot dummy hallucination detected in double extraction. Resetting all fields to null.")
                card_data.full_name = None
                card_data.company = None
                card_data.email = None
                card_data.country = None
                card_data.phone_number = None

            return card_data
        else:
            raise ValueError("Parsed JSON is not a dictionary.")
    except (json.JSONDecodeError, ValueError) as e:
        logger.exception(f"Failed to parse double model output as JSON: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse model output as JSON: {str(e)}"
        )


async def _resolve_image_to_base64(image_path: str, label: str) -> str:
    """Resolve a URL or local path to a base64-encoded JPEG string."""
    is_url = image_path.startswith(("http://", "https://"))
    resolved_path = image_path

    if not is_url:
        if not os.path.exists(resolved_path) and resolved_path.startswith("media/"):
            alternative_path = "/" + resolved_path
            if os.path.exists(alternative_path):
                resolved_path = alternative_path

        if os.path.exists(resolved_path):
            if os.path.getsize(resolved_path) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{label} is too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
                )

    if is_url:
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                async with client.stream("GET", resolved_path) as resp:
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type", "")
                    if content_type and not content_type.startswith("image/"):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsupported content type '{content_type}' for {label}. Must be an image."
                        )
                    content_length = resp.headers.get("Content-Length")
                    if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{label} URL content is too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
                        )
                    image_bytes = await resp.aread()
                    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{label} is too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
                        )
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise e
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to fetch {label} from URL: {str(e)}"
                )
        return await asyncio.to_thread(process_image_bytes, image_bytes)
    else:
        return await asyncio.to_thread(process_local_image, resolved_path)


@router.post("/extract-double", response_model=BusinessCardData, dependencies=[Depends(verify_api_key)])
async def extract_double(request: DoubleExtractionRequest):
    """Extract lead information from front and back images of a business card (URL or local path)."""
    logger.info(
        f"Received extract-double request. Front: {request.front_image_path}, Back: {request.back_image_path}"
    )
    try:
        # Resolve both images concurrently
        front_base64, back_base64 = await asyncio.gather(
            _resolve_image_to_base64(request.front_image_path, "front_image_path"),
            _resolve_image_to_base64(request.back_image_path, "back_image_path"),
        )

        card_data = await _extract_business_card_double(front_base64, back_base64)
        logger.info(f"Double extraction successful. Name: {card_data.full_name}, Company: {card_data.company}")
        return card_data

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.exception("Double extraction error occurred")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-double-file", response_model=BusinessCardData, dependencies=[Depends(verify_api_key)])
async def extract_double_file(
    front_image: UploadFile = File(..., description="Front side of the business card"),
    back_image: UploadFile = File(..., description="Back side of the business card"),
):
    """Extract lead information from uploaded front and back images of a business card."""
    logger.info("Received extract-double-file request")
    try:
        # Validate and read both files concurrently
        front_bytes, back_bytes = await asyncio.gather(
            _validate_single_file(front_image, "front_image"),
            _validate_single_file(back_image, "back_image"),
        )

        # Process both images concurrently in thread pool
        front_base64, back_base64 = await asyncio.gather(
            asyncio.to_thread(process_image_bytes, front_bytes),
            asyncio.to_thread(process_image_bytes, back_bytes),
        )

        card_data = await _extract_business_card_double(front_base64, back_base64)
        logger.info(f"Double file extraction successful. Name: {card_data.full_name}, Company: {card_data.company}")
        return card_data

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.exception("Double file extraction error occurred")
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
