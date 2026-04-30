from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
import ipaddress
from collections import OrderedDict, defaultdict, deque
import logging
import os
import pickle
import socket
import re
import nltk
import requests
import threading
import time
from nltk.corpus import stopwords
import traceback
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

# ========================
# App Config
# ========================
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# Allow configurable frontend origins via env.
# Example: FRONTEND_ORIGINS=http://127.0.0.1:5500,https://yourdomain.com
frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://127.0.0.1:5500").split(",")
    if origin.strip()
]
CORS(app, resources={r"/analyze": {"origins": frontend_origins}})

# Limit request size (prevent abuse)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB

FACT_CHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
FACT_CHECK_TIMEOUT_SECONDS = 6
URL_FETCH_TIMEOUT_SECONDS = 5
URL_FETCH_MAX_BYTES = 1_000_000
ANALYZE_CACHE_TTL_SECONDS = 300
ANALYZE_CACHE_MAX_ENTRIES = 256
FACT_CHECK_CACHE_TTL_SECONDS = 900
FACT_CHECK_CACHE_MAX_ENTRIES = 256
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 30

_cache_lock = threading.Lock()
_analysis_cache = OrderedDict()
_fact_check_cache = OrderedDict()
_request_windows = defaultdict(deque)

# ========================
# Load Resources
# ========================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

try:
    with open("backend/model/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("backend/model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Fix for older sklearn models
    if model is not None and not hasattr(model, "multi_class"):
        model.multi_class = "auto"

except Exception as e:
    print("❌ Error loading model:", e)
    model = None
    vectorizer = None


# ========================
# Utils
# ========================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    words = text.split()
    words = [w for w in words if len(w) > 2 and w not in stop_words]
    return " ".join(words)


def is_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def make_cache_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def _cache_get(cache, key):
    now = time.monotonic()

    with _cache_lock:
        entry = cache.get(key)
        if entry is None:
            return False, None

        value, expires_at = entry
        if expires_at <= now:
            del cache[key]
            return False, None

        cache.move_to_end(key)
        return True, value


def _cache_set(cache, key, value, ttl_seconds: int, max_entries: int):
    expires_at = time.monotonic() + ttl_seconds

    with _cache_lock:
        cache[key] = (value, expires_at)
        cache.move_to_end(key)

        while len(cache) > max_entries:
            cache.popitem(last=False)


def get_client_identifier() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    return forwarded_for or request.remote_addr or "unknown"


def is_rate_limited() -> tuple[bool, int]:
    now = time.monotonic()
    client_id = get_client_identifier()

    with _cache_lock:
        window = _request_windows[client_id]

        while window and (now - window[0]) > RATE_LIMIT_WINDOW_SECONDS:
            window.popleft()

        if len(window) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - window[0])))
            return True, retry_after

        window.append(now)
        return False, 0


def is_public_hostname(hostname: str) -> bool:
    if not hostname:
        return False

    try:
        for _, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
            address = sockaddr[0]
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                return False

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                return False
    except socket.gaierror as exc:
        raise ValueError("The provided URL could not be resolved.") from exc

    return True


def extract_text_from_url(url: str) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if not is_public_hostname(hostname):
        raise ValueError("Only public news URLs are allowed.")

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=URL_FETCH_TIMEOUT_SECONDS) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit() and int(content_length) > URL_FETCH_MAX_BYTES:
                raise ValueError("The article page is too large to process safely.")

            raw_html = response.read(URL_FETCH_MAX_BYTES + 1)
            if len(raw_html) > URL_FETCH_MAX_BYTES:
                raise ValueError("The article page is too large to process safely.")

            html = raw_html.decode(response.headers.get_content_charset() or "utf-8", errors="ignore")
    except (HTTPError, URLError, ValueError) as exc:
        raise ValueError("Unable to fetch article from the provided URL. Try a different news link or paste the article text directly.") from exc

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    paragraphs = [
        paragraph.get_text(" ", strip=True)
        for paragraph in soup.find_all("p")
    ]

    if not paragraphs:
        paragraphs = [soup.get_text(" ", strip=True)]

    text_only = " ".join(paragraphs)
    text_only = unescape(re.sub(r"\s+", " ", text_only).strip())

    combined = f"{title} {text_only}".strip()
    return combined[:15000]


def generate_reasons(text: str, confidence: float = 0.0):
    reasons = []
    t = text.lower()

    if any(marker in t for marker in ["breaking", "shocking", "miracle", "secret", "urgent", "exposed", "cure"]):
        reasons.append("Contains sensational language")

    if not any(marker in t for marker in ["according to", "reported by", "reuters", "ap news", "bbc", "official", "minister", "agency"]):
        reasons.append("No credible sources found")

    if any(marker in t for marker in ["hoax", "deep state", "fake news", "conspiracy", "miracle cure"]):
        reasons.append("Matches known misinformation patterns")

    if len(t.split()) < 5:
        reasons.append("Very short content (low reliability)")

    if confidence < 55:
        reasons.append("Low confidence prediction")

    if not reasons:
        reasons.append("Matches known linguistic patterns")

    return reasons


def normalize_fact_check_verdict(raw_rating: str) -> str:
    rating = (raw_rating or "").strip().lower()
    if not rating:
        return "Unknown"

    false_markers = ["false", "mostly false", "pants on fire", "incorrect", "misleading"]
    true_markers = ["true", "mostly true", "correct", "accurate"]

    if any(marker in rating for marker in false_markers):
        return "False"
    if any(marker in rating for marker in true_markers):
        return "True"
    return raw_rating.strip()


def fetch_fact_check(query: str):
    cache_key = make_cache_key(query.strip().lower())
    found, cached_value = _cache_get(_fact_check_cache, cache_key)
    if found:
        return cached_value

    api_key = os.getenv("FACT_CHECK_API_KEY")
    if not api_key:
        return None

    params = {
        "query": query,
        "key": api_key,
        "languageCode": "en-US"
    }

    try:
        response = requests.get(FACT_CHECK_ENDPOINT, params=params, timeout=FACT_CHECK_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout:
        app.logger.warning("Fact check request timed out")
        _cache_set(_fact_check_cache, cache_key, None, 30, FACT_CHECK_CACHE_MAX_ENTRIES)
        return None
    except requests.RequestException:
        app.logger.warning("Fact check request failed")
        _cache_set(_fact_check_cache, cache_key, None, 30, FACT_CHECK_CACHE_MAX_ENTRIES)
        return None
    except ValueError:
        app.logger.warning("Fact check response was not valid JSON")
        _cache_set(_fact_check_cache, cache_key, None, 30, FACT_CHECK_CACHE_MAX_ENTRIES)
        return None

    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        result = {
            "status": "No verified claim found",
            "verdict": "Unknown",
            "source": "",
            "claim": ""
        }
        _cache_set(_fact_check_cache, cache_key, result, FACT_CHECK_CACHE_TTL_SECONDS, FACT_CHECK_CACHE_MAX_ENTRIES)
        return result

    for item in claims:
        if not isinstance(item, dict):
            continue

        claim_text = str(item.get("text") or "").strip()
        reviews = item.get("claimReview")

        if not isinstance(reviews, list) or not reviews:
            continue

        for review in reviews:
            if not isinstance(review, dict):
                continue

            raw_rating = str(review.get("textualRating") or "").strip()
            publisher = review.get("publisher") if isinstance(review.get("publisher"), dict) else {}
            publisher_name = str(publisher.get("name") or "Unknown publisher").strip()

            normalized = normalize_fact_check_verdict(raw_rating)

            result = {
                "status": "Verified claim found",
                "verdict": normalized,
                "source": publisher_name,
                "claim": claim_text or "Claim text missing"
            }
            _cache_set(_fact_check_cache, cache_key, result, FACT_CHECK_CACHE_TTL_SECONDS, FACT_CHECK_CACHE_MAX_ENTRIES)
            return result

    result = {
        "status": "No verified claim found",
        "verdict": "Unknown",
        "source": "",
        "claim": ""
    }
    _cache_set(_fact_check_cache, cache_key, result, FACT_CHECK_CACHE_TTL_SECONDS, FACT_CHECK_CACHE_MAX_ENTRIES)
    return result


# ========================
# Routes
# ========================
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    # Check model loaded
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 500

    limited, retry_after = is_rate_limited()
    if limited:
        response = jsonify({"error": "Too many requests. Please try again shortly."})
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    # Validate request type
    if not request.is_json:
        return jsonify({"error": "Invalid request format (JSON required)"}), 400

    data = request.get_json(silent=True) or {}

    # Validate input
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Valid text input is required"}), 400

    if len(text) > 5000:
        return jsonify({"error": "Text too long (max 5000 characters)"}), 400

    cache_key = make_cache_key(text.strip())
    found, cached_result = _cache_get(_analysis_cache, cache_key)
    if found:
        app.logger.debug("Analyze cache hit")
        return jsonify(cached_result), 200

    try:
        if is_url(text):
            text = extract_text_from_url(text)

        app.logger.debug("Analyze input: %s", text[:200])

        # Process
        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        confidence = max(probability) * 100

        app.logger.debug("ML prediction: %s | confidence: %.2f", prediction, confidence)

        # Generate reasons once
        reasons = generate_reasons(text, confidence)

        result = "Real" if prediction == 1 else "Fake"

        if confidence < 55:
            if prediction == 1:
                result = "Real"
            else:
                result = "Uncertain"
                if "Low confidence prediction" not in reasons:
                    reasons.append("Low confidence prediction")

        fact_check = fetch_fact_check(text) if len(text.split()) > 5 else None

        if fact_check is not None:
            app.logger.debug("Fact-check result: %s", fact_check)

            fact_verdict = str(fact_check.get("verdict", "Unknown")).lower()
            api_has_verification = fact_check.get("status") == "Verified claim found"

            if api_has_verification:
                if result in {"Fake", "Uncertain"} and "false" in fact_verdict:
                    result = "Strong Fake"
                    reasons.append("Google Fact Check also rated related claim as False")
                elif result == "Real" and "true" in fact_verdict:
                    result = "Strong Real"
                    reasons.append("Google Fact Check confirms related claim as True")
                elif ((result in {"Real", "Strong Real"} and "false" in fact_verdict) or
                      (result in {"Fake", "Strong Fake", "Uncertain"} and "true" in fact_verdict)):
                    result = "Needs Verification"
                    reasons.append("ML prediction conflicts with fact-check verdict")
        else:
            app.logger.debug("Fact-check result: null")

        result_payload = {
            "result": result,
            "confidence": round(confidence, 2),
            "reasons": reasons,
            "fact_check": fact_check
        }

        _cache_set(_analysis_cache, cache_key, result_payload, ANALYZE_CACHE_TTL_SECONDS, ANALYZE_CACHE_MAX_ENTRIES)

        return jsonify(result_payload), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {exc}"}), 500


# ========================
# Error Handlers
# ========================
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Request too large"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ========================
# Run
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
