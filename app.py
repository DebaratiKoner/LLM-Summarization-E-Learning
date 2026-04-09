from flask import Flask, flash, redirect, render_template, request, session, url_for
import base64
import io
import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from youtube_transcript_api import YouTubeTranscriptApi

try:
    import docx
except ImportError:
    docx = None

try:
    import graphviz
except ImportError:
    graphviz = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "models/gemini-2.5-flash-lite"
MAX_MODEL_INPUT_CHARS = 12000
MAX_PREDICTION_INPUT_CHARS = 4000
FAST_LOCAL_SUMMARY = os.getenv("FAST_LOCAL_SUMMARY", "1") == "1"
ENABLE_PREDICTION = os.getenv("ENABLE_PREDICTION", "0") == "1"
FAST_VISUAL_MODE = os.getenv("FAST_VISUAL_MODE", "1") == "1"
FAST_LOCAL_QUIZ = os.getenv("FAST_LOCAL_QUIZ", "1") == "1"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
HF_VIDEO_MODEL = os.getenv("HF_VIDEO_MODEL", "genmo/mochi-1-preview")
PALETTE = {
    "deep": "#5E0006",
    "crimson": "#9B0F06",
    "orange": "#D53E0F",
    "sand": "#EED9B9",
    "paper": "#FBF5EA",
    "ink": "#2B1814",
    "line": "#C9A47A",
}


@lru_cache(maxsize=16)
def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(["arialbd.ttf", "segoeuib.ttf", "calibrib.ttf"])
    candidates.extend(["arial.ttf", "segoeui.ttf", "calibri.ttf"])

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


MODEL_AI = None


def get_model_ai():
    global MODEL_AI
    if MODEL_AI is None:
        MODEL_AI = genai.GenerativeModel(MODEL_NAME)
    return MODEL_AI


def get_prediction_model():
    if joblib is None:
        return None
    try:
        return joblib.load("model.pkl")
    except Exception:
        return None


prediction_model = get_prediction_model()
SERVER_OUTPUTS = {}


def ensure_session_defaults():
    session.setdefault("history", [])
    session.setdefault("chat_messages", [])
    session.setdefault("contact_messages", [])
    session.setdefault("total_points", 0)
    session.setdefault("badges", [])
    session.setdefault("open_chat_widget", False)
    session.setdefault("current_output_id", "")


BADGE_RULES = [
    {"name": "First Spark", "points": 10, "description": "Completed your first learning action."},
    {"name": "Summary Starter", "points": 25, "description": "Built momentum through repeated study sessions."},
    {"name": "Quiz Climber", "points": 60, "description": "Earned strong quiz progress."},
    {"name": "Knowledge Builder", "points": 120, "description": "Sustained learning across multiple tasks."},
    {"name": "Adaptive Scholar", "points": 220, "description": "Reached an advanced adaptive learning milestone."},
]


def award_points(points):
    ensure_session_defaults()
    total_points = session.get("total_points", 0) + max(0, int(points))
    session["total_points"] = total_points

    current_badges = session.get("badges", [])
    current_names = {badge["name"] for badge in current_badges}
    for badge in BADGE_RULES:
        if total_points >= badge["points"] and badge["name"] not in current_names:
            current_badges.append(badge)
    session["badges"] = current_badges


def save_server_output(payload):
    output_id = str(uuid.uuid4())
    SERVER_OUTPUTS[output_id] = payload
    session["current_output_id"] = output_id
    return output_id


def get_server_output():
    output_id = session.get("current_output_id", "")
    if not output_id:
        return {}
    return SERVER_OUTPUTS.get(output_id, {})


def get_next_badge(points):
    for badge in BADGE_RULES:
        if points < badge["points"]:
            return {
                "name": badge["name"],
                "points_needed": badge["points"] - points,
                "target": badge["points"],
            }
    return None


def build_badge_status(points, earned_badges):
    earned_names = {badge["name"] for badge in earned_badges}
    status = []
    for badge in BADGE_RULES:
        status.append(
            {
                "name": badge["name"],
                "points": badge["points"],
                "description": badge["description"],
                "earned": badge["name"] in earned_names,
            }
        )
    return status


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def clean_media_prompt_text(text):
    cleaned = clean_text(text)
    cleaned = re.sub(r"\b(title text|body text|website navigation|hero section|button text|menu bar)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    return clean_text(cleaned)


def prepare_model_text(text, limit=MAX_MODEL_INPUT_CHARS):
    cleaned = clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + " ..."


def unique_preserve(items):
    seen = set()
    result = []
    for item in items:
        cleaned = clean_text(item)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def dedupe_repeated_clause(text):
    cleaned = clean_text(text)
    if " - " not in cleaned:
        return cleaned
    parts = [clean_text(part) for part in cleaned.split(" - ") if clean_text(part)]
    if len(parts) < 2:
        return cleaned
    first = parts[0].lower()
    second = parts[1].lower()
    if first.startswith(second) or second.startswith(first):
        return parts[0] if len(parts[0]) >= len(parts[1]) else parts[1]
    return cleaned


def looks_like_meaningful_point(text):
    cleaned = clean_text(text)
    lowered = cleaned.lower()
    if len(cleaned.split()) < 5:
        return False
    if any(token in lowered for token in ["concept 1", "concept 2", "concept 3", "concept 4", "topic overview"]):
        return False
    return True


def is_near_duplicate(candidate, existing):
    left = clean_text(candidate).lower()
    right = clean_text(existing).lower()
    return left == right or left in right or right in left


def sanitize_learning_text(text):
    raw_lines = [dedupe_repeated_clause(line) for line in (text or "").replace("\r", "\n").split("\n")]
    sentence_pool = []
    for line in raw_lines:
        sentence_pool.extend(re.split(r"(?<=[.!?])\s+", line))

    cleaned_items = []
    for item in raw_lines + sentence_pool:
        cleaned = dedupe_repeated_clause(item)
        if not looks_like_meaningful_point(cleaned):
            continue
        if any(is_near_duplicate(cleaned, existing) for existing in cleaned_items):
            continue
        cleaned_items.append(cleaned)

    return "\n".join(cleaned_items[:8])


def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?&/]+)",
        r"shorts/([^?&/]+)",
        r"embed/([^?&/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url or "")
        if match:
            return match.group(1)
    stripped = (url or "").strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", stripped):
        return stripped
    return None


def transcript_items_to_text(items):
    parts = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = getattr(item, "text", "")
        if text:
            parts.append(text)
    return clean_text(" ".join(parts))


@lru_cache(maxsize=32)
def fetch_video_transcript_text(video_id):
    errors = []

    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["en", "en-US", "en-GB", "hi"],
        )
        text = transcript_items_to_text(transcript)
        if text:
            return text, ""
    except Exception as exc:
        errors.append(f"get_transcript: {exc}")

    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB", "hi"])
        text = transcript_items_to_text(fetched)
        if text:
            return text, ""
    except Exception as exc:
        errors.append(f"fetch: {exc}")

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        for language in ["en", "en-US", "en-GB", "hi"]:
            try:
                transcript = transcript_list.find_transcript([language])
                fetched = transcript.fetch()
                text = transcript_items_to_text(fetched)
                if text:
                    return text, ""
            except Exception:
                continue
    except Exception as exc:
        errors.append(f"list: {exc}")

    return "", " | ".join(errors)


def extract_pdf_text(pdf_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    extracted_pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_pages.append(page_text)
    return "\n".join(extracted_pages)


def extract_docx_text(doc_bytes):
    if docx is None:
        raise RuntimeError("DOCX support is unavailable because python-docx is not installed.")
    document = docx.Document(io.BytesIO(doc_bytes))
    return "\n".join(para.text for para in document.paragraphs if para.text.strip())


def transcribe_audio_bytes(audio_bytes, file_extension):
    if sr is None:
        raise RuntimeError("SpeechRecognition is not installed.")

    recognizer = sr.Recognizer()
    suffix = file_extension if file_extension in [".wav", ".mp3"] else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    wav_path = None

    try:
        source_path = temp_path
        if suffix == ".mp3":
            import soundfile as sf

            conversion_errors = []
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                wav_path = temp_wav.name

            try:
                audio_data, sample_rate = sf.read(temp_path)
                sf.write(wav_path, audio_data, sample_rate)
                source_path = wav_path
            except Exception as exc:
                conversion_errors.append(f"soundfile: {exc}")

            if source_path == temp_path:
                try:
                    from moviepy import AudioFileClip

                    clip = AudioFileClip(temp_path)
                    audio_data = clip.to_soundarray(fps=16000)
                    clip.close()
                    sf.write(wav_path, audio_data, 16000)
                    source_path = wav_path
                except Exception as exc:
                    conversion_errors.append(f"moviepy: {exc}")

            if source_path == temp_path:
                raise RuntimeError("MP3 conversion failed. " + " | ".join(conversion_errors))

        with sr.AudioFile(source_path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
    except Exception as exc:
        raise RuntimeError(f"Audio transcription failed: {exc}") from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def analyze_image_topic(image_bytes):
    prompt = """
    Analyze this educational image and return plain text in exactly this format:
    Topic: <short topic name>
    Visual Focus: <2 or 3 short phrases separated by commas>
    Learning Notes: <3 or 4 sentences that explain the image for a student>
    """
    response = get_model_ai().generate_content([prompt, Image.open(io.BytesIO(image_bytes))])
    return response.text


def parse_image_analysis(analysis_text):
    topic_match = re.search(r"Topic:\s*(.*)", analysis_text)
    focus_match = re.search(r"Visual Focus:\s*(.*)", analysis_text)
    notes_match = re.search(r"Learning Notes:\s*(.*)", analysis_text, re.DOTALL)

    topic = topic_match.group(1).strip() if topic_match else "Educational topic"
    visual_focus = focus_match.group(1).strip() if focus_match else ""
    learning_notes = notes_match.group(1).strip() if notes_match else analysis_text.strip()

    combined_text = learning_notes
    if visual_focus:
        combined_text = f"{topic}. {visual_focus}. {learning_notes}"

    return topic, combined_text


def extract_key_points(summary_text, limit=4):
    sanitized_text = sanitize_learning_text(summary_text)
    lines = [line.strip("-• ").strip() for line in sanitized_text.splitlines() if line.strip()]
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", sanitized_text or "") if sentence.strip()]
    candidates = unique_preserve(lines + sentences)

    filtered = []
    for candidate in candidates:
        if not looks_like_meaningful_point(candidate):
            continue
        filtered.append(candidate)
        if len(filtered) == limit:
            break

    return filtered


def build_video_search_topic(content_text, input_kind):
    key_points = extract_key_points(content_text, limit=3)
    topic = " ".join(key_points)

    if input_kind == "Image":
        return f"{topic} educational visual explanation animation"
    if input_kind == "Video":
        return f"{topic} educational explanation"
    return f"{topic} visual explanation for students"


def extract_topic_title(text, fallback="Study Topic"):
    points = extract_key_points(text, limit=1)
    topic = clean_text(points[0] if points else fallback)
    topic = re.sub(r"^(what|how|why|explain|describe)\s+", "", topic, flags=re.IGNORECASE)
    topic = re.split(r"[.:;!?]", topic)[0]
    leading_phrase = re.split(r"\b(is|are|was|were|refers to|means)\b", topic, maxsplit=1, flags=re.IGNORECASE)[0]
    if clean_text(leading_phrase):
        topic = leading_phrase
    words = topic.split()
    if len(words) > 8:
        topic = " ".join(words[:8])
    return topic[:70] or fallback


def build_youtube_search_url(search_topic):
    return f"https://www.youtube.com/results?search_query={quote_plus(search_topic)}"


def should_show_chart(text, points):
    combined = clean_text(f"{text} {' '.join(points)}").lower()
    chart_signals = [
        "increase",
        "decrease",
        "compare",
        "comparison",
        "trend",
        "growth",
        "decline",
        "stages",
        "phase",
        "timeline",
        "period",
        "dynasty",
        "economy",
        "population",
        "emotion",
        "distribution",
    ]
    return any(signal in combined for signal in chart_signals)


def should_show_concept_map(text, points):
    combined = clean_text(f"{text} {' '.join(points)}").lower()
    concept_signals = [
        "system",
        "structure",
        "relationship",
        "network",
        "classification",
        "types",
        "causes",
        "effects",
        "components",
        "concept",
        "sultanate",
        "kingdom",
        "process",
    ]
    return any(signal in combined for signal in concept_signals)


def build_visual_cards(topic, visual_frames):
    labels = ["Core Scene", "Process Scene", "Meaning Scene", "Recall Scene"]
    prompt_starters = {
        "Core Scene": "Create a polished educational illustration that introduces the core definition",
        "Process Scene": "Create a process-focused classroom graphic that explains how the topic works",
        "Meaning Scene": "Create a meaningful applied illustration that shows why the topic matters",
        "Recall Scene": "Create a memory-friendly revision visual that helps students retain the topic",
    }
    cards = []
    for index, frame in enumerate(visual_frames[:4]):
        title = clean_text(frame.get("title", "")) or labels[index]
        focus = clamp_text(frame.get("focus", ""), 90)
        explanation = clamp_text(frame.get("explanation", ""), 150)
        label = labels[index]
        scene_prompt = clamp_text(
            f"{prompt_starters.get(label, 'Create a professional educational illustration')} for {topic}. "
            f"Main emphasis: {focus}. Supporting explanation: {explanation}. "
            "Clean classroom poster style, clear composition, readable labels, warm academic palette.",
            180,
        )
        cards.append(
            {
                "label": label,
                "title": title,
                "focus": focus,
                "explanation": explanation,
                "scene_prompt": scene_prompt,
                "image_note": "Use this scene to present the topic in a clean visual way for student understanding.",
            }
        )
    return cards


def build_hf_image_prompt(topic, card):
    focus = clamp_text(clean_media_prompt_text(card.get("focus", "")), 120)
    explanation = clamp_text(clean_media_prompt_text(card.get("explanation", "")), 140)
    return (
        f"Educational illustration for final year project presentation about {topic}. "
        f"Show {focus}. Include visual explanation of {explanation}. "
        "Professional classroom poster style, clean composition, readable labels, visually engaging, "
        "high quality educational artwork, warm academic colors, no watermark, no distorted text."
    )


def build_hf_video_prompt(topic, card):
    focus = clamp_text(clean_media_prompt_text(card.get("focus", "")), 120)
    explanation = clamp_text(clean_media_prompt_text(card.get("explanation", "")), 140)
    return (
        f"Short educational video for students about {topic}. "
        f"Scene focus: {focus}. Narratively show: {explanation}. "
        "Professional classroom animation, clean educational style, readable visual storytelling, "
        "presentation ready, smooth camera motion, no watermark, no distorted text."
    )


@lru_cache(maxsize=16)
def generate_hf_image_base64(prompt):
    if not HUGGINGFACE_API_KEY:
        return ""

    api_url = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"
    payload = json.dumps(
        {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": "blurry, distorted, unreadable, watermark, duplicate, low quality, bad anatomy",
            },
        }
    ).encode("utf-8")
    request_obj = Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        response = urlopen(request_obj, timeout=60)
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
        if content_type.startswith("image/") and data:
            return base64.b64encode(data).decode("utf-8")
        return ""
    except (HTTPError, URLError, TimeoutError):
        return ""
    except Exception:
        return ""


@lru_cache(maxsize=8)
def generate_hf_video_base64(prompt):
    if not HUGGINGFACE_API_KEY:
        return ""

    api_url = f"https://api-inference.huggingface.co/models/{HF_VIDEO_MODEL}"
    payload = json.dumps({"inputs": prompt}).encode("utf-8")
    request_obj = Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        response = urlopen(request_obj, timeout=120)
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
        if content_type.startswith("video/") and data:
            return base64.b64encode(data).decode("utf-8")
        return ""
    except (HTTPError, URLError, TimeoutError):
        return ""
    except Exception:
        return ""


def build_generated_visuals(topic, visual_cards, limit=2):
    visuals = []
    for card in (visual_cards or [])[:limit]:
        prompt = build_hf_image_prompt(topic, card)
        image_b64 = generate_hf_image_base64(prompt)
        if not image_b64:
            continue
        visuals.append(
            {
                "title": card.get("title", "AI Visual"),
                "caption": clamp_text(card.get("explanation", card.get("focus", "AI generated topic illustration")), 140),
                "image_b64": image_b64,
            }
        )
    return visuals


def build_generated_video(topic, visual_cards):
    if not visual_cards:
        return {}
    primary_card = visual_cards[0]
    prompt = build_hf_video_prompt(topic, primary_card)
    video_b64 = generate_hf_video_base64(prompt)
    if not video_b64:
        return {}
    return {
        "title": primary_card.get("title", "AI Topic Video"),
        "caption": clamp_text(primary_card.get("explanation", primary_card.get("focus", "AI generated topic video")), 160),
        "video_b64": video_b64,
        "mime": "video/mp4",
    }


def build_topic_support_points(topic, points):
    support_points = []
    for point in unique_preserve(points):
        cleaned = clamp_text(dedupe_repeated_clause(point), 150)
        if looks_like_meaningful_point(cleaned) and not any(is_near_duplicate(cleaned, existing) for existing in support_points):
            support_points.append(cleaned)
    templates = [
        f"{topic} can be understood through its definition, structure, and purpose.",
        f"The most important parts of {topic} should be revised with examples and clear cause-and-effect links.",
        f"Students should connect the main idea of {topic} with its significance, outcomes, or historical role.",
        f"A good revision strategy for {topic} is to compare the key features and remember them with one clear example.",
    ]
    for template in templates:
        if len(support_points) >= 4:
            break
        if not any(is_near_duplicate(template, existing) for existing in support_points):
            support_points.append(template)
    return support_points[:3]


def build_exam_ready_points(topic, points):
    base_points = unique_preserve(points)
    exam_points = []
    for point in base_points[:4]:
        point_text = clamp_text(dedupe_repeated_clause(point), 170)
        if looks_like_meaningful_point(point_text) and not any(is_near_duplicate(point_text, existing) for existing in exam_points):
            exam_points.append(point_text)

    defaults = [
        f"{topic} should be explained with its definition, core features, and significance.",
        f"Students should connect {topic} with its causes, structure, and results.",
        f"A complete exam answer on {topic} should include one example or application.",
        f"The safest revision method for {topic} is to remember the topic, its features, and its importance together.",
    ]
    for item in defaults:
        if len(exam_points) >= 4:
            break
        if not any(is_near_duplicate(item, existing) for existing in exam_points):
            exam_points.append(item)
    return exam_points[:4]


def build_video_learning_paths(topic):
    return [
        {"label": "Quick overview", "query": f"{topic} in 5 minutes"},
        {"label": "Detailed explanation", "query": f"{topic} full explanation for students"},
        {"label": "Exam revision", "query": f"{topic} important questions and answers"},
    ]


def build_theory_notes(topic, points):
    notes = build_topic_support_points(topic, points)
    return [
        {"heading": "Definition", "content": notes[0]},
        {"heading": "Core Feature", "content": notes[1] if len(notes) > 1 else notes[0]},
        {"heading": "Importance", "content": notes[2] if len(notes) > 2 else notes[0]},
    ]


def build_audio_learning_items(topic, points):
    items = []
    templates = [
        ("Podcast Opening", f"Start by introducing the main idea of {topic} clearly."),
        ("Main Explanation", f"Explain one core feature of {topic} in spoken form."),
        ("Memory Cue", f"End with why {topic} matters so the learner remembers it."),
    ]
    for index, point in enumerate(points[:3]):
        title, lead = templates[index]
        items.append(
            {
                "title": title,
                "lead": lead,
                "content": clamp_text(point, 170),
            }
        )
    return items


def build_audio_script(topic, points):
    sentences = [clamp_text(point, 160) for point in points[:3]]
    if not sentences:
        return topic
    intro = f"Today's learning podcast is about {topic}."
    closing = f"In short, remember {topic} through its definition, working idea, and importance."
    return " ".join([intro, *sentences, closing])


def build_visual_learning_path(topic, points):
    base = points[:3] if points else [f"{topic} should be understood clearly."]
    step_labels = ["See It", "Connect It", "Remember It"]
    guides = [
        f"Start by observing the main picture of {topic} and identify the core idea.",
        f"Link the visual with how {topic} works or why it changes.",
        f"Store one strong takeaway about {topic} that you can recall in exams.",
    ]
    items = []
    for index, label in enumerate(step_labels):
        point = base[index] if index < len(base) else base[-1]
        items.append(
            {
                "label": label,
                "title": clamp_text(point, 68),
                "description": guides[index],
            }
        )
    return items


def build_visual_memory_boosters(topic, points):
    base = points[:3] if points else [f"{topic} is important for understanding."]
    prompts = [
        "Picture this",
        "Ask yourself",
        "One-line recall",
    ]
    boosters = []
    for index, prompt in enumerate(prompts):
        point = base[index] if index < len(base) else base[-1]
        boosters.append(
            {
                "prompt": prompt,
                "content": clamp_text(point, 130),
            }
        )
    return boosters


def find_related_youtube_video_by_query(search_topic):
    query = quote_plus(search_topic)
    search_url = f"https://www.youtube.com/results?search_query={query}"
    request_obj = Request(search_url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        html = urlopen(request_obj, timeout=1.5).read().decode("utf-8", errors="ignore")
        matches = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
        if matches:
            return f"https://www.youtube.com/watch?v={matches[0]}"
    except Exception:
        return ""

    return ""


def youtube_thumbnail_from_url(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return ""
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def build_fallback_study_pack(text, explicit_topic=""):
    topic = explicit_topic or extract_topic_title(text)
    points = build_topic_support_points(topic, extract_key_points(text, limit=4))
    theory_notes = build_theory_notes(topic, points)
    audio_items = build_audio_learning_items(topic, points)
    overview = clean_text(" ".join(points[:2]))[:220]

    chart_items = []
    for index, point in enumerate(points[:3]):
        chart_items.append(
            {
                "label": clean_text(point)[:28] or clamp_text(topic, 28),
                "value": min(100, 42 + len(point.split()) * 5 + index * 4),
            }
        )

    core_point = points[0] if points else f"{topic} should be understood through its central definition."
    process_point = points[1] if len(points) > 1 else f"The learner should understand how {topic} is structured or explained."
    meaning_point = points[2] if len(points) > 2 else f"The importance of {topic} should be connected with its real significance."
    visual_frames = [
        {
            "title": "Core Scene",
            "focus": clamp_text(f"Definition and central idea of {topic}", 72),
            "explanation": clean_text(core_point),
        },
        {
            "title": "Process Scene",
            "focus": clamp_text(f"How {topic} works or is organized", 72),
            "explanation": clean_text(process_point),
        },
        {
            "title": "Meaning Scene",
            "focus": clamp_text(f"Why {topic} matters for learners", 72),
            "explanation": clean_text(meaning_point),
        },
    ]

    return {
        "topic": topic,
        "overview": overview,
        "key_points": [clean_text(point)[:160] for point in points[:3]],
        "theory_notes": theory_notes,
        "audio_learning_items": audio_items,
        "audio_script": build_audio_script(topic, points),
        "visual_learning_path": build_visual_learning_path(topic, points),
        "visual_memory_boosters": build_visual_memory_boosters(topic, points),
        "chart_title": f"{topic[:36]} learning emphasis",
        "chart_items": chart_items,
        "map_nodes": [clean_text(point)[:40] for point in points[:3]],
        "visual_frames": visual_frames,
        "visual_cards": build_visual_cards(topic, visual_frames),
        "video_query": f"{topic} explanation for students",
        "show_chart": should_show_chart(text, points),
        "show_concept_map": should_show_concept_map(text, points),
    }


def build_local_summary(text, learner):
    clean_text_input = sanitize_learning_text(text) or text
    study_pack = build_fallback_study_pack(clean_text_input)
    topic = study_pack.get("topic", "Study Topic")

    if learner == "Theory":
        return f"{topic}: {study_pack.get('overview', '')}".strip()

    if learner == "Audio":
        return study_pack.get("audio_script", study_pack.get("overview", ""))

    return study_pack.get("overview", "")


def build_study_pack(summary_text, explicit_topic=""):
    trimmed_summary = prepare_model_text(summary_text, limit=5000)
    topic_hint = clean_text(explicit_topic)[:80]
    return build_study_pack_cached(trimmed_summary, topic_hint)


@lru_cache(maxsize=32)
def build_study_pack_cached(summary_text, explicit_topic=""):
    return build_fallback_study_pack(summary_text, explicit_topic=explicit_topic)


def text_wrap(text, width=40):
    words = (text or "").split()
    lines = []
    current_line = []

    for word in words:
        trial_line = " ".join(current_line + [word])
        if len(trial_line) <= width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def draw_text_block(draw, xy, text, font, fill, width):
    wrapped = text_wrap(text, width=width)
    draw.multiline_text(xy, wrapped, fill=fill, font=font, spacing=8)


def clamp_text(text, max_chars=180):
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[: max_chars - 3].rsplit(" ", 1)[0]
    return (truncated or cleaned[: max_chars - 3]) + "..."


def split_point_for_slide(point):
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", point or "") if part.strip()]
    if len(sentences) >= 2:
        main_idea = sentences[0]
        support = " ".join(sentences[1:3])
    else:
        chunks = re.split(r",|;|:", point or "")
        main_idea = chunks[0].strip() if chunks else "Main idea"
        support = ". ".join(chunk.strip() for chunk in chunks[1:3] if chunk.strip())

    if not support:
        support = f"This idea supports understanding of {main_idea.lower()} in a visual way."

    return main_idea, support


def create_storyboard_images(summary_text, topic="Topic", study_pack=None):
    study_pack = study_pack or build_fallback_study_pack(summary_text, explicit_topic=topic)
    frames_data = study_pack.get("visual_frames", [])
    frames = []
    colors = [PALETTE["deep"], PALETTE["crimson"], PALETTE["orange"], "#B85C38", "#8C2F39"]

    for idx, frame_data in enumerate(frames_data):
        main_idea = clamp_text(frame_data.get("focus", "") or f"{topic} idea {idx + 1}", 96)
        support_text = clamp_text(frame_data.get("explanation", "") or main_idea, 150)
        section_title = clean_text(frame_data.get("title", "")) or f"Slide {idx + 1}"
        topic_label = clamp_text(study_pack.get("topic", topic), 42)
        image = Image.new("RGB", (1400, 900), "#fbf6ee")
        draw = ImageDraw.Draw(image)
        title_font = load_font(56, bold=True)
        headline_font = load_font(42, bold=True)
        subtitle_font = load_font(30, bold=True)
        body_font = load_font(28)
        small_font = load_font(22)
        accent = colors[idx % len(colors)]

        draw.rounded_rectangle((36, 34, 1364, 864), radius=36, fill="#fffdf8", outline="#d9c7ab", width=3)
        draw.rounded_rectangle((36, 34, 1364, 170), radius=36, fill=accent)
        draw.text((74, 62), topic_label, fill="white", font=title_font)
        draw.text((76, 120), f"Study Frame {idx + 1}  |  {clamp_text(section_title, 30)}", fill=PALETTE["sand"], font=small_font)

        draw.rounded_rectangle((72, 214, 498, 802), radius=28, fill="#f6ebdd", outline="#dbc4a3", width=2)
        draw.text((108, 246), "Visual Anchor", fill=PALETTE["deep"], font=subtitle_font)
        draw.ellipse((122, 320, 290, 488), fill=accent, outline=PALETTE["deep"], width=4)
        draw.rounded_rectangle((312, 336, 446, 470), radius=24, fill="#fff7eb", outline=PALETTE["deep"], width=3)
        draw.line((170, 552, 420, 552), fill=PALETTE["crimson"], width=5)
        draw.rounded_rectangle((104, 594, 458, 742), radius=18, fill="#fff7ee", outline="#d9c7ab", width=2)
        draw.text((156, 385), "Core", fill="white", font=subtitle_font)
        draw.text((340, 382), "Link", fill=PALETTE["deep"], font=subtitle_font)
        draw.text((122, 610), "Image cue", fill=PALETTE["crimson"], font=small_font)
        draw_text_block(draw, (122, 652), clamp_text(main_idea, 70), small_font, "#58342B", 20)

        draw.rounded_rectangle((540, 214, 1328, 496), radius=28, fill="#fff8ef", outline="#dbc4a3", width=2)
        draw.text((584, 246), clamp_text(section_title, 28), fill=PALETTE["deep"], font=subtitle_font)
        draw_text_block(draw, (584, 306), main_idea, headline_font, PALETTE["ink"], 26)

        draw.rounded_rectangle((540, 530, 1328, 802), radius=28, fill="#fffdf8", outline="#dbc4a3", width=2)
        draw.text((584, 564), "Topic Connection", fill=PALETTE["deep"], font=subtitle_font)
        draw_text_block(draw, (584, 618), support_text, body_font, "#58342B", 32)
        draw.line((582, 702, 1286, 702), fill="#deccb4", width=2)
        draw.text((584, 738), "Study Cue", fill=PALETTE["crimson"], font=small_font)
        focus_text = clamp_text(
            f"Remember how this point supports {topic_label} and connects to the bigger explanation.",
            110,
        )
        draw_text_block(draw, (744, 736), focus_text, small_font, "#58342B", 24)
        frames.append(image)

    return frames


def generate_visual_insights(summary_text, topic, study_pack=None):
    study_pack = study_pack or build_fallback_study_pack(summary_text, explicit_topic=topic)
    return {
        "chart_title": study_pack.get("chart_title", f"{topic[:40]} concept weight chart"),
        "chart_items": study_pack.get("chart_items", []),
        "map_nodes": study_pack.get("map_nodes", []),
    }


def create_visual_dashboard_image(summary_text, topic="Topic", study_pack=None):
    image = Image.new("RGB", (1200, 720), PALETTE["paper"])
    draw = ImageDraw.Draw(image)
    title_font = load_font(34, bold=True)
    subtitle_font = load_font(22, bold=True)
    body_font = load_font(22)
    study_pack = study_pack or build_fallback_study_pack(summary_text, explicit_topic=topic)
    points = study_pack.get("key_points", extract_key_points(summary_text, limit=4))

    draw.rounded_rectangle((24, 24, 1176, 696), radius=26, fill="#FFFDF9", outline=PALETTE["line"], width=2)
    draw.rounded_rectangle((24, 24, 1176, 150), radius=26, fill=PALETTE["deep"])
    draw.text((56, 54), study_pack.get("topic", topic)[:44], fill="white", font=title_font)
    draw.text((56, 94), "Topic Snapshot", fill=PALETTE["sand"], font=subtitle_font)

    card_positions = [
        (56, 190, 570, 360),
        (630, 190, 1144, 360),
        (56, 400, 570, 570),
        (630, 400, 1144, 570),
    ]
    card_colors = [PALETTE["sand"], "#F6C7B5", "#F4E2C8", "#F7D9A6"]

    for idx, point in enumerate(points):
        x1, y1, x2, y2 = card_positions[idx]
        draw.rounded_rectangle((x1, y1, x2, y2), radius=20, fill=card_colors[idx], outline=PALETTE["line"], width=2)
        draw.text((x1 + 20, y1 + 18), f"Concept {idx + 1}", fill=PALETTE["deep"], font=subtitle_font)
        draw_text_block(draw, (x1 + 20, y1 + 58), point, body_font, PALETTE["ink"], 38)

    draw.rounded_rectangle((56, 602, 1144, 660), radius=16, fill="#FFF4E4", outline=PALETTE["line"], width=2)
    footer = study_pack.get("overview", f"Use these four concepts as the main study checkpoints for {topic[:50]}.")
    draw_text_block(draw, (76, 618), footer, body_font, "#6D4033", 90)
    return image


def create_bar_chart_image(summary_text, topic="Topic", insights=None, study_pack=None):
    image = Image.new("RGB", (1200, 620), "#FFFBF4")
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    subtitle_font = load_font(20, bold=True)
    body_font = load_font(20)
    study_pack = study_pack or build_fallback_study_pack(summary_text, explicit_topic=topic)
    points = study_pack.get("key_points", extract_key_points(summary_text, limit=4))
    insights = insights or generate_visual_insights(summary_text, topic, study_pack)
    chart_items = insights.get("chart_items", [])
    labels = [item["label"] for item in chart_items[:4]] or [clean_text(point)[:26] for point in points]
    scores = [item["value"] for item in chart_items[:4]] or [max(20, min(100, len(point.split()) * 7)) for point in points]
    colors = [PALETTE["deep"], PALETTE["crimson"], PALETTE["orange"], "#B85C38"]

    draw.rounded_rectangle((24, 24, 1176, 596), radius=24, fill="#FFFDF8", outline=PALETTE["line"], width=2)
    draw.text((52, 48), f"{study_pack.get('topic', topic)[:42]} learning emphasis", fill=PALETTE["deep"], font=title_font)
    draw.text((52, 86), insights.get("chart_title", "Concept weight chart")[:55], fill="#6D4033", font=subtitle_font)

    chart_left = 120
    chart_right = 1080
    chart_bottom = 520
    chart_top = 150

    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=PALETTE["deep"], width=3)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=PALETTE["deep"], width=3)

    for step in range(0, 101, 25):
        y = chart_bottom - ((chart_bottom - chart_top) * step / 100)
        draw.line((chart_left, y, chart_right, y), fill="#E7D2B1", width=1)
        draw.text((62, y - 10), str(step), fill="#6D4033", font=body_font)

    bar_width = 150
    gap = 70
    for idx, (label, score) in enumerate(zip(labels, scores)):
        x1 = chart_left + 70 + idx * (bar_width + gap)
        x2 = x1 + bar_width
        y1 = chart_bottom - ((chart_bottom - chart_top) * score / 100)
        draw.rounded_rectangle((x1, y1, x2, chart_bottom), radius=16, fill=colors[idx], outline=colors[idx])
        draw.text((x1 + 44, y1 - 28), str(score), fill=PALETTE["ink"], font=subtitle_font)
        draw_text_block(draw, (x1, chart_bottom + 18), label[:32], body_font, PALETTE["ink"], 16)

    return image


def create_concept_map_image(summary_text, topic="Topic", insights=None, study_pack=None):
    image = Image.new("RGB", (1200, 720), PALETTE["paper"])
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    subtitle_font = load_font(20, bold=True)
    body_font = load_font(18)
    study_pack = study_pack or build_fallback_study_pack(summary_text, explicit_topic=topic)
    insights = insights or generate_visual_insights(summary_text, topic, study_pack)
    points = insights.get("map_nodes", extract_key_points(summary_text, limit=4))

    draw.rounded_rectangle((28, 28, 1172, 692), radius=28, fill="#FFFCF7", outline=PALETTE["line"], width=2)
    center_box = (445, 275, 755, 445)
    draw.rounded_rectangle(center_box, radius=26, fill=PALETTE["deep"], outline=PALETTE["deep"], width=2)
    draw.text((475, 316), study_pack.get("topic", topic)[:28], fill="white", font=title_font)
    draw.text((475, 354), "Core Topic", fill=PALETTE["sand"], font=subtitle_font)

    node_boxes = [
        (90, 120, 360, 250),
        (840, 120, 1110, 250),
        (90, 470, 360, 600),
        (840, 470, 1110, 600),
    ]
    node_colors = [PALETTE["sand"], "#F8D0C0", "#F5E3C9", "#F9D79B"]
    connectors = [
        ((360, 185), (445, 330)),
        ((840, 185), (755, 330)),
        ((360, 535), (445, 390)),
        ((840, 535), (755, 390)),
    ]

    for (start, end) in connectors:
        draw.line((start[0], start[1], end[0], end[1]), fill=PALETTE["crimson"], width=5)

    for idx, point in enumerate(points):
        box = node_boxes[idx]
        draw.rounded_rectangle(box, radius=22, fill=node_colors[idx], outline=PALETTE["line"], width=2)
        draw.text((box[0] + 18, box[1] + 16), f"Key Area {idx + 1}", fill=PALETTE["deep"], font=subtitle_font)
        draw_text_block(draw, (box[0] + 18, box[1] + 50), point, body_font, PALETTE["ink"], 20)

    return image


def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_storyboard_frame_b64(summary_text, topic="Topic", study_pack=None):
    return [image_to_base64(frame) for frame in create_storyboard_images(summary_text, topic, study_pack)]


def build_visual_assets(summary_text, topic, study_pack=None):
    study_pack = study_pack or build_fallback_study_pack(summary_text, explicit_topic=topic)
    insights = generate_visual_insights(summary_text, topic, study_pack)
    assets = {"dashboard": image_to_base64(create_visual_dashboard_image(summary_text, topic, study_pack))}
    if study_pack.get("show_chart"):
        assets["chart"] = image_to_base64(create_bar_chart_image(summary_text, topic, insights, study_pack))
    if not FAST_VISUAL_MODE and study_pack.get("show_concept_map"):
        assets["concept_map"] = image_to_base64(create_concept_map_image(summary_text, topic, insights, study_pack))
    return assets


def build_visual_package(summary_text, topic, study_pack=None):
    if FAST_VISUAL_MODE:
        return {}, []
    serialized_pack = json.dumps(study_pack or build_fallback_study_pack(summary_text, explicit_topic=topic), sort_keys=True)
    return build_visual_package_cached(summary_text, topic, serialized_pack)


@lru_cache(maxsize=24)
def build_visual_package_cached(summary_text, topic, serialized_pack):
    study_pack = json.loads(serialized_pack)
    visual_assets = {}
    visual_slides = []
    try:
        visual_assets = build_visual_assets(summary_text, topic, study_pack)
    except Exception:
        visual_assets = {}

    try:
        slide_limit = 1 if FAST_VISUAL_MODE else 3
        visual_slides = create_storyboard_frame_b64(summary_text, topic, study_pack)[:slide_limit]
    except Exception:
        visual_slides = []

    return visual_assets, visual_slides


def generate_summary(text, learner):
    if FAST_LOCAL_SUMMARY:
        return build_local_summary(text, learner)
    return generate_summary_cached(prepare_model_text(text), learner)


@lru_cache(maxsize=32)
def generate_summary_cached(text, learner):
    if learner == "Visual":
        base = "concise, topic-focused points that are easy to convert into charts, diagrams, and visual study cards"
    elif learner == "Audio":
        base = "spoken, easy-to-understand explanation that stays tightly on topic"
    else:
        base = "clear professional theory explanation that stays tightly on topic"

    prompt = f"""
    Create a professional student-friendly summary for the learner.
    Style of explanation: {base}
    Rules:
    - Stay focused on one topic.
    - Do not drift into unrelated examples.
    - Use precise academic wording that is still easy to understand.

    Content:
    {text}
    """
    try:
        response = get_model_ai().generate_content(prompt)
        return response.text
    except Exception:
        fallback_points = extract_key_points(text, limit=4)
        return "\n".join(f"- {point}" for point in fallback_points)


def generate_audio_base64(text):
    if gTTS is None:
        raise RuntimeError("gTTS is not installed.")
    buffer = io.BytesIO()
    gTTS(text).write_to_fp(buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_mcq(text, level, num, topic="", study_pack=None):
    study_pack = study_pack or build_fallback_study_pack(text, explicit_topic=topic)
    key_points = "\n".join(f"- {point}" for point in study_pack.get("key_points", []))
    prompt = f"""
    Generate EXACTLY {num} topic-specific MCQs.

    STRICT RULES:
    - Each question MUST have 4 options (A, B, C, D)
    - Answer MUST be correct and randomly A/B/C/D (not always A)
    - Do NOT repeat same answer pattern
    - Every question and every correct answer must stay inside the topic
    - Avoid generic wording and avoid unrelated distractors
    - Format strictly:

    Q1. Question
    A) option
    B) option
    C) option
    D) option
    Answer: C

    Topic: {study_pack.get("topic", topic)}
    Focus points:
    {key_points}

    Content:
    {text}
    """
    return get_model_ai().generate_content(prompt).text


def generate_mcq_retry(text, level, num, topic="", study_pack=None):
    study_pack = study_pack or build_fallback_study_pack(text, explicit_topic=topic)
    prompt = f"""
    Create exactly {num} multiple choice questions from the text below.
    Difficulty: {level}
    Topic: {study_pack.get("topic", topic)}

    Very important rules:
    1. Start each question with Q1., Q2., Q3.
    2. Give exactly 4 options labeled A), B), C), D)
    3. End each question with Answer: A or B or C or D
    4. Do not add markdown, explanations, headings, or extra text
    5. Questions must stay tightly connected to the topic and key ideas

    Text:
    {text}
    """
    return get_model_ai().generate_content(prompt).text


def generate_broad_questions(text, level, num, topic="", study_pack=None):
    study_pack = study_pack or build_fallback_study_pack(text, explicit_topic=topic)
    key_points = "\n".join(f"- {point}" for point in study_pack.get("key_points", []))
    prompt = f"""
    Create exactly {num} broad descriptive questions from the text below.
    Difficulty: {level}
    Topic: {study_pack.get("topic", topic)}

    Very important rules:
    1. Start each question with Q1., Q2., Q3.
    2. Return only the questions
    3. Do not add answers, headings, markdown, or extra explanation
    4. Questions should encourage descriptive student answers
    5. Keep all questions inside this topic and its key points

    Key points:
    {key_points}

    Text:
    {text}
    """
    return get_model_ai().generate_content(prompt).text


def parse_mcq(mcq_text):
    questions = []
    cleaned_text = (mcq_text or "").replace("\r\n", "\n")
    cleaned_text = re.sub(r"```.*?```", "", cleaned_text, flags=re.DOTALL)
    cleaned_text = cleaned_text.replace("**", "")

    blocks = re.split(r"\n(?=Q\d+\.)", cleaned_text)
    for block in blocks:
        block = block.strip()
        if not block or not re.match(r"Q\d+\.", block):
            continue

        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 6:
            continue

        question_line = lines[0]
        option_lines = []
        answer_letter = None

        for line in lines[1:]:
            normalized_line = line.replace("*", "").strip()
            if re.match(r"^[A-D][\).]\s*", normalized_line):
                option_lines.append(re.sub(r"^([A-D])[\).]\s*", r"\1) ", normalized_line))
            elif normalized_line.lower().startswith("answer:"):
                match = re.search(r"\b([ABCD])\b", normalized_line.upper())
                if match:
                    answer_letter = match.group(1)

        if len(option_lines) != 4 or answer_letter is None:
            continue

        correct_option = next((opt for opt in option_lines if opt.startswith(f"{answer_letter})")), "")
        question_text = re.sub(r"^Q\d+\.\s*", "", question_line).strip()
        questions.append(
            {
                "question": question_text,
                "options": option_lines,
                "answer": answer_letter,
                "correct_option": correct_option,
            }
        )

    return questions


def parse_broad_questions(question_text):
    matches = re.findall(r"Q\d+\.\s*(.*)", question_text or "")
    return [match.strip() for match in matches if match.strip()]


def build_fallback_mcq_questions(text, num_questions, topic="", study_pack=None):
    study_pack = study_pack or build_fallback_study_pack(text, explicit_topic=topic)
    points = study_pack.get("key_points", extract_key_points(text, limit=max(4, num_questions)))
    topic_name = study_pack.get("topic", topic) or extract_topic_title(text)
    questions = []
    for index in range(num_questions):
        point = points[index % len(points)]
        core = clean_text(point)
        distractors = []
        for candidate in points:
            candidate_clean = clean_text(candidate)
            if candidate_clean != core and candidate_clean not in distractors:
                distractors.append(candidate_clean)
            if len(distractors) == 3:
                break
        while len(distractors) < 3:
            distractors.append(build_topic_support_points(topic_name, points)[len(distractors) % 4])

        question_templates = [
            f"Which statement is most accurate about {topic_name}?",
            f"Which option best explains this key idea from {topic_name}?",
            f"Which point should a student remember about {topic_name}?",
            f"Which statement correctly matches the current topic, {topic_name}?",
        ]
        options = [core, distractors[0], distractors[1], distractors[2]]
        option_lines = [
            f"A) {options[0]}",
            f"B) {options[1]}",
            f"C) {options[2]}",
            f"D) {options[3]}",
        ]
        questions.append(
            {
                "question": f"{question_templates[index % len(question_templates)]} Focus: {core[:80]}",
                "options": option_lines,
                "answer": "A",
                "correct_option": option_lines[0],
            }
        )
    return questions


def build_fallback_broad_questions(text, num_questions, topic="", study_pack=None):
    study_pack = study_pack or build_fallback_study_pack(text, explicit_topic=topic)
    points = study_pack.get("key_points", extract_key_points(text, limit=max(4, num_questions)))
    topic_name = study_pack.get("topic", topic) or extract_topic_title(text)
    questions = []
    for index in range(num_questions):
        point = clean_text(points[index % len(points)])
        questions.append(f"Explain how this idea supports the topic {topic_name}: {point}")
    return questions


def evaluate_broad_answer(question, answer, topic="", topic_summary=""):
    prompt = f"""
    Evaluate this student's descriptive answer.

    Topic: {topic}
    Topic summary: {topic_summary}
    Question: {question}
    Answer: {answer}

    Scoring rules:
    - Give 1 if the answer is relevant, meaningful, and reasonably correct
    - Give 0 if the answer is blank, too short, off-topic, or clearly incorrect
    - Respond with only one character: 0 or 1
    """
    try:
        result = get_model_ai().generate_content(prompt).text.strip()
        return 1 if result.startswith("1") else 0
    except Exception:
        return 1 if len(clean_text(answer)) > 30 else 0


def calculate_reward(score, difficulty):
    points_per_question = {"Easy": 5, "Medium": 10, "Hard": 15}
    return score * points_per_question[difficulty]


def predict(text):
    if not ENABLE_PREDICTION:
        return ""
    normalized = prepare_model_text(text, limit=MAX_PREDICTION_INPUT_CHARS)
    return predict_cached(normalized)


@lru_cache(maxsize=64)
def predict_cached(text):
    if prediction_model is None:
        return "Prediction model unavailable"
    try:
        return prediction_model.predict([text])[0]
    except Exception:
        return "Prediction Error"


def build_context():
    ensure_session_defaults()
    server_output = get_server_output()
    summary = session.get("last_summary", "")
    learner = session.get("learner_type", "Theory")
    processed_text = session.get("processed_text", "")
    processed_topic = session.get("processed_topic", "")
    input_type = session.get("processed_input_type", "")
    video_url = session.get("processed_video_url", "")
    image_b64 = server_output.get("processed_image_b64")
    search_topic = session.get("video_search_topic", "")

    related_video_url = session.get("related_video_url", "")

    visual_slides = server_output.get("visual_slides", [])
    visual_assets = server_output.get("visual_assets", {})
    generated_visuals = server_output.get("generated_visuals", [])
    generated_video = server_output.get("generated_video", {})
    study_pack = server_output.get("study_pack", {})
    audio_b64 = server_output.get("audio_b64")
    audio_error = server_output.get("audio_error")
    primary_video_url = video_url or related_video_url
    primary_video_thumbnail = youtube_thumbnail_from_url(primary_video_url)

    return {
        "summary": summary,
        "study_pack": study_pack,
        "learner_type": learner,
        "question_type": session.get("question_type", "MCQ"),
        "difficulty": session.get("difficulty", "Easy"),
        "num_questions": session.get("num_questions", 3),
        "source_label": session.get("processed_source", ""),
        "processed_text": processed_text,
        "processed_input_type": input_type,
        "processed_topic": processed_topic,
        "processed_video_url": video_url,
        "related_video_url": related_video_url,
        "video_search_url": build_youtube_search_url(search_topic) if search_topic else "",
        "primary_video_thumbnail": primary_video_thumbnail,
        "processed_image_b64": image_b64,
        "visual_assets": visual_assets,
        "generated_visuals": generated_visuals,
        "generated_video": generated_video,
        "visual_slides": visual_slides,
        "audio_b64": audio_b64,
        "audio_error": audio_error,
        "quiz_data": session.get("quiz_data", []),
        "quiz_topic": session.get("quiz_topic", ""),
        "quiz_score": session.get("quiz_score"),
        "quiz_reward": session.get("quiz_reward"),
        "points_per_question": {"Easy": 5, "Medium": 10, "Hard": 15}[session.get("difficulty", "Easy")],
        "broad_questions": session.get("broad_questions", []),
        "broad_topic": session.get("broad_topic", ""),
        "broad_results": session.get("broad_results", []),
        "broad_score": session.get("broad_score"),
        "broad_reward": session.get("broad_reward"),
        "prediction": session.get("prediction", ""),
        "processed_image_mime": server_output.get("processed_image_mime", "image/png"),
        "has_output": bool(processed_text),
        "history": session.get("history", []),
        "chat_messages": session.get("chat_messages", []),
        "contact_messages": session.get("contact_messages", []),
        "total_points": session.get("total_points", 0),
        "badges": session.get("badges", []),
        "badge_status": build_badge_status(session.get("total_points", 0), session.get("badges", [])),
        "next_badge": get_next_badge(session.get("total_points", 0)),
        "view_section": session.get("view_section", "home"),
        "open_chat_widget": session.get("open_chat_widget", False),
    }


def get_defaults():
    return {
        "input_type": session.get("input_type", "Text"),
        "text_option": session.get("text_option", "Type Text"),
        "learner_type": session.get("learner_type", "Visual"),
        "question_type": session.get("question_type", "MCQ"),
    }


def render_home(status_code=200):
    return (
        render_template(
            "index.html",
            defaults=get_defaults(),
            docx_available=docx is not None,
            **build_context(),
        ),
        status_code,
    )


@app.route("/")
def index():
    ensure_session_defaults()
    defaults = {
        "input_type": session.get("input_type", "Text"),
        "text_option": session.get("text_option", "Type Text"),
        "learner_type": session.get("learner_type", "Visual"),
        "question_type": session.get("question_type", "MCQ"),
    }
    return render_template("index.html", defaults=defaults, docx_available=docx is not None, **build_context())


@app.route("/process", methods=["POST"])
def process():
    ensure_session_defaults()
    session["view_section"] = "analysis"
    session["open_chat_widget"] = False
    input_type = request.form.get("input_type", "Text")
    learner_type = request.form.get("learner_type", "Visual")
    question_type = request.form.get("question_type", "MCQ")
    text_option = request.form.get("text_option", "Type Text")

    session["input_type"] = input_type
    session["text_option"] = text_option
    session["learner_type"] = learner_type
    session["question_type"] = question_type

    text_data = ""
    source_label = ""
    video_url = ""
    processed_topic = ""
    image_b64 = None
    image_mime = "image/png"

    try:
        if input_type == "Text":
            if text_option == "Type Text":
                text_data = request.form.get("text_input", "")
                source_label = "Typed text"
            elif text_option == "Upload PDF":
                pdf_file = request.files.get("pdf_file")
                if pdf_file and pdf_file.filename:
                    text_data = extract_pdf_text(pdf_file.read())
                    source_label = "PDF document"
            elif text_option == "Upload DOCX":
                doc_file = request.files.get("docx_file")
                if doc_file and doc_file.filename:
                    text_data = extract_docx_text(doc_file.read())
                    source_label = "DOCX document"

        elif input_type == "Video":
            url = request.form.get("youtube_url", "")
            video_id = extract_video_id(url)
            if not video_id:
                flash("Invalid YouTube URL.", "error")
                return render_home(400)

            video_url = url
            transcript_text, transcript_error = fetch_video_transcript_text(video_id)
            if not transcript_text:
                flash("Failed to fetch transcript for this video.", "error")
                if transcript_error:
                    flash(transcript_error, "error")
                return render_home(400)

            text_data = transcript_text
            source_label = "Video transcript"

        elif input_type == "Audio":
            audio_file = request.files.get("audio_file")
            if audio_file and audio_file.filename:
                extension = os.path.splitext(audio_file.filename)[1].lower()
                text_data = transcribe_audio_bytes(audio_file.read(), extension)
                source_label = "Audio transcript"

        elif input_type == "Image":
            image_file = request.files.get("image_file")
            if image_file and image_file.filename:
                image_bytes = image_file.read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                image_mime = image_file.mimetype or "image/png"
                analysis_text = analyze_image_topic(image_bytes)
                image_topic, image_notes = parse_image_analysis(analysis_text)
                text_data = image_notes
                source_label = f"Image topic: {image_topic}"
                processed_topic = image_topic
    except Exception as exc:
        flash(str(exc), "error")
        return render_home(500)

    normalized_text = sanitize_learning_text(text_data) or clean_text(text_data)
    if not normalized_text:
        flash("Please provide a valid input first.", "error")
        return render_home(400)

    summary = generate_summary(normalized_text, learner_type)
    prediction = predict(normalized_text)
    display_topic = processed_topic if input_type == "Image" else extract_topic_title(summary or normalized_text)
    study_pack = build_study_pack(summary, explicit_topic=display_topic)
    session["processed_text"] = normalized_text
    session["processed_source"] = source_label
    session["processed_input_type"] = input_type
    session["processed_video_url"] = video_url if input_type == "Video" else ""
    session["processed_topic"] = study_pack.get("topic", display_topic)
    session["video_search_topic"] = study_pack.get("video_query") or build_video_search_topic(
        session["processed_topic"],
        input_type,
    )
    session["related_video_url"] = ""
    if input_type != "Video":
        session["related_video_url"] = build_youtube_search_url(session["video_search_topic"])
    session["last_summary"] = summary
    session["prediction"] = prediction
    session["quiz_data"] = []
    session["quiz_topic"] = ""
    session["quiz_score"] = None
    session["quiz_reward"] = None
    session["broad_questions"] = []
    session["broad_topic"] = ""
    session["broad_results"] = []
    session["broad_score"] = None
    session["broad_reward"] = None
    session["chat_messages"] = []
    output_payload = {
        "processed_image_b64": image_b64,
        "processed_image_mime": image_mime if input_type == "Image" else "image/png",
        "study_pack": study_pack,
        "visual_assets": {},
        "generated_visuals": [],
        "generated_video": {},
        "visual_slides": [],
        "audio_b64": None,
        "audio_error": None,
    }

    if learner_type == "Visual":
        visual_assets, visual_slides = build_visual_package(summary, session["processed_topic"] or "Topic", study_pack)
        output_payload["visual_assets"] = visual_assets
        output_payload["visual_slides"] = visual_slides
        output_payload["generated_visuals"] = build_generated_visuals(
            session["processed_topic"] or "Topic",
            study_pack.get("visual_cards", []),
            limit=3,
        )
        output_payload["generated_video"] = build_generated_video(
            session["processed_topic"] or "Topic",
            study_pack.get("visual_cards", []),
        )
    elif learner_type == "Audio":
        try:
            output_payload["audio_b64"] = generate_audio_base64(study_pack.get("audio_script", summary))
        except Exception as exc:
            output_payload["audio_error"] = str(exc)

    save_server_output(output_payload)

    history = session.get("history", [])
    history.insert(
        0,
        {
            "created_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "source": source_label,
            "learner": learner_type,
            "topic": (session["processed_topic"] or normalized_text)[:90],
        },
    )
    session["history"] = history[:12]

    return render_home()


@app.route("/result")
def result():
    if not session.get("processed_text"):
        flash("Please process some content first.", "error")
        return render_home(400)
    return render_home()


@app.route("/generate_questions", methods=["POST"])
def generate_questions_route():
    ensure_session_defaults()
    session["view_section"] = "analysis"
    session["open_chat_widget"] = False
    if not session.get("processed_text"):
        flash("Please process some content first.", "error")
        return render_home(400)

    question_type = request.form.get("question_type", "MCQ")
    difficulty = request.form.get("difficulty", "Easy")
    num_questions = int(request.form.get("num_questions", 3))

    session["question_type"] = question_type
    session["difficulty"] = difficulty
    session["num_questions"] = num_questions
    session["quiz_score"] = None
    session["quiz_reward"] = None
    session["broad_results"] = []
    session["broad_score"] = None
    session["broad_reward"] = None

    normalized_text = session["processed_text"]
    study_pack = get_server_output().get("study_pack", build_fallback_study_pack(normalized_text, session.get("processed_topic", "")))
    topic_name = study_pack.get("topic", session.get("processed_topic", ""))

    if question_type == "MCQ":
        questions = []
        if not FAST_LOCAL_QUIZ:
            try:
                mcq_text = generate_mcq(normalized_text, difficulty, num_questions, topic_name, study_pack)
                questions = parse_mcq(mcq_text)
                if len(questions) != num_questions:
                    mcq_text = generate_mcq_retry(normalized_text, difficulty, num_questions, topic_name, study_pack)
                    questions = parse_mcq(mcq_text)
            except Exception:
                questions = []

        if len(questions) != num_questions:
            questions = build_fallback_mcq_questions(normalized_text, num_questions, topic_name, study_pack)
            flash("APSG generated a topic-focused quiz instantly.", "success")
        else:
            flash("MCQ quiz generated successfully.", "success")

        session["quiz_data"] = questions
        session["quiz_topic"] = topic_name
        session["broad_questions"] = []
        session["broad_topic"] = ""
    else:
        questions = []
        if not FAST_LOCAL_QUIZ:
            try:
                broad_text = generate_broad_questions(normalized_text, difficulty, num_questions, topic_name, study_pack)
                questions = parse_broad_questions(broad_text)
            except Exception:
                questions = []

        if len(questions) != num_questions:
            questions = build_fallback_broad_questions(normalized_text, num_questions, topic_name, study_pack)
            flash("APSG generated topic-focused descriptive questions instantly.", "success")
        else:
            flash("Broad questions generated successfully.", "success")

        session["broad_questions"] = questions
        session["broad_topic"] = topic_name
        session["quiz_data"] = []
        session["quiz_topic"] = ""

    return render_home()


@app.route("/submit_mcq", methods=["POST"])
def submit_mcq():
    ensure_session_defaults()
    session["view_section"] = "analysis"
    session["open_chat_widget"] = False
    quiz_data = session.get("quiz_data", [])
    difficulty = session.get("difficulty", "Easy")
    if not quiz_data:
        flash("Generate an MCQ quiz first.", "error")
        return render_home(400)

    score = 0
    for index, item in enumerate(quiz_data):
        selected_answer = request.form.get(f"answer_{index}", "").strip().upper()
        if selected_answer == item["answer"]:
            score += 1

    session["quiz_score"] = score
    session["quiz_reward"] = calculate_reward(score, difficulty)
    award_points(session["quiz_reward"])
    return render_home()


@app.route("/submit_broad", methods=["POST"])
def submit_broad():
    ensure_session_defaults()
    session["view_section"] = "analysis"
    session["open_chat_widget"] = False
    broad_questions = session.get("broad_questions", [])
    difficulty = session.get("difficulty", "Easy")
    if not broad_questions:
        flash("Generate broad questions first.", "error")
        return render_home(400)

    score = 0
    results = []
    topic_name = session.get("processed_topic", "")
    topic_summary = session.get("last_summary", "")
    for index, question in enumerate(broad_questions):
        answer = request.form.get(f"broad_answer_{index}", "")
        question_score = evaluate_broad_answer(question, answer, topic_name, topic_summary)
        score += question_score
        results.append({"question": question, "score": question_score, "answer": answer})

    session["broad_results"] = results
    session["broad_score"] = score
    session["broad_reward"] = calculate_reward(score, difficulty)
    award_points(session["broad_reward"])
    return render_home()


@app.route("/chat", methods=["POST"])
def chat():
    ensure_session_defaults()
    session["view_section"] = "analysis"
    session["open_chat_widget"] = True
    user_message = clean_text(request.form.get("chat_message", ""))
    if not user_message:
        flash("Enter a question for the chatbot.", "error")
        return render_home(400)

    history = session.get("chat_messages", [])[-8:]
    history_lines = []
    for item in history:
        role = "User" if item.get("role") == "user" else "Assistant"
        history_lines.append(f"{role}: {item.get('content', '')}")
    conversation_history = "\n".join(history_lines)

    topic_context = session.get("processed_text", "")
    topic_summary = session.get("last_summary", "")
    topic_block = ""
    if topic_context:
        topic_block = f"""
    Current study topic context:
    {topic_context}

    Current study summary:
    {topic_summary}
    """

    prompt = f"""
    You are APSG's academic assistant.
    Your first responsibility is to stay connected to the student's current study topic.
    If the user asks something unrelated, answer briefly and gently reconnect it to the current topic when possible.
    Keep answers clear, useful, professional, and easy to read.
    Use normal paragraphs and only use bullets when they genuinely improve clarity.
    Avoid overly casual language, filler, or repetitive phrasing.

    Recent conversation:
    {conversation_history or "No previous conversation."}
    {topic_block}

    Student question:
    {user_message}
    """

    try:
        reply = get_model_ai().generate_content(prompt).text.strip()
    except Exception:
        reply = "I could not generate a response right now. Please try again."

    messages = session.get("chat_messages", [])
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": reply})
    session["chat_messages"] = messages[-12:]
    return render_home()


@app.route("/contact", methods=["POST"])
def contact():
    ensure_session_defaults()
    session["view_section"] = "contact"
    session["open_chat_widget"] = False
    name = clean_text(request.form.get("name", ""))
    email = clean_text(request.form.get("email", ""))
    message = clean_text(request.form.get("message", ""))

    if not name or not email or not message:
        flash("Please fill in name, email, and message.", "error")
        return render_home(400)

    contact_messages = session.get("contact_messages", [])
    contact_messages.insert(
        0,
        {
            "name": name,
            "email": email,
            "message": message,
            "created_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        },
    )
    session["contact_messages"] = contact_messages[:10]
    flash("Your message has been recorded.", "success")
    return render_home()


@app.route("/reset", methods=["POST"])
def reset():
    previous_output_id = session.get("current_output_id", "")
    if previous_output_id in SERVER_OUTPUTS:
        SERVER_OUTPUTS.pop(previous_output_id, None)
    previous_defaults = {"history": [], "chat_messages": [], "contact_messages": [], "total_points": 0, "badges": [], "open_chat_widget": False, "current_output_id": ""}
    session.clear()
    session.update(previous_defaults)
    session["view_section"] = "home"
    flash("Session cleared.", "success")
    return render_home()


if __name__ == "__main__":
    app.run(debug=True)
