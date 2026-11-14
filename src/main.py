import asyncio
import atexit
import base64
import flet as ft
import glob
import html as html_module
import io
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx


# ---------------------------------------------------------------------------
# Local model configuration (llama.cpp / llama-server, OpenAI-compatible API)
# ---------------------------------------------------------------------------
# The app starts llama-server automatically on launch. Configure it with these
# environment variables (all optional):
#   LLAMA_AUTOSTART=0        don't start the server (connect to an existing one)
#   LLAMA_SERVER_BIN         path to llama-server (default: found on PATH)
#   LLAMA_MODEL_PATH         path to the .gguf model (default: auto-detected)
#   LLAMA_SERVER_ARGS        extra arguments, e.g. "-c 4096 -ngl 99"
#   LLAMA_START_TIMEOUT      seconds to wait for the model to load (default 180)
LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "qwen")
TEMPERATURE = float(os.environ.get("LLAMA_TEMPERATURE", "0.7"))
SYSTEM_PROMPT = os.environ.get(
    "LLAMA_SYSTEM_PROMPT",
    "You are HECRIG, a helpful, concise AI assistant running locally on the user's machine.",
)

LLAMA_AUTOSTART = os.environ.get("LLAMA_AUTOSTART", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "").strip()
LLAMA_SERVER_ARGS = os.environ.get("LLAMA_SERVER_ARGS", "").strip()
LLAMA_START_TIMEOUT = float(os.environ.get("LLAMA_START_TIMEOUT", "180"))
# Context size used when HECRIG starts the server itself. Larger contexts need
# more memory but leave room for attachments. Ignored if LLAMA_SERVER_ARGS
# already contains -c/--ctx-size.
LLAMA_CTX = int(os.environ.get("HECRIG_CTX", "8192"))

# Speed settings. By default HECRIG does NOT pass -ngl, which lets llama.cpp
# auto-fit as many layers to the GPU as fit in VRAM. Set HECRIG_NGL=99 to force
# full GPU offload, or HECRIG_NGL=0 for CPU only. HECRIG_PARALLEL=1 gives one
# chat the whole context instead of splitting it across slots.
#
# Flash attention is left at llama.cpp's default ("auto") on purpose: in some
# builds passing "-fa on" makes the argument parser swallow later flags, which
# silently disables GPU offload. Set HECRIG_FLASH_ATTN=on|off|auto to override.
LLAMA_GPU_LAYERS = os.environ.get("HECRIG_NGL", "").strip()
LLAMA_PARALLEL = os.environ.get("HECRIG_PARALLEL", "1").strip()
LLAMA_FLASH_ATTN = os.environ.get("HECRIG_FLASH_ATTN", "").strip().lower()

# Web access (tool calling). Enabled by default; set HECRIG_WEB_SEARCH=0 to
# disable. DuckDuckGo is used for search and needs no API key. Set TAVILY_API_KEY
# to use Tavily instead (better quality, requires a free API key).
WEB_SEARCH_ENABLED = os.environ.get("HECRIG_WEB_SEARCH", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
WEB_SEARCH_MAX_RESULTS = int(os.environ.get("HECRIG_SEARCH_RESULTS", "5"))
WEB_FETCH_MAX_CHARS = int(os.environ.get("HECRIG_FETCH_CHARS", "6000"))
MAX_TOOL_STEPS = int(os.environ.get("HECRIG_MAX_TOOL_STEPS", "4"))

# Persistent memory and learned skills. Stored as JSON in the user's local
# application data folder (override with HECRIG_DATA_DIR).
HECRIG_DATA_DIR = os.environ.get("HECRIG_DATA_DIR", "").strip()
MAX_MEMORIES = int(os.environ.get("HECRIG_MAX_MEMORIES", "50"))
MAX_SKILLS = int(os.environ.get("HECRIG_MAX_SKILLS", "50"))

# Attachments (images, PDFs, office documents). Text is extracted locally and,
# when the model has a vision projector (mmproj), images are sent to the model.
ATTACH_MAX_FILES = int(os.environ.get("HECRIG_MAX_ATTACHMENTS", "4"))
ATTACH_MAX_BYTES = int(os.environ.get("HECRIG_MAX_ATTACHMENT_MB", "40")) * 1024 * 1024
ATTACH_MAX_CHARS = int(os.environ.get("HECRIG_ATTACH_CHARS", "6000"))
ATTACH_TOTAL_CHARS = int(os.environ.get("HECRIG_ATTACH_TOTAL_CHARS", "12000"))
ATTACH_IMAGE_MAX_SIDE = int(os.environ.get("HECRIG_IMAGE_MAX_SIDE", "1024"))
ATTACH_MAX_PDF_PAGES = int(os.environ.get("HECRIG_PDF_MAX_PAGES", "30"))
ATTACH_MAX_OCR_PAGES = int(os.environ.get("HECRIG_OCR_MAX_PAGES", "6"))
ATTACH_MAX_IMAGES = int(os.environ.get("HECRIG_MAX_IMAGES", "4"))

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
}
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml",
    ".yaml", ".yml", ".html", ".htm", ".log", ".ini", ".cfg", ".conf",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sql", ".sh", ".ps1",
    ".bat", ".cmd", ".toml", ".tex", ".r", ".swift", ".kt",
}
DOCUMENT_EXTENSIONS = {
    ".pdf", ".docx", ".rtf", ".odt", ".ods", ".odp",
    ".xlsx", ".pptx",
}
ALLOWED_ATTACHMENT_EXTENSIONS = sorted(
    IMAGE_EXTENSIONS | TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS
)

# Set once we know whether the running llama-server has vision support.
VISION_STATE = {"available": False, "checked": False}

_parsed_url = urlparse(LLAMA_BASE_URL)
LLAMA_HOST = _parsed_url.hostname or "127.0.0.1"
LLAMA_PORT = _parsed_url.port or 8080

SERVER_LOG_PATH = os.path.join(tempfile.gettempdir(), "hecrig_llama_server.log")

_server_process = None
_server_lock = threading.Lock()
_atexit_registered = False


# ---------------------------------------------------------------------------
# llama-server process management
# ---------------------------------------------------------------------------
def _resolve_server_bin():
    """Return a usable llama-server path, or None if it can't be found."""
    if os.path.isfile(LLAMA_SERVER_BIN):
        return LLAMA_SERVER_BIN
    return shutil.which(LLAMA_SERVER_BIN)


def _find_model():
    """Locate a .gguf model: env var first, then common local folders."""
    if LLAMA_MODEL_PATH:
        return LLAMA_MODEL_PATH if os.path.isfile(LLAMA_MODEL_PATH) else ""

    search_dirs = [
        os.path.join(os.getcwd(), "models"),
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        matches = [
            path
            for path in sorted(glob.glob(os.path.join(directory, "*.gguf")))
            if not os.path.basename(path).lower().startswith("mmproj")
        ]
        if matches:
            return matches[0]

    # Fall back to models downloaded by huggingface_hub.
    hf_pattern = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
        "models--*",
        "snapshots",
        "*",
        "*.gguf",
    )
    matches = [
        path
        for path in sorted(glob.glob(hf_pattern))
        if not os.path.basename(path).lower().startswith("mmproj")
    ]
    if matches:
        # Prefer a quantized model over full-precision ones when several exist.
        preferred = [p for p in matches if "q4" in os.path.basename(p).lower()]
        return (preferred or matches)[0]
    return ""


def _find_mmproj(model_path: str = "") -> str:
    """Locate a multimodal projector (mmproj) so the model can see images."""
    configured = (
        os.environ.get("HECRIG_MMPROJ", "").strip()
        or os.environ.get("LLAMA_MMPROJ", "").strip()
    )
    if configured:
        return configured if os.path.isfile(configured) else ""

    candidates = []
    if model_path:
        candidates.append(
            os.path.join(os.path.dirname(os.path.abspath(model_path)), "mmproj*.gguf")
        )
    for directory in (
        os.path.join(os.getcwd(), "models"),
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
    ):
        if os.path.isdir(directory):
            candidates.append(os.path.join(directory, "mmproj*.gguf"))
    candidates.append(
        os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "huggingface",
            "hub",
            "models--*",
            "snapshots",
            "*",
            "mmproj*.gguf",
        )
    )
    for pattern in candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return ""


def _port_is_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _register_exit_handler():
    global _atexit_registered
    if _atexit_registered:
        return
    _atexit_registered = True
    atexit.register(_stop_server)


def start_llama_server():
    """Start llama-server if needed. Returns the Popen handle or None."""
    global _server_process

    with _server_lock:
        if _server_process is not None and _server_process.poll() is None:
            return _server_process

        exe = _resolve_server_bin()
        model = _find_model()
        if not exe or not model:
            return None

        args = [
            exe,
            "-m",
            model,
            "--host",
            LLAMA_HOST,
            "--port",
            str(LLAMA_PORT),
        ]
        extra_args = (
            shlex.split(LLAMA_SERVER_ARGS, posix=os.name != "nt")
            if LLAMA_SERVER_ARGS
            else []
        )
        has_ctx = any(
            arg in ("-c", "--ctx-size", "--ctx_size") for arg in extra_args
        )
        if LLAMA_CTX and not has_ctx:
            args.extend(["-c", str(LLAMA_CTX)])

        # Speed: keep one slot so a single chat owns the whole context and
        # offload every layer to the GPU.
        has_parallel = any(
            arg in ("-np", "--parallel", "--parallel-slots") for arg in extra_args
        )
        if LLAMA_PARALLEL and not has_parallel:
            args.extend(["-np", LLAMA_PARALLEL])
        has_ngl = any(
            arg in ("-ngl", "--gpu-layers", "--n-gpu-layers") for arg in extra_args
        )
        if LLAMA_GPU_LAYERS and not has_ngl:
            args.extend(["-ngl", LLAMA_GPU_LAYERS])
        # Only add -fa when the user asks for it: some builds mis-parse "-fa on"
        # and then ignore -ngl, which silently drops GPU offload.
        has_fa = any(arg in ("-fa", "--flash-attn") for arg in extra_args)
        if LLAMA_FLASH_ATTN and not has_fa:
            args.extend(["-fa", LLAMA_FLASH_ATTN])

        mmproj = _find_mmproj(model)
        if mmproj and "--mmproj" not in extra_args:
            args.extend(["--mmproj", mmproj])
            VISION_STATE.update({"available": True, "checked": True})
        args.extend(extra_args)

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW

        log_file = open(SERVER_LOG_PATH, "ab", buffering=0)
        try:
            _server_process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except OSError:
            return None
        finally:
            log_file.close()

        _register_exit_handler()
        return _server_process


def _stop_server():
    """Terminate the llama-server process this app started, if any."""
    global _server_process

    with _server_lock:
        proc = _server_process
        _server_process = None

    if proc is None or proc.poll() is not None:
        return

    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def _log_tail(max_chars: int = 700) -> str:
    try:
        with open(SERVER_LOG_PATH, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()[-max_chars:].strip()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Persistent memory and learned skills
# ---------------------------------------------------------------------------
def _data_dir() -> Path:
    if HECRIG_DATA_DIR:
        path = Path(HECRIG_DATA_DIR).expanduser()
    elif os.name == "nt":
        path = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "HECRIG"
    else:
        path = Path.home() / ".hecrig"
    path.mkdir(parents=True, exist_ok=True)
    return path


class _JsonListStore:
    """Small JSON-backed list store with atomic writes."""

    def __init__(self, path: Path, max_items: int):
        self.path = path
        self.max_items = max_items
        self._lock = threading.Lock()

    def _load(self) -> list:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return []
        items = payload.get("items") if isinstance(payload, dict) else None
        return items if isinstance(items, list) else []

    def _save(self, items: list) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"items": items}, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def all(self) -> list:
        with self._lock:
            return self._load()

    def clear(self) -> None:
        with self._lock:
            self._save([])


class _MemoryStore(_JsonListStore):
    def add(self, text: str) -> str:
        text = " ".join((text or "").split()).strip()
        if not text:
            return "Error: memory text was empty."
        text = text[:500]
        with self._lock:
            items = self._load()
            if any(
                item.get("text", "").casefold() == text.casefold() for item in items
            ):
                return f"Already remembered: {text}"
            now = datetime.now().isoformat(timespec="seconds")
            item = {
                "id": uuid.uuid4().hex[:8],
                "text": text,
                "created_at": now,
                "updated_at": now,
            }
            items.append(item)
            self._save(items[-self.max_items :])
        return f"Remembered [{item['id']}]: {text}"

    def forget(self, ref: str) -> str:
        ref = (ref or "").strip()
        if not ref:
            return "Error: no memory reference provided."
        with self._lock:
            items = self._load()
            target = next((i for i in items if i.get("id") == ref), None)
            if target is None:
                matches = [
                    i for i in items if ref.casefold() in i.get("text", "").casefold()
                ]
                if len(matches) == 1:
                    target = matches[0]
                elif len(matches) > 1:
                    return "Multiple memories match; use the id from the memory list."
            if target is None:
                return f"No saved memory found for: {ref}"
            items.remove(target)
            self._save(items)
        return f"Forgot: {target['text']}"

    def prompt_text(self) -> str:
        items = self.all()
        if not items:
            return "(no saved memories)"
        return "\n".join(f"- [{item['id']}] {item['text']}" for item in items)


class _SkillStore(_JsonListStore):
    def save(self, name: str, description: str, instructions: str) -> str:
        name = " ".join((name or "").split()).strip()[:80]
        description = " ".join((description or "").split()).strip()[:240]
        instructions = (instructions or "").strip()[:6000]
        if not name:
            return "Error: skill name is required."
        if not instructions:
            return "Error: skill instructions are required."
        with self._lock:
            items = self._load()
            existing = next(
                (i for i in items if i.get("name", "").casefold() == name.casefold()),
                None,
            )
            now = datetime.now().isoformat(timespec="seconds")
            if existing is not None:
                existing.update(
                    {
                        "name": name,
                        "description": description,
                        "instructions": instructions,
                        "updated_at": now,
                    }
                )
                action = "Updated"
                skill = existing
            else:
                skill = {
                    "id": uuid.uuid4().hex[:8],
                    "name": name,
                    "description": description,
                    "instructions": instructions,
                    "created_at": now,
                    "updated_at": now,
                }
                items.append(skill)
                action = "Saved"
            self._save(items[-self.max_items :])
        return f"{action} skill [{skill['id']}]: {name}\n{description}".strip()

    def _find(self, items: list, ref: str):
        ref = (ref or "").strip()
        if not ref:
            return None, "Error: no skill name provided."
        lowered = ref.casefold()
        for item in items:
            if item.get("id", "").casefold() == lowered:
                return item, None
        for item in items:
            if item.get("name", "").casefold() == lowered:
                return item, None
        matches = [i for i in items if lowered in i.get("name", "").casefold()]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, "Multiple skills match; be more specific."
        return None, f"No saved skill found for: {ref}"

    def use(self, ref: str) -> str:
        items = self.all()
        skill, error = self._find(items, ref)
        if error:
            return error
        return (
            f"Skill: {skill['name']}\n"
            f"{skill.get('description') or ''}\n\n"
            f"{skill['instructions']}"
        ).strip()

    def delete(self, ref: str) -> str:
        with self._lock:
            items = self._load()
            skill, error = self._find(items, ref)
            if error:
                return error
            items.remove(skill)
            self._save(items)
        return f"Deleted skill: {skill['name']}"

    def prompt_text(self) -> str:
        items = self.all()
        if not items:
            return "(no saved skills)"
        return "\n".join(
            f"- {item['name']}: {item.get('description') or '(no description)'}"
            for item in items
        )


DATA_DIR = _data_dir()
MEMORY_STORE = _MemoryStore(DATA_DIR / "memory.json", MAX_MEMORIES)
SKILL_STORE = _SkillStore(DATA_DIR / "skills.json", MAX_SKILLS)


async def remember_memory(text: str) -> str:
    return MEMORY_STORE.add(text)


async def forget_memory(ref: str) -> str:
    return MEMORY_STORE.forget(ref)


async def save_skill(name: str, description: str = "", instructions: str = "") -> str:
    return SKILL_STORE.save(name, description, instructions)


async def use_skill(name: str) -> str:
    return SKILL_STORE.use(name)


async def list_skills() -> str:
    items = SKILL_STORE.all()
    if not items:
        return "No skills saved yet."
    return "\n".join(
        f"- [{item['id']}] {item['name']}: "
        f"{item.get('description') or '(no description)'}"
        for item in items
    )


async def delete_skill(name: str) -> str:
    return SKILL_STORE.delete(name)


MEMORY_SKILL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a durable fact or preference about the user so it is "
                "remembered in future conversations. Use only for lasting "
                "information (name, timezone, preferences, projects), never "
                "for secrets like passwords or API keys."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The fact to remember, written clearly.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Delete a saved memory by its id or exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "string",
                        "description": "Memory id or text to forget.",
                    }
                },
                "required": ["memory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_skill",
            "description": (
                "Create or update a reusable skill (a repeatable procedure). "
                "Use this when the user teaches a workflow or says to learn "
                "how to do something. Always write clear step-by-step "
                "instructions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short skill name, e.g. 'Weekly report'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "One line describing when to use it.",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Detailed steps to follow.",
                    },
                },
                "required": ["name", "instructions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "use_skill",
            "description": (
                "Load the full instructions for a saved skill. Call this "
                "before performing a task that matches a saved skill."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name or id.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List all saved skills and their descriptions.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_skill",
            "description": "Delete a saved skill by name or id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name or id to delete.",
                    }
                },
                "required": ["name"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Internet access: web search + page reader + llama.cpp tool calling
# ---------------------------------------------------------------------------
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

WEB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live internet for current information. Use this for "
                "news, recent events, prices, weather, sports, software releases, "
                "or any fact that may have changed recently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Download a specific web page and return its readable text. Use "
                "this to read an article or verify details from a search result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full http:// or https:// URL to read.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


MUTATING_MEMORY_TOOLS = {"remember", "forget", "save_skill", "delete_skill"}


def build_tools(web_enabled: bool = True):
    tools = list(MEMORY_SKILL_TOOLS) + list(DOC_TOOLS)
    if web_enabled:
        tools.extend(WEB_TOOLS)
    return tools or None


class _DuckDuckGoParser(HTMLParser):
    """Extract result titles, URLs and snippets from DuckDuckGo HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").split()
        if "result__a" in classes:
            self.results.append(
                {"url": attrs.get("href", ""), "title": "", "snippet": ""}
            )
            self._in_title = True
        elif "result__snippet" in classes and self.results:
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a":
            self._in_title = False
            self._in_snippet = False

    def handle_data(self, data):
        if not self.results:
            return
        if self._in_title:
            self.results[-1]["title"] += data
        elif self._in_snippet:
            self.results[-1]["snippet"] += data


class _ReadableTextParser(HTMLParser):
    """Turn an HTML document into readable plain text."""

    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
    BLOCK = {
        "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
        "section", "article", "header", "blockquote", "pre", "table",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif not self._skip and tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            if self._skip:
                self._skip -= 1
        elif not self._skip and tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _clean_text(value: str) -> str:
    value = html_module.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    return re.sub(r"\n\s*\n+", "\n\n", value).strip()


def _strip_tool_call_text(text: str) -> str:
    """Remove raw <tool_call> syntax a model may print when tools are off."""
    return re.sub(r"<tool_call>.*?</tool_call>", "", text or "", flags=re.S).strip()


def _unwrap_ddg_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return url


def _is_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host or host == "localhost" or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )


def _format_search_results(query: str, results: list, max_results: int) -> str:
    if not results:
        return f'No web results found for "{query}".'
    lines = [f'Web search results for "{query}":', ""]
    for index, item in enumerate(results[:max_results], 1):
        title = _clean_text(item.get("title", "")) or "(no title)"
        url = item.get("url", "")
        snippet = _clean_text(item.get("snippet") or item.get("content") or "")
        lines.append(f"{index}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet[:500]}")
        lines.append("")
    lines.append(
        "Cite the source URLs you rely on. If these snippets are not enough, "
        "call fetch_url on the most relevant URL before answering."
    )
    return "\n".join(lines)


async def _duckduckgo_search(query: str, max_results: int) -> str:
    async with httpx.AsyncClient(
        timeout=20, follow_redirects=True, headers=_BROWSER_HEADERS
    ) as client:
        response = await client.get(
            "https://html.duckduckgo.com/html/", params={"q": query}
        )
        response.raise_for_status()
        parser = _DuckDuckGoParser()
        parser.feed(response.text)
        results = parser.results

        if not results:
            # Fall back to the lightweight endpoint if the HTML layout changes.
            lite = await client.get(
                "https://lite.duckduckgo.com/lite/", params={"q": query}
            )
            lite.raise_for_status()
            links = re.findall(
                r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                lite.text,
                re.S,
            )
            snippets = re.findall(
                r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', lite.text, re.S
            )
            results = [
                {
                    "url": _unwrap_ddg_url(url),
                    "title": _clean_text(title),
                    "snippet": _clean_text(snippets[i]) if i < len(snippets) else "",
                }
                for i, (url, title) in enumerate(links)
            ]

    for item in results:
        item["url"] = _unwrap_ddg_url(item["url"])
    return _format_search_results(query, results, max_results)


async def _tavily_search(query: str, max_results: int) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
    return _format_search_results(query, payload.get("results") or [], max_results)


async def web_search(query: str, max_results: int = None) -> str:
    """Search the live web. Returns formatted text for the model."""
    query = (query or "").strip()
    if not query:
        return "Error: the search query was empty."
    max_results = max_results or WEB_SEARCH_MAX_RESULTS
    try:
        if TAVILY_API_KEY:
            return await _tavily_search(query, max_results)
        return await _duckduckgo_search(query, max_results)
    except Exception as ex:
        return f"Web search failed: {type(ex).__name__}: {ex}"


async def fetch_url(url: str) -> str:
    """Download a public web page and return readable text for the model."""
    url = (url or "").strip()
    if not _is_public_url(url):
        return "Error: only public http:// or https:// URLs are allowed."

    content_type = ""
    raw = b""
    try:
        async with httpx.AsyncClient(
            timeout=25, follow_redirects=True, headers=_BROWSER_HEADERS
        ) as client:
            async with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    return f"Error: HTTP {response.status_code} while fetching {url}"
                content_type = (response.headers.get("content-type") or "").lower()
                total = 0
                chunks = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= 2_000_000:
                        break
                raw = b"".join(chunks)
                encoding = response.encoding or "utf-8"
    except Exception as ex:
        return f"Error fetching {url}: {type(ex).__name__}: {ex}"

    if content_type and not any(
        kind in content_type for kind in ("html", "text/", "xml", "json")
    ):
        return f"Error: unsupported content type: {content_type}"

    text = raw.decode(encoding, "ignore")
    if "html" in content_type or "<html" in text[:500].lower():
        parser = _ReadableTextParser()
        parser.feed(text)
        text = _clean_text("".join(parser.parts))
    else:
        text = _clean_text(text)

    if not text:
        return f"The page at {url} had no readable text."
    if len(text) > WEB_FETCH_MAX_CHARS:
        text = text[:WEB_FETCH_MAX_CHARS] + "\n\n[... page truncated ...]"
    return f"Content of {url}:\n\n{text}"


async def stream_chat_completion(messages, tools=None, on_delta=None):
    """Stream one chat completion, returning (content, tool_calls, finish_reason)."""
    payload = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": TEMPERATURE,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    content_parts = []
    tool_calls = {}
    finish_reason = None

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{LLAMA_BASE_URL}/v1/chat/completions", json=payload
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", "ignore")[:500]
                raise RuntimeError(f"HTTP {response.status_code}: {body}")
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}

                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)
                    if on_delta:
                        on_delta("".join(content_parts))

                for call in delta.get("tool_calls") or []:
                    index = call.get("index")
                    if index is None:
                        index = len(tool_calls)
                    slot = tool_calls.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if call.get("id"):
                        slot["id"] = call["id"]
                    function = call.get("function") or {}
                    if function.get("name") and not slot["function"]["name"]:
                        slot["function"]["name"] = function["name"]
                    if function.get("arguments"):
                        slot["function"]["arguments"] += function["arguments"]

    calls = [tool_calls[index] for index in sorted(tool_calls)]
    for index, call in enumerate(calls):
        if not call["id"]:
            call["id"] = f"call_{index}"
    return "".join(content_parts), calls, finish_reason


def build_system_prompt(web_enabled: bool = True) -> str:
    now = datetime.now().astimezone()
    today = now.strftime("%A, %B %d, %Y at %H:%M %Z")
    prompt = SYSTEM_PROMPT + f"\n\nCurrent date and time: {today}."
    if web_enabled:
        prompt += (
            " You have real-time internet access through the web_search and "
            "fetch_url tools. Use web_search whenever the user asks about news, "
            "current events, prices, weather, sports, software releases, or any "
            "fact that may have changed recently. Cite the source URLs you use. "
            "Use fetch_url when you need to read a page in detail."
        )
    prompt += (
        "\n\nUsers can attach images, PDFs, and documents in the app. Their "
        "contents appear in the conversation: images may be provided directly "
        "for you to look at, and documents are converted to text. You can also "
        "call read_document with a local file path to read a file yourself. "
        "When an image or scanned page is provided, describe what you actually "
        "see and quote any visible text."
        "\n\n## Saved memories\n"
        + MEMORY_STORE.prompt_text()
        + "\n\n## Saved skills\n"
        + SKILL_STORE.prompt_text()
        + "\n\nUse the remember tool for durable facts or preferences the user "
        "shares (name, timezone, likes/dislikes, ongoing projects). Never save "
        "passwords, API keys, or other secrets. Use save_skill when the user "
        "teaches a repeatable procedure or asks you to learn a workflow; write "
        "clear step-by-step instructions. Before starting a task, check the "
        "saved skills above and call use_skill when one matches. Use forget or "
        "delete_skill only when the user asks. When the user asks you to "
        "remember or learn something, actually call the tool instead of only "
        "saying that you will."
    )
    return prompt


# ---------------------------------------------------------------------------
# Attachments: images, PDFs, and office documents
# ---------------------------------------------------------------------------
def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _pil_to_data_url(image, max_side: int = None) -> str:
    from PIL import Image

    max_side = max_side or ATTACH_IMAGE_MAX_SIDE
    rgb = image.convert("RGB")
    rgb.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _pil_to_thumb(image, side: int = 64) -> str:
    from PIL import Image

    rgb = image.convert("RGB")
    rgb.thumbnail((side, side), Image.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _winrt_ocr_png(data: bytes) -> str:
    """OCR a PNG using the OCR engine built into Windows 10/11."""
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream)
    writer.write_bytes(data)
    await writer.store_async()
    await writer.flush_async()
    stream.seek(0)
    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()
    engine = OcrEngine.try_create_from_user_profile_languages()
    if engine is None:
        return ""
    result = await engine.recognize_async(bitmap)
    return "\n".join(line.text for line in result.lines).strip()


def _windows_ocr(image) -> str:
    if os.name != "nt":
        return ""
    try:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return _run_coro_sync(_winrt_ocr_png(buffer.getvalue()))
    except Exception:
        return ""


_rapidocr_engine = None


def _get_rapidocr():
    global _rapidocr_engine
    if _rapidocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _rapidocr_engine = RapidOCR()
    return _rapidocr_engine


def _ocr_pil(image) -> str:
    """Extract text from a PIL image using whatever OCR engine is available."""
    text = _windows_ocr(image)
    if text:
        return text
    try:
        import pytesseract

        return (pytesseract.image_to_string(image) or "").strip()
    except Exception:
        pass
    try:
        import numpy as np

        result, _ = _get_rapidocr()(np.array(image.convert("RGB")))
        if result:
            return "\n".join(str(line[1]) for line in result).strip()
    except Exception:
        pass
    return ""


def _extract_pdf(path: Path, prefer_text: bool = False):
    """Return (text, page_image_urls, note, error) for a PDF."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return (
            "",
            [],
            "",
            "PDF reading needs pypdfium2. Install it with: pip install pypdfium2",
        )

    document = pdfium.PdfDocument(str(path))
    chunks = []
    images = []
    notes = []
    try:
        total = len(document)
        page_count = min(total, ATTACH_MAX_PDF_PAGES)
        scanned_pages = 0
        for index in range(page_count):
            page = document[index]
            try:
                text = (page.get_textpage().get_text_range() or "").strip()
            except Exception:
                text = ""
            if len(text) >= 20:
                chunks.append(f"--- Page {index + 1} ---\n{text}")
                continue
            if scanned_pages >= ATTACH_MAX_OCR_PAGES:
                continue
            scanned_pages += 1
            try:
                rendered = page.render(scale=2).to_pil()
            except Exception:
                continue
            if VISION_STATE["available"] and not prefer_text:
                images.append(_pil_to_data_url(rendered))
                chunks.append(f"--- Page {index + 1} (image) ---")
            else:
                ocr = _ocr_pil(rendered)
                if ocr:
                    chunks.append(f"--- Page {index + 1} (OCR) ---\n{ocr}")
        if total > page_count:
            notes.append(f"Read the first {page_count} of {total} pages.")
        if not chunks:
            notes.append("No readable text was found in this PDF.")
        elif images:
            notes.append(f"{len(images)} scanned page(s) were attached as images.")
        return "\n\n".join(chunks), images, " ".join(notes), ""
    finally:
        document.close()


def _extract_docx(path: Path):
    with zipfile.ZipFile(path) as archive:
        if "word/document.xml" not in archive.namelist():
            return "", "This file is not a valid .docx document."
        xml = archive.read("word/document.xml").decode("utf-8", "ignore")
    xml = xml.replace("</w:tc>", " | ")
    xml = xml.replace("</w:tr>", "\n")
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    xml = xml.replace("</w:p>", "\n")
    text = _clean_text(html_module.unescape(re.sub(r"<[^>]+>", "", xml)))
    return text, ""


def _extract_xlsx(path: Path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        shared = []
        if "xl/sharedStrings.xml" in names:
            sheet_xml = archive.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
            for item in re.findall(r"<si>(.*?)</si>", sheet_xml, re.S):
                shared.append(html_module.unescape(re.sub(r"<[^>]+>", "", item)))
        workbook = ""
        if "xl/workbook.xml" in names:
            workbook = archive.read("xl/workbook.xml").decode("utf-8", "ignore")
        sheet_names = re.findall(r'<sheet[^>]*name="([^"]+)"', workbook)
        sheets = sorted(
            (name for name in names if re.match(r"xl/worksheets/sheet\d+\.xml$", name)),
            key=lambda name: int(re.search(r"(\d+)", name.rsplit("/", 1)[-1]).group(1)),
        )
        lines = []
        for index, sheet in enumerate(sheets[:10]):
            title = sheet_names[index] if index < len(sheet_names) else sheet
            lines.append(f"## Sheet: {title}")
            sheet_xml = archive.read(sheet).decode("utf-8", "ignore")
            for row in re.findall(r"<row[^>]*>(.*?)</row>", sheet_xml, re.S):
                cells = []
                for attrs, body in re.findall(r"<c([^>]*)>(.*?)</c>", row, re.S):
                    kind = re.search(r't="([^"]+)"', attrs)
                    kind = kind.group(1) if kind else ""
                    value = re.search(r"<v>(.*?)</v>", body, re.S)
                    if kind == "s" and value:
                        try:
                            cells.append(shared[int(value.group(1))].strip())
                        except (ValueError, IndexError):
                            cells.append("")
                    elif kind == "inlineStr":
                        cells.append(
                            html_module.unescape(
                                re.sub(r"<[^>]+>", "", body)
                            ).strip()
                        )
                    elif value:
                        cells.append(html_module.unescape(value.group(1)).strip())
                    else:
                        cells.append("")
                line = " | ".join(cells).rstrip(" |")
                if line.strip(" |"):
                    lines.append(line)
        return "\n".join(lines), ""


def _extract_pptx(path: Path):
    with zipfile.ZipFile(path) as archive:
        slides = sorted(
            (
                name
                for name in archive.namelist()
                if re.match(r"ppt/slides/slide\d+\.xml$", name)
            ),
            key=lambda name: int(re.search(r"(\d+)", name.rsplit("/", 1)[-1]).group(1)),
        )
        chunks = []
        for index, slide in enumerate(slides[:40], 1):
            xml = archive.read(slide).decode("utf-8", "ignore")
            xml = re.sub(r"</a:p>", "\n", xml)
            text = _clean_text(html_module.unescape(re.sub(r"<[^>]+>", "", xml)))
            if text:
                chunks.append(f"--- Slide {index} ---\n{text}")
        return "\n\n".join(chunks), ""


def _extract_odf(path: Path):
    with zipfile.ZipFile(path) as archive:
        if "content.xml" not in archive.namelist():
            return "", "This file is not a valid OpenDocument file."
        xml = archive.read("content.xml").decode("utf-8", "ignore")
    xml = re.sub(r"</text:(?:p|h)>", "\n", xml)
    xml = re.sub(r"<text:tab[^>]*/>", "\t", xml)
    text = _clean_text(html_module.unescape(re.sub(r"<[^>]+>", "", xml)))
    return text, ""


def _extract_rtf(path: Path):
    raw = path.read_bytes().decode("latin-1", "ignore")
    raw = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), raw)
    raw = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) % 65536), raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", raw)
    raw = raw.replace("{", "").replace("}", "")
    return _clean_text(raw), ""


def _extract_plain(path: Path, ext: str):
    raw = path.read_bytes()
    text = ""
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", "ignore")
    if ext in (".html", ".htm", ".xml"):
        try:
            parser = _ReadableTextParser()
            parser.feed(text)
            text = "".join(parser.parts)
        except Exception:
            pass
    return _clean_text(text), ""


def extract_attachment(path, display_name=None, prefer_text: bool = False) -> dict:
    """Read an image, PDF, or document and return text and/or image data."""
    path = Path(path)
    name = display_name or path.name
    ext = path.suffix.lower()
    attachment = {
        "name": name,
        "kind": "file",
        "text": "",
        "images": [],
        "thumb": "",
        "note": "",
        "error": None,
        "size": 0,
        "pages": None,
    }
    try:
        size = path.stat().st_size
    except OSError as ex:
        attachment["error"] = f"Could not open {name}: {ex}"
        return attachment
    attachment["size"] = size
    if size > ATTACH_MAX_BYTES:
        attachment["error"] = (
            f"{name} is too large ({_human_size(size)}); "
            f"the limit is {_human_size(ATTACH_MAX_BYTES)}."
        )
        return attachment

    note = ""
    try:
        if ext in IMAGE_EXTENSIONS:
            attachment["kind"] = "image"
            try:
                from PIL import Image

                with Image.open(path) as image:
                    image.load()
                    rgb = image.convert("RGB")
                    attachment["thumb"] = _pil_to_thumb(rgb)
                    if VISION_STATE["available"] and not prefer_text:
                        attachment["images"] = [_pil_to_data_url(rgb)]
                    else:
                        attachment["text"] = _ocr_pil(rgb)
            except Exception as ex:
                attachment["error"] = f"Could not read image: {ex}"
                return attachment
            if not attachment["images"] and not attachment["text"].strip():
                attachment["error"] = (
                    "No text was found in this image. Enable vision support "
                    "(mmproj) or install an OCR engine to scan images."
                )
        elif ext == ".pdf":
            attachment["kind"] = "pdf"
            text, images, note, error = _extract_pdf(path, prefer_text)
            attachment["text"], attachment["images"] = text, images
            if error:
                attachment["error"] = error
            try:
                import pypdfium2 as pdfium

                document = pdfium.PdfDocument(str(path))
                attachment["pages"] = len(document)
                if images:
                    attachment["thumb"] = _pil_to_thumb(
                        document[0].render(scale=1).to_pil()
                    )
                document.close()
            except Exception:
                pass
        elif ext == ".docx":
            attachment["kind"] = "document"
            attachment["text"], note = _extract_docx(path)
        elif ext == ".xlsx":
            attachment["kind"] = "spreadsheet"
            attachment["text"], note = _extract_xlsx(path)
        elif ext == ".pptx":
            attachment["kind"] = "presentation"
            attachment["text"], note = _extract_pptx(path)
        elif ext in (".odt", ".ods", ".odp"):
            attachment["kind"] = "document"
            attachment["text"], note = _extract_odf(path)
        elif ext == ".rtf":
            attachment["kind"] = "document"
            attachment["text"], note = _extract_rtf(path)
        elif ext in TEXT_EXTENSIONS:
            attachment["kind"] = "text"
            attachment["text"], note = _extract_plain(path, ext)
        else:
            attachment["error"] = (
                f"Unsupported file type: {ext or '(none)'}. "
                "Try an image, PDF, DOCX, XLSX, PPTX, ODT, RTF, or text file."
            )
            return attachment
    except Exception as ex:
        attachment["error"] = f"Could not read {name}: {type(ex).__name__}: {ex}"
        return attachment

    text = (attachment["text"] or "").strip()
    if len(text) > ATTACH_MAX_CHARS:
        attachment["text"] = text[:ATTACH_MAX_CHARS] + "\n[... truncated ...]"
        note = (note + " Text was truncated.").strip()
    attachment["note"] = note
    return attachment


async def extract_attachment_async(
    path, display_name=None, prefer_text: bool = False
) -> dict:
    return await asyncio.to_thread(
        extract_attachment, path, display_name, prefer_text
    )


def format_attachments_for_prompt(attachments, max_total: int = None) -> str:
    if not attachments:
        return ""
    max_total = max_total or ATTACH_TOTAL_CHARS
    blocks = []
    used = 0
    for attachment in attachments:
        name = attachment.get("name", "file")
        kind = attachment.get("kind", "file")
        body = (attachment.get("text") or "").strip()
        if attachment.get("images"):
            body = (
                body
                + f"\n({len(attachment['images'])} page(s) are attached as images.)"
            ).strip()
        if attachment.get("note"):
            body += f"\nNote: {attachment['note']}"
        remaining = max_total - used
        if remaining <= 0:
            blocks.append(f"[Attached file: {name} ({kind})]\n(skipped: limit reached)")
            continue
        if len(body) > remaining:
            body = body[:remaining] + "\n[... truncated ...]"
        used += len(body)
        blocks.append(f"[Attached file: {name} ({kind})]\n{body}\n[End of {name}]")
    return "\n\n".join(blocks)


def attachment_images(attachments) -> list:
    images = []
    for attachment in attachments or []:
        images.extend(attachment.get("images") or [])
    return images[:ATTACH_MAX_IMAGES]


async def read_document(path: str) -> str:
    """Read a local image, PDF, or document for the model."""
    raw = (path or "").strip().strip("\"'")
    if not raw:
        return "Error: no file path was provided."
    expanded = os.path.expandvars(os.path.expanduser(raw))
    target = Path(expanded)
    if not target.is_absolute():
        target = Path.cwd() / target
    if not target.exists():
        return f"Error: file not found: {target}"
    if target.is_dir():
        return f"Error: {target} is a folder, not a file."
    if target.suffix.lower() not in ALLOWED_ATTACHMENT_EXTENSIONS:
        return f"Error: unsupported file type '{target.suffix}'."
    attachment = await extract_attachment_async(target, prefer_text=True)
    if attachment.get("error"):
        return f"Error reading {target.name}: {attachment['error']}"
    body = (attachment.get("text") or "").strip()
    if not body:
        return f"No readable text was found in {target.name}."
    if len(body) > ATTACH_MAX_CHARS:
        body = body[:ATTACH_MAX_CHARS] + "\n[... truncated ...]"
    return f"Contents of {target.name}:\n\n{body}"


async def _detect_server_vision(timeout: float = 3.0) -> bool:
    """Ask llama-server whether the loaded model supports vision."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{LLAMA_BASE_URL}/props")
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return False
    modalities = payload.get("modalities") or {}
    return bool(modalities.get("vision"))


DOC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Read a local image, PDF, or document (DOCX, XLSX, PPTX, ODT, "
                "RTF, or text) from this computer. Use it when the user gives a "
                "file path or asks about a file that was not attached."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Full path to the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
PRIMARY = "#6366F1"
PRIMARY_FAINT = "#E0E7FF"
STOP_RED = "#EF4444"

BG_DARK = "#0F172A"
SURFACE_DARK = "#1E293B"
CARD_DARK = "#334155"
TEXT_DARK = "#F8FAFC"
TEXT_MUTED_DARK = "#94A3B8"

BG_LIGHT = "#F8FAFC"
SURFACE_LIGHT = "#FFFFFF"
CARD_LIGHT = "#F1F5F9"
TEXT_LIGHT = "#0F172A"
TEXT_MUTED_LIGHT = "#64748B"

BORDER_RADIUS = 16


def tones(dark: bool):
    if dark:
        return {
            "bg": BG_DARK,
            "surface": SURFACE_DARK,
            "card": CARD_DARK,
            "text": TEXT_DARK,
            "text_muted": TEXT_MUTED_DARK,
            "user_bubble": PRIMARY,
            "user_text": "#FFFFFF",
            "ai_bubble": CARD_DARK,
            "ai_text": TEXT_DARK,
            "border": "#334155",
            "item_active": PRIMARY_FAINT,
            "item_active_text": PRIMARY,
        }
    return {
        "bg": BG_LIGHT,
        "surface": SURFACE_LIGHT,
        "card": CARD_LIGHT,
        "text": TEXT_LIGHT,
        "text_muted": TEXT_MUTED_LIGHT,
        "user_bubble": PRIMARY,
        "user_text": "#FFFFFF",
        "ai_bubble": CARD_LIGHT,
        "ai_text": TEXT_LIGHT,
        "border": "#E2E8F0",
        "item_active": "#E0E7FF",
        "item_active_text": PRIMARY,
    }


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main(page: ft.Page):
    page.title = "HECRIG — AI Assistant"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.spacing = 0
    page.window_width = 1100
    page.window_height = 760
    page.window_min_width = 640
    page.window_min_height = 480

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    chats = []
    current_chat_index = None
    is_streaming = False
    current_theme_dark = True
    web_state = {"enabled": WEB_SEARCH_ENABLED}
    pending_attachments = []
    stream_state = {"task": None}
    # The chat input stays locked until llama-server is up and answering.
    model_state = {"ready": False}
    INPUT_HINT = "Message HECRIG..."
    LOADING_HINT = "Waiting for the model to load…"

    # Drives the "HECRIG is writing" animation. The loop only touches the one
    # control that is currently on screen, so it stays cheap during streaming.
    WRITING_ANIM = {
        "active": False,
        "frame": 0,
        "dots": None,
        "cursor": None,
        "base": "",
    }

    # ------------------------------------------------------------------
    # Refs
    # ------------------------------------------------------------------
    chat_list = ft.ListView(expand=True, spacing=8, auto_scroll=True)
    history_list = ft.ListView(expand=True, spacing=4)
    input_field = ft.TextField()
    send_btn = ft.IconButton()
    theme_switch = ft.Switch()

    status_dot = ft.Text("●", size=11, color="#22C55E", weight=ft.FontWeight.W_600)
    status_text = ft.Text("online", size=11, color="#22C55E", weight=ft.FontWeight.W_600)
    status_badge = ft.Container(
        content=ft.Row(controls=[status_dot, status_text], spacing=4, tight=True),
        padding=ft.Padding.symmetric(horizontal=8, vertical=2),
        bgcolor="#166534",
        border_radius=12,
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def palette():
        return tones(current_theme_dark)

    def _apply_status(label: str, color: str, bgcolor: str):
        status_dot.color = color
        status_text.value = label
        status_text.color = color
        status_badge.bgcolor = bgcolor
        try:
            status_badge.update()
        except Exception:
            pass

    def set_status(online: bool):
        if online:
            _apply_status("online", "#22C55E", "#166534")
        else:
            _apply_status("offline", "#F87171", "#7F1D1D")

    def set_status_starting():
        _apply_status("starting…", "#FBBF24", "#78350F")

    async def _is_healthy(timeout: float = 2.0) -> bool:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{LLAMA_BASE_URL}/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def _wait_for_server(timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await _is_healthy(1.5):
                return True
            proc = _server_process
            if proc is not None and proc.poll() is not None:
                return False
            await asyncio.sleep(0.75)
        return False

    def _startup_error(message: str):
        add_ai_message(message)

    async def _refresh_vision_state():
        VISION_STATE["available"] = await _detect_server_vision()
        VISION_STATE["checked"] = True

    async def ensure_server():
        """Start llama-server on launch (or connect to one already running)."""
        if await _is_healthy(2.0):
            set_status(True)
            await _refresh_vision_state()
            return

        if not LLAMA_AUTOSTART:
            set_status(False)
            return

        # Something is already listening on the port (perhaps still loading):
        # wait for it instead of launching a second server.
        if _port_is_open(LLAMA_HOST, LLAMA_PORT):
            set_status_starting()
            if await _wait_for_server(LLAMA_START_TIMEOUT):
                set_status(True)
                await _refresh_vision_state()
            else:
                set_status(False)
                _startup_error(
                    "⚠️ A server is listening on "
                    f"{LLAMA_HOST}:{LLAMA_PORT} but never became ready.\n\n"
                    "Check that process, or stop it so HECRIG can start its own."
                )
            return

        exe = _resolve_server_bin()
        if not exe:
            set_status(False)
            _startup_error(
                "⚠️ llama-server was not found.\n\n"
                "Install llama.cpp, or set LLAMA_SERVER_BIN to the full path of "
                "the llama-server executable."
            )
            return

        model = _find_model()
        if not model:
            set_status(False)
            if LLAMA_MODEL_PATH:
                message = (
                    "⚠️ The model file was not found:\n\n"
                    f"{LLAMA_MODEL_PATH}\n\n"
                    "Check the LLAMA_MODEL_PATH environment variable."
                )
            else:
                message = (
                    "⚠️ No GGUF model found.\n\n"
                    "Set LLAMA_MODEL_PATH to your model file, for example:\n"
                    "LLAMA_MODEL_PATH=C:\\models\\qwen.gguf"
                )
            _startup_error(message)
            return

        set_status_starting()
        proc = start_llama_server()
        if proc is None:
            set_status(False)
            _startup_error(
                "⚠️ llama-server could not be started.\n\n"
                f"Log: {SERVER_LOG_PATH}"
            )
            return

        if await _wait_for_server(LLAMA_START_TIMEOUT):
            set_status(True)
            await _refresh_vision_state()
            return

        set_status(False)
        tail = _log_tail()
        detail = f"\n\nLast log lines:\n{tail}" if tail else ""
        _startup_error(
            "⚠️ The local model did not finish loading in time.\n\n"
            f"Model: {model}\n"
            f"Log: {SERVER_LOG_PATH}{detail}"
        )

    def update_theme_colors():
        p = palette()
        page.bgcolor = p["bg"]
        sidebar.bgcolor = p["surface"]
        header.bgcolor = p["surface"]
        input_bar.bgcolor = p["surface"]
        input_bar.border = ft.Border.all(1, p["border"])
        input_field.bgcolor = p["card"]
        input_field.border_color = p["border"]
        input_field.color = p["text"]
        input_field.cursor_color = p["text"]
        web_btn.icon_color = PRIMARY if web_state["enabled"] else p["text_muted"]
        attach_btn.icon_color = p["text_muted"]
        attach_status.color = p["text_muted"]
        theme_switch.label = "Light mode" if current_theme_dark else "Dark mode"
        page.update()
        refresh_messages()
        refresh_history()

    def current_chat():
        if current_chat_index is None or not (0 <= current_chat_index < len(chats)):
            return None
        return chats[current_chat_index]

    def set_current_chat(index: int):
        nonlocal current_chat_index
        current_chat_index = index
        refresh_messages()
        refresh_history()

    def start_new_chat():
        chat = current_chat()
        if chat and chat["messages"]:
            chat["title"] = chat_title_from_messages(chat["messages"])
            chat["updated_at"] = datetime.now()
        chats.append({
            "id": len(chats) + 1,
            "title": "New chat",
            "messages": [],
            "updated_at": datetime.now(),
        })
        set_current_chat(len(chats) - 1)

    def switch_chat(index: int):
        chat = current_chat()
        if chat and chat["messages"]:
            chat["title"] = chat_title_from_messages(chat["messages"])
            chat["updated_at"] = datetime.now()
        set_current_chat(index)

    def chat_title_from_messages(messages: list) -> str:
        for msg in messages:
            if msg["role"] == "user":
                text = (msg.get("text") or "").strip()
                if text:
                    return text[:40] + ("…" if len(text) > 40 else "")
                attachments = msg.get("attachments") or []
                if attachments:
                    return attachments[0].get("name", "Attachment")[:40]
        return "New chat"

    def refresh_messages():
        chat_list.controls.clear()
        WRITING_ANIM["dots"] = None
        WRITING_ANIM["cursor"] = None
        WRITING_ANIM["base"] = ""
        p = palette()
        chat = current_chat()
        messages = chat["messages"] if chat else []
        for msg in messages:
            chat_list.controls.append(
                message_bubble(
                    msg.get("text", ""),
                    msg["role"] == "user",
                    msg.get("streaming", False),
                    p,
                    msg.get("attachments"),
                )
            )
        page.update()

    def refresh_history():
        history_list.controls.clear()
        p = palette()

        for idx, chat in enumerate(chats):
            is_active = (idx == current_chat_index)
            preview = ""
            for msg in chat["messages"]:
                if msg["role"] == "user":
                    preview = msg["text"][:38] + ("…" if len(msg["text"]) > 38 else "")
                    break
            time_str = chat["updated_at"].strftime("%H:%M") if chat.get("updated_at") else ""

            item = ft.Container(
                content=ft.Row(
                    controls=[
                        ft.Container(
                            content=ft.Icon(
                                ft.Icons.SMART_DISPLAY,
                                size=16,
                                color=p["item_active_text"] if is_active else p["text_muted"],
                            ),
                            width=32,
                            height=32,
                            border_radius=16,
                            bgcolor=p["item_active"] if is_active else p["card"],
                            alignment=ft.Alignment.CENTER,
                        ),
                        ft.Column(
                            controls=[
                                ft.Text(
                                    chat["title"],
                                    size=13,
                                    weight=ft.FontWeight.W_600 if is_active else ft.FontWeight.W_400,
                                    color=p["item_active_text"] if is_active else p["text"],
                                    no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                ),
                                ft.Text(
                                    (preview if preview else "Start conversation") + "  ·  " + time_str,
                                    size=11,
                                    color=p["text_muted"],
                                    no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                ),
                            ],
                            spacing=2,
                            tight=True,
                            expand=True,
                        ),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10,
                ),
                padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                border_radius=ft.BorderRadius.all(12),
                bgcolor=p["item_active"] if is_active else p["card"],
                border=ft.Border.all(1, p["border"]),
                margin=ft.Margin.symmetric(horizontal=12),
                on_click=lambda e, idx=idx: switch_chat(idx),
                ink=True,
            )
            history_list.controls.append(item)

        history_list.update()
        page.update()

    async def writing_animation_loop():
        """Animate the dots / blinking cursor while a reply is streaming."""
        while WRITING_ANIM["active"]:
            WRITING_ANIM["frame"] += 1
            frame = WRITING_ANIM["frame"]

            dots = WRITING_ANIM["dots"]
            if dots is not None:
                dots.value = "." * (frame % 4)
                try:
                    dots.update()
                except Exception:
                    pass

            cursor = WRITING_ANIM["cursor"]
            if cursor is not None:
                cursor.value = WRITING_ANIM["base"] + (" ▋" if frame % 2 == 0 else "")
                try:
                    cursor.update()
                except Exception:
                    pass

            await asyncio.sleep(0.4)

    def start_writing_animation():
        if WRITING_ANIM["active"]:
            return
        WRITING_ANIM["active"] = True
        WRITING_ANIM["frame"] = 0
        page.run_task(writing_animation_loop)

    def stop_writing_animation():
        WRITING_ANIM["active"] = False
        WRITING_ANIM["dots"] = None
        WRITING_ANIM["cursor"] = None

    async def handle_stop(e=None):
        """Abort the reply that is currently streaming."""
        task = stream_state.get("task")
        stream_state["task"] = None
        if task is not None and not task.done():
            task.cancel()

    def set_streaming(value: bool):
        nonlocal is_streaming
        is_streaming = value
        if value:
            # While a reply streams the send button becomes a stop button.
            send_btn.icon = ft.Icons.STOP
            send_btn.bgcolor = STOP_RED
            send_btn.tooltip = "Stop"
            send_btn.on_click = handle_stop
            send_btn.disabled = False
            input_field.disabled = True
            attach_btn.disabled = True
            start_writing_animation()
        else:
            send_btn.icon = ft.Icons.SEND
            send_btn.bgcolor = PRIMARY
            send_btn.tooltip = "Send"
            send_btn.on_click = handle_send
            send_btn.disabled = False
            input_field.disabled = False
            attach_btn.disabled = False
            stop_writing_animation()
        try:
            send_btn.update()
            input_field.update()
            attach_btn.update()
        except Exception:
            pass

    def attachment_chip(attachment, removable=False, on_remove=None):
        p = palette()
        kind = attachment.get("kind", "file")
        icon = {
            "image": ft.Icons.IMAGE,
            "pdf": ft.Icons.PICTURE_AS_PDF,
            "spreadsheet": ft.Icons.TABLE_CHART,
            "presentation": ft.Icons.SLIDESHOW,
            "document": ft.Icons.DESCRIPTION,
            "text": ft.Icons.ARTICLE,
        }.get(kind, ft.Icons.INSERT_DRIVE_FILE)
        meta = ""
        if attachment.get("error"):
            meta = " · error"
        elif kind == "pdf" and attachment.get("pages"):
            meta = f" · {attachment['pages']} pages"
        elif kind == "image":
            meta = " · image"
        label = attachment.get("name", "file")
        if len(label) > 34:
            label = label[:31] + "…"
        controls = []
        if attachment.get("thumb"):
            controls.append(
                ft.Image(
                    src=attachment["thumb"],
                    width=22,
                    height=22,
                    fit=ft.BoxFit.COVER,
                    border_radius=4,
                )
            )
        else:
            controls.append(ft.Icon(icon, size=14, color=p["text_muted"]))
        controls.append(
            ft.Text(
                label + meta,
                size=11,
                color=p["text_muted"],
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
            )
        )
        if removable:
            controls.append(
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_size=12,
                    tooltip="Remove",
                    width=22,
                    height=22,
                    padding=0,
                    on_click=on_remove,
                )
            )
        return ft.Container(
            content=ft.Row(
                controls=controls,
                spacing=6,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.symmetric(horizontal=8, vertical=4),
            border_radius=10,
            bgcolor=p["card"],
            border=ft.Border.all(1, p["border"]),
        )

    def message_bubble(text: str, is_user: bool, streaming: bool, p: dict, attachments=None):
        bubble_children = []
        if attachments:
            bubble_children.append(
                ft.Row(
                    controls=[
                        attachment_chip(attachment, removable=False)
                        for attachment in attachments
                    ],
                    wrap=True,
                    spacing=6,
                )
            )
        if streaming and not text:
            # Nothing has arrived yet, so show the "writing" indicator.
            dots = ft.Text(".", size=13, color=p["text_muted"], width=16)
            WRITING_ANIM["dots"] = dots
            WRITING_ANIM["cursor"] = None
            bubble_children.append(
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.EDIT, size=14, color=p["text_muted"]),
                        ft.Text(
                            "HECRIG is writing",
                            size=13,
                            color=p["text_muted"],
                            italic=True,
                        ),
                        dots,
                    ],
                    spacing=6,
                    tight=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
            )
        elif text or streaming:
            text_control = ft.Text(
                text + (" ▋" if streaming else ""),
                color=p["user_text"] if is_user else p["ai_text"],
                size=14,
                selectable=True,
                no_wrap=False,
            )
            if streaming:
                WRITING_ANIM["cursor"] = text_control
                WRITING_ANIM["base"] = text
                WRITING_ANIM["dots"] = None
            bubble_children.append(text_control)
        bubble = ft.Container(
            content=ft.Column(controls=bubble_children, spacing=6, tight=True),
            padding=ft.Padding.all(12),
            border_radius=ft.BorderRadius.all(BORDER_RADIUS),
            bgcolor=p["user_bubble"] if is_user else p["ai_bubble"],
        )
        avatar = ft.Container(
            content=ft.Icon(
                ft.Icons.PERSON if is_user else ft.Icons.SMART_DISPLAY,
                size=16,
                color=p["user_text"] if is_user else PRIMARY,
            ),
            width=28,
            height=28,
            border_radius=14,
            bgcolor=p["user_bubble"] if is_user else PRIMARY,
            alignment=ft.Alignment.CENTER,
        )
        # The expanding container constrains the bubble width so long text wraps
        # instead of running off the screen. Alignment keeps it on the right side
        # for user messages and the left side for AI messages.
        bubble_slot = ft.Container(
            content=bubble,
            expand=True,
            alignment=ft.Alignment.CENTER_RIGHT if is_user else ft.Alignment.CENTER_LEFT,
        )
        row = ft.Row(
            controls=(
                [bubble_slot, avatar] if is_user else [avatar, bubble_slot]
            ),
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=8,
        )
        return ft.Container(
            content=row,
            padding=ft.Padding.symmetric(horizontal=16, vertical=4),
        )

    def add_user_message(text: str, attachments=None):
        chat = current_chat()
        if chat is None:
            start_new_chat()
            chat = current_chat()
        chat["messages"].append(
            {
                "role": "user",
                "text": text,
                "attachments": list(attachments or []),
                "streaming": False,
            }
        )
        chat["title"] = chat_title_from_messages(chat["messages"])
        chat["updated_at"] = datetime.now()
        refresh_messages()
        refresh_history()

    def add_ai_message(text: str):
        chat = current_chat()
        if chat is None:
            start_new_chat()
            chat = current_chat()
        chat["messages"].append({"role": "assistant", "text": text, "streaming": False})
        chat["updated_at"] = datetime.now()
        refresh_messages()

    def _append_or_update_streaming(chat, text: str):
        msgs = chat["messages"]
        if msgs and msgs[-1]["role"] == "assistant" and msgs[-1].get("streaming"):
            msgs[-1]["text"] = text
        else:
            msgs.append({"role": "assistant", "text": text, "streaming": True})
        chat["updated_at"] = datetime.now()
        if chat is current_chat():
            refresh_messages()

    def _finalize_streaming(chat):
        msgs = chat["messages"]
        if msgs and msgs[-1]["role"] == "assistant":
            msgs[-1]["streaming"] = False
        chat["updated_at"] = datetime.now()
        if chat is current_chat():
            refresh_messages()
        refresh_history()

    def _user_content(message):
        text = (message.get("text") or "").strip()
        attachments = message.get("attachments") or []
        if not attachments:
            return text
        attachment_text = format_attachments_for_prompt(attachments)
        combined = (attachment_text + ("\n\n" + text if text else "")).strip()
        images = attachment_images(attachments)
        if not images:
            return combined
        parts = [
            {
                "type": "text",
                "text": combined or "Please analyze the attached file(s).",
            }
        ]
        for url in images:
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts

    def _build_api_messages(chat, web_enabled=True):
        history = []
        if SYSTEM_PROMPT:
            history.append(
                {"role": "system", "content": build_system_prompt(web_enabled)}
            )
        for m in chat["messages"]:
            if m.get("streaming"):
                continue
            if m["role"] not in ("user", "assistant"):
                continue
            text = (m.get("text") or "").strip()
            attachments = m.get("attachments") or []
            if not text and not attachments:
                continue
            if m["role"] == "user" and attachments:
                content = _user_content(m)
            else:
                content = text
            history.append({"role": m["role"], "content": content})
        return history

    async def stream_ai_response(chat):
        set_streaming(True)
        _append_or_update_streaming(chat, "")
        revealed = ""
        error = None
        cancelled = False
        last_render = 0.0

        web_enabled = web_state["enabled"]
        messages = _build_api_messages(chat, web_enabled)
        tools = build_tools(web_enabled)

        def render(step_text: str):
            nonlocal last_render
            now = time.monotonic()
            if now - last_render > 0.05:
                _append_or_update_streaming(chat, revealed + step_text)
                last_render = now

        def final_answer_messages(source_messages):
            """Copy the history with a system prompt that forbids tool calls."""
            copied = [dict(message) for message in source_messages]
            if copied and copied[0].get("role") == "system":
                copied[0] = {
                    "role": "system",
                    "content": (
                        build_system_prompt(False)
                        + " You have already gathered the necessary information. "
                        "Do not call any tools and do not output tool-call syntax. "
                        "Answer the user's question directly now, using the "
                        "information above."
                    ),
                }
            return copied

        try:
            for step in range(MAX_TOOL_STEPS + 1):
                # On the last step, drop the tools so the model must answer.
                final_step = step >= MAX_TOOL_STEPS
                step_tools = None if final_step else tools
                step_messages = (
                    final_answer_messages(messages) if final_step else messages
                )
                try:
                    content, calls, _ = await stream_chat_completion(
                        step_messages, tools=step_tools, on_delta=render
                    )
                except RuntimeError as ex:
                    # Some llama.cpp builds/templates reject the tools field.
                    # Fall back to a plain answer instead of failing the chat.
                    if step == 0 and step_tools and "HTTP 400" in str(ex):
                        tools = None
                        content, calls, _ = await stream_chat_completion(
                            final_answer_messages(messages),
                            tools=None,
                            on_delta=render,
                        )
                        final_step = True
                    else:
                        raise

                if final_step:
                    # Tools are off: never accept another tool call.
                    content = _strip_tool_call_text(content)
                    calls = []

                revealed += content
                if not calls:
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": calls,
                    }
                )
                for call in calls:
                    name = call["function"]["name"]
                    raw_arguments = call["function"]["arguments"] or "{}"
                    try:
                        arguments = json.loads(raw_arguments)
                    except (TypeError, json.JSONDecodeError):
                        arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}

                    if name == "web_search":
                        query = str(arguments.get("query", "")).strip()
                        if query:
                            revealed += (
                                f"\n\n🔎 Searching the web for **{query}**…\n\n"
                            )
                            _append_or_update_streaming(chat, revealed)
                            result = await web_search(query)
                        else:
                            result = "Error: missing search query."
                    elif name == "fetch_url":
                        url = str(arguments.get("url", "")).strip()
                        if url:
                            revealed += f"\n\n🌐 Reading {url}…\n\n"
                            _append_or_update_streaming(chat, revealed)
                            result = await fetch_url(url)
                        else:
                            result = "Error: missing URL."
                    elif name == "remember":
                        text = str(arguments.get("text", "")).strip()
                        if text:
                            revealed += f"\n\n🧠 Remembering: {text}\n\n"
                            _append_or_update_streaming(chat, revealed)
                            result = await remember_memory(text)
                        else:
                            result = "Error: missing memory text."
                    elif name == "forget":
                        ref = str(arguments.get("memory", "")).strip()
                        if ref:
                            revealed += f"\n\n🧠 Forgetting: {ref}\n\n"
                            _append_or_update_streaming(chat, revealed)
                            result = await forget_memory(ref)
                        else:
                            result = "Error: missing memory reference."
                    elif name == "save_skill":
                        skill_name = str(arguments.get("name", "")).strip()
                        description = str(arguments.get("description", "")).strip()
                        instructions = str(
                            arguments.get("instructions", "")
                        ).strip()
                        if skill_name and instructions:
                            revealed += (
                                f"\n\n📚 Learning skill: **{skill_name}**\n\n"
                            )
                            _append_or_update_streaming(chat, revealed)
                            result = await save_skill(
                                skill_name, description, instructions
                            )
                        else:
                            result = (
                                "Error: skill name and instructions are required."
                            )
                    elif name == "use_skill":
                        skill_name = str(arguments.get("name", "")).strip()
                        if skill_name:
                            revealed += (
                                f"\n\n📚 Using skill: **{skill_name}**\n\n"
                            )
                            _append_or_update_streaming(chat, revealed)
                            result = await use_skill(skill_name)
                        else:
                            result = "Error: missing skill name."
                    elif name == "list_skills":
                        result = await list_skills()
                    elif name == "delete_skill":
                        skill_name = str(arguments.get("name", "")).strip()
                        if skill_name:
                            revealed += (
                                f"\n\n📚 Deleting skill: **{skill_name}**\n\n"
                            )
                            _append_or_update_streaming(chat, revealed)
                            result = await delete_skill(skill_name)
                        else:
                            result = "Error: missing skill name."
                    elif name == "read_document":
                        file_path = str(arguments.get("path", "")).strip()
                        if file_path:
                            revealed += (
                                f"\n\n📄 Reading {file_path}…\n\n"
                            )
                            _append_or_update_streaming(chat, revealed)
                            result = await read_document(file_path)
                        else:
                            result = "Error: missing file path."
                    else:
                        result = f"Error: unknown tool '{name}'."

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result,
                        }
                    )

                if (
                    any(
                        call["function"]["name"] in MUTATING_MEMORY_TOOLS
                        for call in calls
                    )
                    and messages
                    and messages[0].get("role") == "system"
                ):
                    messages[0]["content"] = build_system_prompt(web_enabled)
            set_status(True)
        except asyncio.CancelledError:
            cancelled = True
        except Exception as ex:
            error = ex
            set_status(False)
        finally:
            try:
                if cancelled:
                    # Keep whatever the user already saw; only add a note when
                    # the reply was stopped before any text arrived.
                    msgs = chat["messages"]
                    current = (msgs[-1].get("text") or "").strip() if msgs else ""
                    if not current:
                        _append_or_update_streaming(chat, "⏹ Stopped.")
                elif error is not None:
                    hint = (
                        "⚠️ Couldn't reach the local model.\n\n"
                        f"Endpoint: {LLAMA_BASE_URL}/v1/chat/completions\n\n"
                        "HECRIG starts llama-server automatically. If the model is "
                        "still loading, wait a moment and try again. Check the "
                        f"status badge, or see the log:\n{SERVER_LOG_PATH}\n\n"
                        f"Details: {error}"
                    )
                    _append_or_update_streaming(chat, hint)
                elif not revealed.strip():
                    _append_or_update_streaming(chat, "(no response)")
            finally:
                _finalize_streaming(chat)
                set_streaming(False)
                stream_state["task"] = None

    async def handle_send(e=None):
        text = (input_field.value or "").strip()
        if is_streaming:
            return
        if not text and not pending_attachments:
            return
        attachments = list(pending_attachments)
        pending_attachments.clear()
        set_attach_status("")
        refresh_attachment_bar()
        input_field.value = ""
        input_field.update()
        add_user_message(
            text or "Please analyze the attached file(s).", attachments
        )
        chat = current_chat()
        if chat is None:
            return
        stream_state["task"] = page.run_task(stream_ai_response, chat)

    # ------------------------------------------------------------------
    # Memory & skills dialog
    # ------------------------------------------------------------------
    def open_brain(e=None):
        p = palette()
        memories = MEMORY_STORE.all()
        skills = SKILL_STORE.all()

        memory_input = ft.TextField(
            hint_text="Add a fact about yourself...",
            hint_style=ft.TextStyle(color=p["text_muted"]),
            border=ft.InputBorder.NONE,
            filled=True,
            bgcolor=p["card"],
            border_radius=10,
            color=p["text"],
            cursor_color=p["text"],
            dense=True,
            expand=True,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
        )

        def refresh_dialog():
            try:
                page.pop_dialog()
            except Exception:
                pass
            open_brain()

        def add_memory(e=None):
            text = (memory_input.value or "").strip()
            if not text:
                return
            MEMORY_STORE.add(text)
            refresh_dialog()

        def remove_memory(memory_id):
            MEMORY_STORE.forget(memory_id)
            refresh_dialog()

        def remove_skill(skill_id):
            SKILL_STORE.delete(skill_id)
            refresh_dialog()

        memory_controls = []
        if memories:
            for item in memories:
                memory_controls.append(
                    ft.Row(
                        controls=[
                            ft.Text(
                                item["text"],
                                size=13,
                                color=p["text"],
                                selectable=True,
                                expand=True,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.DELETE_OUTLINE,
                                icon_size=16,
                                tooltip="Forget",
                                on_click=lambda e, mid=item["id"]: remove_memory(mid),
                            ),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=4,
                    )
                )
        else:
            memory_controls.append(
                ft.Text(
                    "Nothing saved yet. Tell HECRIG something about yourself.",
                    size=12,
                    color=p["text_muted"],
                    italic=True,
                )
            )

        skill_controls = []
        if skills:
            for item in skills:
                skill_controls.append(
                    ft.ExpansionTile(
                        title=ft.Text(
                            item["name"],
                            size=13,
                            weight=ft.FontWeight.W_600,
                            color=p["text"],
                        ),
                        subtitle=ft.Text(
                            item.get("description") or "No description",
                            size=11,
                            color=p["text_muted"],
                        ),
                        controls=[
                            ft.Container(
                                content=ft.Text(
                                    item["instructions"],
                                    size=12,
                                    color=p["text"],
                                    selectable=True,
                                ),
                                padding=ft.Padding.only(
                                    left=16, right=16, bottom=10
                                ),
                            )
                        ],
                        trailing=ft.IconButton(
                            icon=ft.Icons.DELETE_OUTLINE,
                            icon_size=16,
                            tooltip="Delete skill",
                            on_click=lambda e, sid=item["id"]: remove_skill(sid),
                        ),
                        dense=True,
                        text_color=p["text"],
                        icon_color=p["text_muted"],
                        collapsed_text_color=p["text"],
                        collapsed_icon_color=p["text_muted"],
                    )
                )
        else:
            skill_controls.append(
                ft.Text(
                    "No skills yet. Teach HECRIG a workflow in chat.",
                    size=12,
                    color=p["text_muted"],
                    italic=True,
                )
            )

        content = ft.Container(
            width=560,
            height=460,
            content=ft.Column(
                controls=[
                    ft.Text(
                        "Memory",
                        size=15,
                        weight=ft.FontWeight.W_600,
                        color=p["text"],
                    ),
                    ft.Text(
                        "Facts HECRIG remembers about you.",
                        size=11,
                        color=p["text_muted"],
                    ),
                    ft.Column(controls=memory_controls, spacing=2),
                    ft.Row(
                        controls=[
                            memory_input,
                            ft.IconButton(
                                icon=ft.Icons.ADD_CIRCLE,
                                icon_size=20,
                                icon_color=PRIMARY,
                                tooltip="Add memory",
                                on_click=add_memory,
                            ),
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Divider(height=1, color=p["border"]),
                    ft.Text(
                        "Skills",
                        size=15,
                        weight=ft.FontWeight.W_600,
                        color=p["text"],
                    ),
                    ft.Text(
                        "Reusable procedures HECRIG has learned.",
                        size=11,
                        color=p["text_muted"],
                    ),
                    ft.Column(controls=skill_controls, spacing=0),
                ],
                spacing=8,
                scroll=ft.ScrollMode.AUTO,
            ),
        )

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(
                "Memory & Skills", size=18, weight=ft.FontWeight.W_600
            ),
            content=content,
            actions=[
                ft.TextButton("Close", on_click=lambda e: page.pop_dialog()),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.show_dialog(dialog)

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    new_chat_btn = ft.FilledButton(
        content=ft.Row(
            controls=[
                ft.Icon(ft.Icons.ADD_CIRCLE, size=16, color="#FFFFFF"),
                ft.Text("New chat", size=14, weight=ft.FontWeight.W_500, color="#FFFFFF"),
            ],
            spacing=8,
            tight=True,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        on_click=lambda e: start_new_chat(),
        style=ft.ButtonStyle(
            bgcolor=PRIMARY,
            color="#FFFFFF",
            padding=ft.Padding.symmetric(vertical=12),
            shape=ft.RoundedRectangleBorder(8),
        ),
    )

    sidebar = ft.Container(
        content=ft.Column(
            controls=[
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Container(
                                content=ft.Icon(ft.Icons.SMART_DISPLAY, size=22, color=PRIMARY),
                                width=40,
                                height=40,
                                border_radius=10,
                                bgcolor=PRIMARY_FAINT,
                                alignment=ft.Alignment.CENTER,
                            ),
                            ft.Text("HECRIG", size=20, weight=ft.FontWeight.BOLD, color=palette()["text"]),
                        ],
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=ft.Padding.all(20),
                ),
                ft.Divider(height=1, color=palette()["border"]),
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.ADD_CIRCLE, size=16, color="#FFFFFF"),
                            ft.Text("New chat", size=14, weight=ft.FontWeight.W_500, color="#FFFFFF"),
                        ],
                        spacing=8,
                        tight=True,
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    bgcolor=PRIMARY,
                    border_radius=ft.BorderRadius.all(8),
                    height=44,
                    alignment=ft.Alignment.CENTER,
                    margin=ft.Margin.symmetric(horizontal=16, vertical=8),
                    on_click=lambda e: start_new_chat(),
                    ink=True,
                ),
                ft.Divider(height=1, color=palette()["border"]),
                ft.Container(
                    content=ft.Text("Chats", size=12, color=palette()["text_muted"], weight=ft.FontWeight.W_600),
                    padding=ft.Padding.only(left=16, top=12, bottom=8, right=16),
                ),
                history_list,
                ft.Divider(height=1, color=palette()["border"]),
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.IconButton(
                                icon=ft.Icons.PSYCHOLOGY,
                                icon_size=18,
                                tooltip="Memory & skills",
                                on_click=open_brain,
                            ),
                            ft.VerticalDivider(width=1, color=palette()["border"]),
                            theme_switch,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                ),
            ],
            spacing=0,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
        width=280,
        padding=0,
    )

    def toggle_theme(e):
        nonlocal current_theme_dark
        current_theme_dark = not current_theme_dark
        page.theme_mode = ft.ThemeMode.DARK if current_theme_dark else ft.ThemeMode.LIGHT
        update_theme_colors()

    theme_switch.on_change = toggle_theme
    theme_switch.value = True

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    model_dropdown = ft.Dropdown(
        label="Model",
        value=LLAMA_MODEL,
        options=[ft.dropdown.Option(LLAMA_MODEL)],
        width=170,
        border_color=palette()["border"],
        color=palette()["text"],
        label_style=ft.TextStyle(color=palette()["text_muted"]),
    )

    web_btn = ft.IconButton(
        icon=ft.Icons.TRAVEL_EXPLORE,
        icon_color=PRIMARY if web_state["enabled"] else palette()["text_muted"],
        tooltip="Web search: on" if web_state["enabled"] else "Web search: off",
    )

    def toggle_web_search(e):
        web_state["enabled"] = not web_state["enabled"]
        web_btn.icon_color = (
            PRIMARY if web_state["enabled"] else palette()["text_muted"]
        )
        web_btn.tooltip = (
            "Web search: on" if web_state["enabled"] else "Web search: off"
        )
        web_btn.update()

    web_btn.on_click = toggle_web_search

    header = ft.Container(
        content=ft.Row(
            controls=[
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Text("Chat", size=18, weight=ft.FontWeight.W_600, color=palette()["text"]),
                            ft.Container(width=8),
                            status_badge,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=ft.Padding.only(left=16, right=16, top=12, bottom=12),
                ),
                ft.Container(expand=True),
                web_btn,
                ft.Container(width=4),
                model_dropdown,
                ft.Container(width=16),
                theme_switch,
                ft.Container(width=20),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        height=64,
        border=ft.Border.only(bottom=ft.BorderSide(1, palette()["border"])),
    )

    # ------------------------------------------------------------------
    # Input bar
    # ------------------------------------------------------------------
    input_field = ft.TextField(
        hint_text="Message HECRIG...",
        hint_style=ft.TextStyle(color=palette()["text_muted"]),
        border=ft.InputBorder.NONE,
        filled=True,
        bgcolor=palette()["card"],
        border_radius=20,
        color=palette()["text"],
        cursor_color=palette()["text"],
        min_lines=1,
        max_lines=5,
        expand=True,
        content_padding=ft.Padding.symmetric(horizontal=18, vertical=12),
        on_submit=handle_send,
    )

    send_btn = ft.IconButton(
        icon=ft.Icons.SEND,
        icon_color="#FFFFFF",
        bgcolor=PRIMARY,
        tooltip="Send",
        on_click=handle_send,
    )

    attach_btn = ft.IconButton(
        icon=ft.Icons.ATTACH_FILE,
        icon_color=palette()["text_muted"],
        tooltip="Attach image, PDF, or document",
    )

    attach_status = ft.Text("", size=11, color=palette()["text_muted"], visible=False)
    attachment_bar = ft.Column(controls=[], spacing=4, visible=False)

    def set_attach_status(message: str, error: bool = False):
        attach_status.value = message
        attach_status.color = "#F87171" if error else palette()["text_muted"]
        attach_status.visible = bool(message)
        try:
            attach_status.update()
        except Exception:
            pass

    def remove_pending(index: int):
        if 0 <= index < len(pending_attachments):
            pending_attachments.pop(index)
            refresh_attachment_bar()

    def refresh_attachment_bar():
        chips = [
            attachment_chip(
                attachment,
                removable=True,
                on_remove=lambda e, index=index: remove_pending(index),
            )
            for index, attachment in enumerate(pending_attachments)
        ]
        attachment_bar.controls = (
            [ft.Row(controls=chips, wrap=True, spacing=6)] if chips else []
        )
        attachment_bar.visible = bool(chips)
        try:
            attachment_bar.update()
        except Exception:
            pass

    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    async def pick_attachments(e=None):
        if is_streaming:
            return
        if len(pending_attachments) >= ATTACH_MAX_FILES:
            set_attach_status(
                f"Up to {ATTACH_MAX_FILES} files per message.", error=True
            )
            return
        try:
            files = await file_picker.pick_files(
                dialog_title="Attach images, PDFs, or documents",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=[
                    ext.lstrip(".") for ext in ALLOWED_ATTACHMENT_EXTENSIONS
                ],
                allow_multiple=True,
            )
        except Exception as ex:
            set_attach_status(f"Could not open the file picker: {ex}", error=True)
            return
        if not files:
            return

        errors = []
        for file in files:
            if len(pending_attachments) >= ATTACH_MAX_FILES:
                errors.append(f"Only {ATTACH_MAX_FILES} files can be attached.")
                break
            path = getattr(file, "path", None)
            if not path:
                continue
            set_attach_status(f"Scanning {file.name}…")
            attachment = await extract_attachment_async(path, file.name)
            if (
                attachment.get("error")
                and not attachment.get("text")
                and not attachment.get("images")
            ):
                errors.append(f"{file.name}: {attachment['error']}")
                continue
            pending_attachments.append(attachment)
            refresh_attachment_bar()
        set_attach_status(" · ".join(errors[:2]), error=bool(errors))

    attach_btn.on_click = pick_attachments

    input_bar = ft.Container(
        content=ft.Column(
            controls=[
                attach_status,
                attachment_bar,
                ft.Row(
                    controls=[
                        ft.Container(width=16),
                        attach_btn,
                        ft.Container(width=8),
                        input_field,
                        ft.Container(width=10),
                        send_btn,
                        ft.Container(width=16),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            spacing=6,
            tight=True,
        ),
        padding=ft.Padding.symmetric(vertical=14),
        border=ft.Border.only(top=ft.BorderSide(1, palette()["border"])),
    )

    # ------------------------------------------------------------------
    # Page assembly
    # ------------------------------------------------------------------
    body = ft.Row(
        controls=[
            sidebar,
            ft.VerticalDivider(width=1, color=palette()["border"]),
            ft.Column(
                controls=[
                    header,
                    ft.Container(content=chat_list, expand=True),
                    input_bar,
                ],
                expand=True,
                spacing=0,
            ),
        ],
        expand=True,
        spacing=0,
    )

    page.add(body)

    # Initialize
    update_theme_colors()

    # Start with one fresh chat and a greeting
    start_new_chat()
    add_ai_message(
        "Hi! I'm HECRIG, running locally. Ask me anything, or attach an "
        "image, PDF, or document with the paperclip."
    )

    # Start llama-server (or connect to one that is already running). This runs
    # in the background so the window appears immediately while the model loads.
    page.run_task(ensure_server)

    # Stop the server we launched when the desktop window closes. In web mode a
    # single tab closing should not kill the shared server.
    if not getattr(page, "web", False):
        page.on_close = lambda e: _stop_server()


if __name__ == "__main__":
    ft.run(main)
