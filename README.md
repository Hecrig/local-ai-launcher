# HECRIG

## Local model

HECRIG starts `llama-server` (from llama.cpp) automatically when the app launches.
It looks for `llama-server` on `PATH` and for a `.gguf` model in `models/`, the
project root, or the Hugging Face cache.

Override the defaults with environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLAMA_AUTOSTART` | `1` | Set to `0` to connect to an already-running server |
| `LLAMA_SERVER_BIN` | `llama-server` | Full path to `llama-server` |
| `LLAMA_MODEL_PATH` | auto-detected | Full path to the `.gguf` model |
| `LLAMA_SERVER_ARGS` | empty | Extra flags, e.g. `-c 4096 -ngl 99` |
| `LLAMA_BASE_URL` | `http://127.0.0.1:8080` | API endpoint |
| `LLAMA_START_TIMEOUT` | `180` | Seconds to wait for the model to load |
| `HECRIG_CTX` | `8192` | Context size when HECRIG starts the server |
| `HECRIG_NGL` | *(auto-fit)* | GPU layers: `99` = full offload, `0` = CPU only |
| `HECRIG_PARALLEL` | `1` | Server slots; `1` gives one chat the whole context |
| `HECRIG_FLASH_ATTN` | *(llama.cpp default)* | Set `on`/`off`/`auto` to override |
| `HECRIG_MMPROJ` | auto-detected | Path to the vision projector (`mmproj*.gguf`) |

The server log is written to `%TEMP%\hecrig_llama_server.log` on Windows.

If a matching `mmproj*.gguf` sits next to the model (or in the Hugging Face
cache), HECRIG passes it to `llama-server` with `--mmproj`, which lets the model
see images directly.

### Speed

By default HECRIG does **not** pass `-ngl`, so `llama-server` auto-fits as many
layers to your GPU as fit in VRAM. It always passes `-np 1`, which gives one chat
the whole context instead of splitting it across four slots.

- `HECRIG_NGL=99` — force full GPU offload.
- `HECRIG_NGL=0` — CPU only.
- `HECRIG_NGL=20` — partial offload.
- `HECRIG_PARALLEL=2` — let two chats generate at once (halves each one's context).
- `HECRIG_FLASH_ATTN=on` — opt in to flash attention. HECRIG leaves this at
  llama.cpp's default because on some builds `-fa on` mis-parses and silently
  disables GPU offload.

Check what your build sees with `llama-server --list-devices`, and whether the
offload happened by looking for `offloaded N/M layers to GPU` in the server log
(run with `-lv 4` to see it).

**If the GPU path misbehaves**, try `HECRIG_NGL=0` (CPU only). Some Vulkan
builds handle automatic fitting better than a forced `-ngl 99`, so leaving
`HECRIG_NGL` unset is usually the best choice. A CUDA build of `llama-server`
is typically much faster than Vulkan on NVIDIA cards.

The most reliable speedup is a smaller model: a 4B model at the same
quantization is roughly twice as fast as a 9B one.

## Stopping a reply

While HECRIG is answering, the send button turns into a red stop button. Press it
to abort the reply; whatever text already arrived stays in the chat and you can
send a new message straight away. If nothing had arrived yet, the chat shows
"Stopped." instead.

## Internet access

HECRIG can search the live web and read pages using llama.cpp tool calling.
Web search is on by default; the globe icon in the header toggles it.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HECRIG_WEB_SEARCH` | `1` | Set to `0` to disable web tools |
| `TAVILY_API_KEY` | empty | Optional Tavily key for higher-quality search |
| `HECRIG_SEARCH_RESULTS` | `5` | Results returned per search |
| `HECRIG_FETCH_CHARS` | `6000` | Max characters read from a page |
| `HECRIG_MAX_TOOL_STEPS` | `4` | Max search/read rounds per answer |

Without a Tavily key it uses DuckDuckGo, which needs no account.

## Images, PDFs, and documents

Click the paperclip in the message bar to attach up to four files. HECRIG reads
them locally before sending your message:

| Type | How it is read |
| --- | --- |
| Images (PNG, JPG, WEBP, BMP, GIF, TIFF) | Sent to the model's vision projector; OCR is used if vision is unavailable |
| PDF | Text is extracted per page; scanned pages are sent as images or OCR'd |
| DOCX, ODT, ODS, ODP, RTF | Text and tables are extracted |
| XLSX | Rows from each sheet are extracted |
| PPTX | Text from each slide is extracted |
| TXT, MD, CSV, JSON, XML, YAML, code | Read as plain text |

The model can also read a file by path with the `read_document` tool, for
example when you paste a path into the chat.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HECRIG_MAX_ATTACHMENTS` | `4` | Files per message |
| `HECRIG_MAX_ATTACHMENT_MB` | `40` | Max size per file |
| `HECRIG_ATTACH_CHARS` | `6000` | Max text characters per file |
| `HECRIG_ATTACH_TOTAL_CHARS` | `12000` | Max text characters per message |
| `HECRIG_IMAGE_MAX_SIDE` | `1024` | Images are resized to this maximum side |
| `HECRIG_PDF_MAX_PAGES` | `30` | PDF pages read per file |
| `HECRIG_OCR_MAX_PAGES` | `6` | Scanned PDF pages OCR'd per file |
| `HECRIG_MAX_IMAGES` | `4` | Images sent to the model per message |

OCR uses the OCR engine built into Windows 10/11. If `pytesseract` or
`rapidocr_onnxruntime` is installed it is used as a fallback.

## Memory and skills

HECRIG can remember facts about you and save reusable skills (repeatable
procedures). Both survive restarts and are injected into every new chat.

- **Memory** — say "remember that I prefer short answers" and HECRIG saves it.
  It also saves durable facts automatically when you share them.
- **Skills** — say "learn a skill called Daily briefing: search the news and
  summarize the top 3 stories" and HECRIG stores the steps. Later, "use my
  Daily briefing skill" loads them.

Open the brain icon in the sidebar (`Memory & skills`) to view, add, or delete
memories and skills. The model can also call `remember`, `forget`, `save_skill`,
`use_skill`, `list_skills`, and `delete_skill` itself.

Data is stored as JSON under `%LOCALAPPDATA%\HECRIG` on Windows
(`~/.hecrig` elsewhere):

| File | Contents |
| --- | --- |
| `memory.json` | Remembered facts |
| `skills.json` | Saved skills and instructions |

| Variable | Default | Purpose |
| --- | --- | --- |
| `HECRIG_DATA_DIR` | OS app data | Override the storage folder |
| `HECRIG_MAX_MEMORIES` | `50` | Max saved memories (oldest dropped) |
| `HECRIG_MAX_SKILLS` | `50` | Max saved skills (oldest dropped) |

Secrets such as passwords and API keys are never saved.

## Run the app

### uv

Run as a desktop app:

```bash
uv run flet run
```

Run as a web app:

```bash
uv run flet run --web
```

For more details on running the app, refer to the [Getting Started Guide](https://flet.dev/docs/).

## Build the app

### Android

```bash
flet build apk -v
```

For more details on building and signing `.apk` or `.aab`, refer to the [Android Packaging Guide](https://flet.dev/docs/publish/android/).

### iOS

```bash
flet build ipa -v
```

For more details on building and signing `.ipa`, refer to the [iOS Packaging Guide](https://flet.dev/docs/publish/ios/).

### macOS

```bash
flet build macos -v
```

For more details on building macOS package, refer to the [macOS Packaging Guide](https://flet.dev/docs/publish/macos/).

### Linux

```bash
flet build linux -v
```

For more details on building Linux package, refer to the [Linux Packaging Guide](https://flet.dev/docs/publish/linux/).

### Windows

```bash
flet build windows -v
```

For more details on building Windows package, refer to the [Windows Packaging Guide](https://flet.dev/docs/publish/windows/).

### Web

```bash
flet build web -v
```

For more details on building Web app, refer to the [Web Packaging Guide](https://flet.dev/docs/publish/web/).
