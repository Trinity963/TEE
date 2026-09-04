# TEE — Trinity's Execution Engine

> *You don't need them. You never did.*

TEE is a sovereign, open source LLM router. It sits between your applications and your local models — handling backend selection, GPU placement, model registration, and inference routing so nothing else has to.

No cloud. No accounts. No API keys to anyone else's kingdom. Your weights. Your hardware. Your rules.

---

## Why TEE exists

Every platform you build on will eventually change, restrict, monetize, or disappear.

- Ollama walled models behind access controls
- HuggingFace is being acquired
- Every "free" inference API has a ceiling

TEE is the exit ramp. Once it's running, none of that touches you.

---

## What TEE does

TEE does **one thing** — and does it completely:

```
Your App  ──▶  TEE  ──▶  llama.cpp / vLLM  ──▶  Your GPU  ──▶  Your Model
```

- Receives inference requests via OpenAI-compatible API
- Knows which model to use
- Knows which backend to use
- Knows which GPU to use
- Returns the response

Everything else calls TEE. TEE calls nothing external.

---

## What TEE does NOT do

- No UI (see VIVARIUM Dashboard for that)
- No database
- No auth layer
- No cloud sync
- No telemetry
- No opinions about what calls it

---

## Quick Start

```bash
# Clone
git clone https://github.com/YOU/TEE.git
cd TEE

# Install
pip install -r requirements.txt

# Configure
cp tee.config.example tee.config
# Edit tee.config — set your models_dirs path

# Run
python tee.py
```

TEE starts, scans your model directories, registers everything it finds, and begins serving on `http://localhost:8765/v1/`

---

## Plug and Play

Drop a GGUF into your models directory. TEE does the rest.

```
/models/mistral-7b-instruct-q4_k_m.gguf  ←  drop it here

TEE detects it
TEE reads the GGUF header
TEE generates the modelfile automatically
TEE registers the model
Model is live on /v1/  ←  ready
```

No commands. No configuration. No technical knowledge required.

---

## Module Integration

TEE is designed to be consumed as a module by any sovereign stack.

```python
from tee import TEE

tee = TEE()

# Chat
response = tee.chat("mistral-7b-instruct", messages)

# List available models
models = tee.list_models()

# Check system status
status = tee.status()
```

### OpenAI-compatible API

Any application that speaks the OpenAI API protocol works with TEE immediately — zero changes required.

```bash
# List models
curl http://localhost:8765/v1/models

# Chat
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b-instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## Multi-Directory Model Support

TEE watches multiple model directories simultaneously. Models can live anywhere — NVMe, HDD, external drive, NAS. TEE sees all of it.

```json
"models_dirs": [
  { "path": "/mnt/nvme/models",    "label": "Fast Drive"     },
  { "path": "/mnt/hdd/models",     "label": "Large Storage"  },
  { "path": "/mnt/external/models","label": "External Drive" }
]
```

TEE auto-detects drive type and suggests a label. You can rename it to anything you want — TEE remembers your choice permanently.

---

## Hardware Intelligence

TEE reads your system on startup and never lets you load a model your hardware cannot run.

```
Your System:
  GPU 0: RTX 5080  16GB
  GPU 1: RTX 5080  16GB
  Total VRAM: 32GB

mistral-7b-instruct Q4_K_M (4.1GB)  ✅ Runs on Single GPU
llama3-70b Q4_K_M (38GB)            🔴 Needs Multi-GPU — TEE will span GPU 0+1
llama3-70b F16 (140GB)              🚫 Cannot Run — exceeds total VRAM
```

For users who don't know what any of that means — TEE says it in plain English, suggests what will work, and never lets a failed load be a surprise.

---

## Supported Backends

| Backend    | Format       | Status  |
|------------|--------------|---------|
| llama.cpp  | GGUF         | ✅ Supported |
| vLLM       | Safetensors  | ✅ Supported |
| koboldcpp  | GGUF         | 🔜 Planned  |
| llamafile  | Llamafile    | 🔜 Planned  |

Backend selection is automatic based on model format. Override in your modelfile if needed.

---

## Directory Structure

```
TEE/
├── core/              # Router engine
│   ├── registry.py    # Model registry — reads manifests, watches dirs
│   ├── runtime.py     # Load/unload, GPU placement, backend selection
│   ├── gateway.py     # OpenAI-compatible API gateway
│   └── detector.py    # GGUF header reader, modelfile generator
├── adapters/          # Backend bridges
│   ├── llamacpp.py    # llama.cpp adapter
│   └── vllm.py        # vLLM adapter
├── model-index/       # Community maintained model database
│   ├── ratings.json   # Ratings, tags, download sources
│   └── profiles/      # Per-model variant profiles
├── drive-profiles/    # Drive type detection and health
│   └── profiles.json
├── gpu-profiles/      # GPU VRAM and capability database
│   └── profiles.json
├── config/
│   └── modelfile.example.json
├── docs/              # Plain English documentation
├── tee.config.example
├── tee.py             # Entry point
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Modelfile Schema

TEE auto-generates modelfiles from GGUF headers. You can also write your own.

```json
{
  "name": "mistral-7b-instruct",
  "file": "mistral-7b-instruct-v0.3-q4_k_m.gguf",
  "format": "gguf",
  "architecture": "mistral",
  "parameters": "7B",
  "quantization": "Q4_K_M",
  "context": 8192,
  "backend": "auto",
  "gpu": "auto",
  "defaults": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048
  }
}
```

Set `backend` and `gpu` to `"auto"` and TEE handles everything. Override only when you need to.

---

## VIVARIUM Integration

TEE was built as the inference backbone of the [VIVARIUM](https://github.com/YOU/VIVARIUM) sovereign AI ecosystem.

Projects that run on TEE:

| Project    | Role                              |
|------------|-----------------------------------|
| MiniTrini  | Sovereign AI stack — guard, chat, vision |
| Ethica     | Sovereign AI platform             |
| TBS        | Trinity Browser IDE               |
| SARA       | Strategic Adaptive Relational Architecture |

Any VIVARIUM project pointing at Ollama or an external API can point at TEE instead. Zero other changes required.

---

## Contributing

TEE is open source. Everything is open. Everything stays open.

**What the community maintains:**

- `model-index/ratings.json` — new models, updated ratings, download sources
- `gpu-profiles/profiles.json` — new GPU VRAM profiles
- `drive-profiles/profiles.json` — new drive type detection
- `adapters/` — new backend bridges
- `docs/` — translations, plain English improvements

**To contribute:**

1. Fork the repo
2. Make your change
3. Submit a PR with a clear description
4. Community reviews — merged if it helps everyone

No CLAs. No corporate overhead. Just useful changes that help people run their own AI.

---

## License

MIT — do whatever you want with it.

Just don't close it.  
Just don't wall it.  
Just don't charge for what was free.

---

## Built by

V — The Architect  
VIVARIUM Sovereign AI Ecosystem  
Toronto, Canada

*"Build it sovereign. Keep it sovereign. Share it sovereign."*
