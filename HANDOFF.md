# TEE — Trinity's Execution Engine
# Session Handoff

---

## Session 1 — 2026-09-04

### What was built
Full core stack — designed, written, patched, and tested live on trinity1.

### Files created
```
/srv/Build_Core/TEE/
├── tee.py                          ← sovereign entry point
├── tee.config                      ← live config, both model dirs saved
├── tee.config.example              ← schema reference
├── README.md                       ← public face, MIT license
├── core/
│   ├── detector.py                 ← GGUF header reader, modelfile auto-generator
│   ├── registry.py                 ← dir watcher, hot-drop, live manifest
│   ├── runtime.py                  ← GPU detection, placement, load/unload
│   └── gateway.py                  ← OpenAI-compatible API gateway
├── config/
│   └── modelfile.example.json      ← modelfile schema reference
├── model-index/
│   └── ratings.json                ← community model index seed
├── gpu-profiles/
│   └── profiles.json               ← GPU VRAM database
└── drive-profiles/
    └── profiles.json               ← drive type detection database
```

### Start command
```bash
python3 /srv/Build_Core/TEE/tee.py
```
No flags needed — reads tee.config automatically.

### Model dirs in tee.config
```
/srv/LLMs/gguf
/srv/LLMs/models
```

### Models currently registered
```
Mistral Nemo 12B Q4_K_M     llama   6.96GB   ctx:131,072   single GPU
Mistral Nemo 12B F16        llama   22.82GB  ctx:131,072   multi-GPU
RiverSovereign 7b           llama   4.07GB   ctx:32,768    single GPU
DeepSeek R1 32B Q4_K_M     qwen2   18.49GB  ctx:131,072   multi-GPU
```

### Hardware detected
```
GPU 0: NVIDIA GeForce RTX 5080  15.92GB VRAM
GPU 1: NVIDIA GeForce RTX 5080  15.92GB VRAM
Backend: llama.cpp @ /srv/Build_Core/llama.cpp/build/bin/llama-server
vLLM: not installed (not needed yet — all models are GGUF)
```

### Gateway
```
http://0.0.0.0:8765/v1/
GET  /v1/models
POST /v1/chat/completions
POST /v1/embeddings
GET  /health
GET  /status
```

### Patches applied this session
- SO_REUSEADDR on gateway — port releases immediately on shutdown
- GGUF_QUANT_INT int map at module level in detector.py — F16/quantization int detection fixed
- Unique model names — quantization suffix appended for uniqueness
- llama-server alt_bins — /srv/Build_Core/llama.cpp/build/bin/llama-server added
- Description fallback uses model name not architecture
- detector.py _extract_quantization guards int types properly

### Known issues
- RiverSovereign modelfile shows F16 — was generated before quant patch
  Fix: rm /srv/LLMs/models/RiverSovereign-7b-Q4_K_M.modelfile.json then restart TEE

### Next session priorities
1. Fix RiverSovereign modelfile label
2. git init /srv/Build_Core/TEE
3. Push to GitHub — TEE goes public
4. Build ui/ — browser interface
   - Dashboard: GPU status, VRAM bars, loaded models
   - Models page: all registered models, add dir, hot-drop
   - Directories panel: multi-dir management, drive health, space
   - Config editor: visual tee.config, no JSON required

### Architecture locked
TEE is the router only. Not a full stack.
Every VIVARIUM project calls TEE. TEE calls nothing external.

```
MiniTrini ──┐
Ethica    ──┼──▶ TEE ──▶ llama.cpp / vLLM ──▶ GPU
TBS       ──┤
SARA      ──┘
```

### Philosophy
> You don't need them. You never did.
> Build it sovereign. Keep it sovereign. Share it sovereign.
