"""
TEE — Trinity's Execution Engine
core/detector.py

GGUF header reader and modelfile auto-generator.
Drop a GGUF — TEE reads it and generates the modelfile automatically.
No user input required.

MIT License — open source, sovereign, forever.
"""

import struct
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


# ── GGUF Magic & Version ──────────────────────────────────────────────────────

GGUF_MAGIC       = b"GGUF"
GGUF_VERSION_MIN = 1
GGUF_VERSION_MAX = 3

# ── GGUF Value Types ──────────────────────────────────────────────────────────

GGUF_TYPE = {
    0:  ("UINT8",   "B", 1),
    1:  ("INT8",    "b", 1),
    2:  ("UINT16",  "H", 2),
    3:  ("INT16",   "h", 2),
    4:  ("UINT32",  "I", 4),
    5:  ("INT32",   "i", 4),
    6:  ("FLOAT32", "f", 4),
    7:  ("BOOL",    "?", 1),
    8:  ("STRING",  None, None),
    9:  ("ARRAY",   None, None),
    10: ("UINT64",  "Q", 8),
    11: ("INT64",   "q", 8),
    12: ("FLOAT64", "d", 8),
}

# ── Architecture map — GGUF general.architecture → human name ─────────────────

ARCH_MAP = {
    "llama":        "llama",
    "llama3":       "llama3",
    "mistral":      "mistral",
    "mixtral":      "mixtral",
    "phi":          "phi",
    "phi2":         "phi2",
    "phi3":         "phi3",
    "gemma":        "gemma",
    "gemma2":       "gemma2",
    "falcon":       "falcon",
    "gpt2":         "gpt2",
    "gptj":         "gptj",
    "gptneox":      "gptneox",
    "mpt":          "mpt",
    "qwen":         "qwen",
    "qwen2":        "qwen2",
    "starcoder":    "starcoder",
    "bloom":        "bloom",
    "stablelm":     "stablelm",
    "deepseek":     "deepseek",
    "deepseek2":    "deepseek2",
    "command-r":    "command-r",
    "internlm2":    "internlm2",
    "solar":        "solar",
    "yi":           "yi",
}

# ── Quantization label map ────────────────────────────────────────────────────

QUANT_LABELS = {
    "q2_k":   "Q2_K",
    "q3_k_s": "Q3_K_S",
    "q3_k_m": "Q3_K_M",
    "q3_k_l": "Q3_K_L",
    "q4_0":   "Q4_0",
    "q4_1":   "Q4_1",
    "q4_k_s": "Q4_K_S",
    "q4_k_m": "Q4_K_M",
    "q5_0":   "Q5_0",
    "q5_1":   "Q5_1",
    "q5_k_s": "Q5_K_S",
    "q5_k_m": "Q5_K_M",
    "q6_k":   "Q6_K",
    "q8_0":   "Q8_0",
    "f16":    "F16",
    "f32":    "F32",
    "bf16":   "BF16",
}

# ── GGUF quantization type integer map ────────────────────────────────────────────

GGUF_QUANT_INT = {
    0:  "F32",
    1:  "F16",
    2:  "F16",
    3:  "Q4_0",
    4:  "Q4_1",
    6:  "Q5_0",
    7:  "Q5_1",
    8:  "Q8_0",
    9:  "Q8_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "Q8_K",
    20: "IQ2_XXS",
    21: "IQ2_XS",
    22: "IQ3_XXS",
    23: "IQ1_S",
    24: "IQ4_NL",
    25: "IQ3_S",
    26: "IQ2_S",
    27: "IQ4_XS",
    28: "IQ1_M",
    29: "BF16",
}

# ── Recommended backend per format ───────────────────────────────────────────

BACKEND_MAP = {
    "gguf":        "llama.cpp",
    "safetensors": "vllm",
    "ggml":        "llama.cpp",
}

# ── Default inference parameters ─────────────────────────────────────────────

DEFAULT_PARAMS = {
    "temperature":    0.7,
    "top_p":          0.9,
    "top_k":          40,
    "repeat_penalty": 1.1,
    "max_tokens":     2048,
}


# ─────────────────────────────────────────────────────────────────────────────
# GGUFReader — binary header parser
# ─────────────────────────────────────────────────────────────────────────────

class GGUFReader:
    """
    Reads the binary GGUF header from a .gguf file.
    Extracts all metadata key/value pairs from the header.
    Does not load weights — header only.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.metadata = {}
        self._offset = 0
        self._data = None

    def read(self) -> dict:
        """Read and return all GGUF header metadata."""
        with open(self.path, "rb") as f:
            self._data = f.read(1024 * 1024)  # Read first 1MB — header only

        self._validate_magic()
        version      = self._read_uint32()
        tensor_count = self._read_uint64()
        kv_count     = self._read_uint64()

        if not (GGUF_VERSION_MIN <= version <= GGUF_VERSION_MAX):
            raise ValueError(f"Unsupported GGUF version: {version}")

        for _ in range(kv_count):
            try:
                key   = self._read_string()
                vtype = self._read_uint32()
                value = self._read_value(vtype)
                self.metadata[key] = value
            except Exception:
                break  # Partial read — stop gracefully

        return self.metadata

    # ── private readers ───────────────────────────────────────────────────────

    def _validate_magic(self):
        magic = self._data[self._offset:self._offset + 4]
        self._offset += 4
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {self.path.name}")

    def _read_uint8(self)  -> int: return self._unpack("B", 1)
    def _read_uint16(self) -> int: return self._unpack("H", 2)
    def _read_uint32(self) -> int: return self._unpack("I", 4)
    def _read_uint64(self) -> int: return self._unpack("Q", 8)
    def _read_int32(self)  -> int: return self._unpack("i", 4)
    def _read_int64(self)  -> int: return self._unpack("q", 8)
    def _read_float32(self) -> float: return self._unpack("f", 4)
    def _read_bool(self)   -> bool: return bool(self._unpack("B", 1))

    def _unpack(self, fmt: str, size: int):
        val = struct.unpack_from("<" + fmt, self._data, self._offset)[0]
        self._offset += size
        return val

    def _read_string(self) -> str:
        length = self._read_uint64()
        raw    = self._data[self._offset:self._offset + length]
        self._offset += length
        return raw.decode("utf-8", errors="replace")

    def _read_value(self, vtype: int):
        if vtype in GGUF_TYPE and GGUF_TYPE[vtype][1] is not None:
            _, fmt, size = GGUF_TYPE[vtype]
            return self._unpack(fmt, size)
        elif vtype == 8:   # STRING
            return self._read_string()
        elif vtype == 9:   # ARRAY
            item_type  = self._read_uint32()
            item_count = self._read_uint64()
            items = []
            for _ in range(item_count):
                try:
                    items.append(self._read_value(item_type))
                except Exception:
                    break
            return items
        else:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Detector — extracts clean model info from raw GGUF metadata
# ─────────────────────────────────────────────────────────────────────────────

class Detector:
    """
    Extracts clean, human-readable model information from GGUF metadata.
    Generates a TEE modelfile automatically.
    """

    def __init__(self, path: str):
        self.path     = Path(path)
        self.metadata = {}
        self.info     = {}

    def detect(self) -> dict:
        """
        Read the GGUF header and return clean model info.
        Returns a dict ready to write as a modelfile.
        """
        reader        = GGUFReader(str(self.path))
        self.metadata = reader.read()
        self.info     = self._extract()
        return self.info

    def generate_modelfile(self, output_path: str = None) -> str:
        """
        Generate a TEE modelfile JSON from detected info.
        Writes alongside the GGUF if output_path not given.
        Returns the path written.
        """
        if not self.info:
            self.detect()

        modelfile = {
            "name":         self.info["name"],
            "version":      self.info.get("version", "auto"),
            "description":  self.info.get("description", ""),
            "file":         self.path.name,
            "format":       "gguf",
            "architecture": self.info.get("architecture", "unknown"),
            "parameters":   self.info.get("parameters", "unknown"),
            "quantization": self.info.get("quantization", "unknown"),
            "context":      self.info.get("context", 4096),
            "backend":      "auto",
            "gpu":          "auto",
            "gpu_layers":   "auto",
            "tags":         self.info.get("tags", []),
            "defaults":     DEFAULT_PARAMS.copy(),
            "system_prompt": "",
            "generated":    True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source":       None,
        }

        if output_path is None:
            output_path = self.path.with_suffix(".modelfile.json")

        with open(output_path, "w") as f:
            json.dump(modelfile, f, indent=2)

        return str(output_path)

    # ── private extraction ────────────────────────────────────────────────────

    def _extract(self) -> dict:
        m    = self.metadata
        info = {}

        # Architecture
        arch_raw           = m.get("general.architecture", "unknown")
        info["architecture"] = ARCH_MAP.get(arch_raw.lower(), arch_raw.lower())

        # Parameter count
        param_keys = [
            f"{arch_raw}.block_count",
            "general.parameter_count",
        ]
        block_count    = m.get(f"{arch_raw}.block_count", 0)
        embed_dim      = m.get(f"{arch_raw}.embedding_length", 0)
        info["parameters"] = self._estimate_params(block_count, embed_dim)

        # Quantization — from filename if not in metadata
        quant_raw          = m.get("general.file_type", "")
        info["quantization"] = self._extract_quantization(quant_raw)

        # Context length
        ctx_keys = [
            f"{arch_raw}.context_length",
            "llama.context_length",
            "general.context_length",
        ]
        ctx = 4096
        for k in ctx_keys:
            if k in m and m[k]:
                ctx = int(m[k])
                break
        info["context"] = ctx

        # Name — from metadata or filename — include quantization for uniqueness
        base_name       = m.get("general.name", "") or self._name_from_file()
        quant_label     = info.get("quantization", "")
        if quant_label and quant_label != "unknown":
            info["name"] = f"{base_name} {quant_label}"
        else:
            info["name"] = base_name
        info["version"] = m.get("general.version", "auto")

        # Description
        info["description"] = m.get(
            "general.description",
            f"{info['name']} {info['parameters']} — auto-detected by TEE"
        )

        # Tags — infer from name and architecture
        info["tags"] = self._infer_tags(info)

        # File size
        info["size_gb"] = round(self.path.stat().st_size / (1024 ** 3), 2)

        return info

    def _estimate_params(self, block_count: int, embed_dim: int) -> str:
        """Estimate parameter count from block count and embedding dimension."""
        if not block_count or not embed_dim:
            return self._params_from_filename()

        # Rough approximation: ~12 * layers * hidden_dim^2
        approx = 12 * block_count * (embed_dim ** 2)
        b      = approx / 1e9

        if b < 1:   return f"{round(approx / 1e6)}M"
        if b < 10:  return f"{round(b, 1)}B"
        return      f"{round(b)}B"

    def _params_from_filename(self) -> str:
        """Extract parameter count from filename as fallback."""
        name  = self.path.name.lower()
        match = re.search(r"(\d+\.?\d*)[_\-]?b", name)
        if match:
            return f"{match.group(1)}B"
        return "unknown"

    def _extract_quantization(self, quant_raw) -> str:
        """Extract quantization level — metadata first, filename fallback."""
        if quant_raw is not None and quant_raw != "" and quant_raw != 0:
            if isinstance(quant_raw, int):
                if quant_raw in GGUF_QUANT_INT:
                    return GGUF_QUANT_INT[quant_raw]
            else:
                quant_str = str(quant_raw).strip()
                label = QUANT_LABELS.get(quant_str.lower())
                if label:
                    return label
                if quant_str:
                    return quant_str.upper()

        # Fallback — parse from filename
        name = self.path.stem.lower()
        for key, label in QUANT_LABELS.items():
            if key in name:
                return label

        return "unknown"

    def _name_from_file(self) -> str:
        """Generate a clean model name from the filename."""
        stem = self.path.stem
        # Strip quantization suffix
        for key in QUANT_LABELS:
            stem = re.sub(r"[_\-]?" + re.escape(key) + r"$", "", stem, flags=re.IGNORECASE)
        # Clean separators
        stem = re.sub(r"[_\-]+", "-", stem).strip("-").lower()
        return stem

    def _infer_tags(self, info: dict) -> list:
        """Infer capability tags from model name and architecture."""
        tags  = []
        name  = (info.get("name", "") + " " + info.get("description", "")).lower()
        arch  = info.get("architecture", "").lower()

        if any(w in name for w in ["instruct", "chat", "assistant"]):
            tags.append("instruct")
        if any(w in name for w in ["code", "coder", "starcoder", "deepseek-coder"]):
            tags.append("code")
        if any(w in name for w in ["math", "mathstral"]):
            tags.append("math")
        if any(w in name for w in ["vision", "llava", "moondream"]):
            tags.append("vision")
        if any(w in name for w in ["embed", "embedding"]):
            tags.append("embedding")
        if not tags:
            tags.append("general")

        return tags


# ─────────────────────────────────────────────────────────────────────────────
# CLI — python3 detector.py /path/to/model.gguf
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 detector.py /path/to/model.gguf")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"\nTEE Detector — reading {Path(path).name}")
    print("─" * 50)

    d    = Detector(path)
    info = d.detect()

    print(f"  Name:         {info['name']}")
    print(f"  Architecture: {info['architecture']}")
    print(f"  Parameters:   {info['parameters']}")
    print(f"  Quantization: {info['quantization']}")
    print(f"  Context:      {info['context']:,} tokens")
    print(f"  Size:         {info['size_gb']} GB")
    print(f"  Tags:         {', '.join(info['tags'])}")

    out = d.generate_modelfile()
    print(f"\n✓ Modelfile written → {out}")
