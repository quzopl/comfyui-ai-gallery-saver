"""Save Image (Rich Metadata) — ComfyUI v3 node with unlimited IMAGE inputs.

Uses ComfyUI v3 API (`comfy_api.latest`) with `Autogrow` so the user can plug
in as many independent image batches as they want (framework caps at 100).

Each batch is saved with three PNG tEXt chunks:
  - ai_gallery_meta : clean JSON (authoritative; consumed by AI Gallery app)
  - prompt + workflow : standard ComfyUI (round-trip)
  - parameters     : A1111-compatible (CivitAI / webui)

Per-slot filename auto-suffix: slot 1 uses `filename_prefix`, slot N>1 uses
`filename_prefix_N`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo
from typing_extensions import override

import folder_paths
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io


# ---------- canonical metadata extraction ----------

def _int_or_none(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _resolve(graph: dict, value: Any) -> dict | None:
    """If `value` is a link [src_id, out_idx], return the source node dict."""
    if isinstance(value, list) and len(value) == 2:
        node = graph.get(str(value[0]))
        if isinstance(node, dict):
            return node
    return None


def _parse_json_list(s: Any) -> list:
    if isinstance(s, str) and s:
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _ig4_dumps(v: Any, lvl: int = 0) -> str:
    """Mirror Ideogram4PromptBuilderKJ's serializer: indent=4 but scalar arrays inline."""
    pad, end = "    " * (lvl + 1), "    " * lvl
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in v):
            return "[" + ", ".join(_ig4_dumps(x, lvl) for x in v) + "]"
        return "[\n" + ",\n".join(pad + _ig4_dumps(x, lvl + 1) for x in v) + "\n" + end + "]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = [pad + json.dumps(k, ensure_ascii=False) + ": " + _ig4_dumps(val, lvl + 1) for k, val in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    return json.dumps(v, ensure_ascii=False)


def _ig4_norm_bbox(box: dict) -> list:
    def c(v):
        return max(0, min(1000, round(v * 1000)))
    x, y, w, h = box.get("x", 0.0), box.get("y", 0.0), box.get("w", 0.0), box.get("h", 0.0)
    ymin, xmin, ymax, xmax = c(y), c(x), c(y + h), c(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]


def _reconstruct_ideogram4(inputs: dict) -> str:
    """Rebuild the caption JSON that Ideogram4PromptBuilderKJ emits at runtime.

    The builder computes its prompt string from its widget inputs, so it is not
    present as a static `text` value anywhere in the graph. Its inputs *are*
    in the graph, so we reproduce the same assembly to recover the real prompt.
    """
    def s(k: str) -> str:
        v = inputs.get(k, "")
        return v if isinstance(v, str) else ""

    caption: dict = {}
    if s("high_level_description").strip():
        caption["high_level_description"] = s("high_level_description")

    kind = s("style") or "none"
    if kind != "none":
        sd: dict = {"aesthetics": s("aesthetics"), "lighting": s("lighting")}
        if kind == "photo":
            sd["photo"] = s("style.photo")
            sd["medium"] = s("medium")
        else:
            sd["medium"] = s("medium")
            sd["art_style"] = s("style.art_style")
        palette = [c.upper() for c in _parse_json_list(s("style_palette_data")) if c]
        if palette:
            sd["color_palette"] = palette
        caption["style_description"] = sd

    elements = []
    for box in _parse_json_list(s("elements_data")):
        if not isinstance(box, dict):
            continue
        etype = "text" if box.get("type") == "text" else "obj"
        elem: dict = {"type": etype}
        if not box.get("nobbox"):
            elem["bbox"] = _ig4_norm_bbox(box)
        if etype == "text":
            elem["text"] = box.get("text", "")
        elem["desc"] = box.get("desc", "")
        pal = [c.upper() for c in (box.get("palette") or []) if c]
        if pal:
            elem["color_palette"] = pal[:5]
        elements.append(elem)

    caption["compositional_deconstruction"] = {"background": s("background"), "elements": elements}
    return _ig4_dumps(caption)


# ---- graph text tracing ---------------------------------------------------
# Input keys that carry (or lead to) prompt text, in preference order. Both
# literal strings and links are tried; the first key that yields text wins.
_TEXT_KEYS = (
    "text", "populated_text", "text_positive", "positive_prompt", "prompt",
    "t5xxl", "clip_l", "value", "string", "wildcard_text", "source",
    "conditioning", "positive", "negative", "text_g", "text_l",
)
# Cached display widgets of ShowText/showAnything-style nodes: hold the LAST
# value the node displayed, i.e. the real output of an upstream generator.
_CACHED_KEYS = ("text_0", "text_1", "text2", "text_2")
# Concatenation nodes: (ordered part keys, delimiter key)
_CONCAT_KEYS = (
    (("text_a", "text_b", "text_c", "text_d"), "delimiter"),
    (("string_a", "string_b"), "delimiter"),
)
_GENERATOR_CLASSES = ("TextGenerate", "Florence2Run", "LLM", "Ollama", "Joy",
                      "Qwen2VL", "Qwen3VL", "Caption", "Describe", "VLM")
_GENERATOR_INPUT_HINTS = ("max_length", "max_new_tokens", "max_tokens",
                          "temperature", "sampling_mode")
_MAX_TRACE_DEPTH = 24


def _is_generator(ct: str, inp: dict) -> bool:
    """Node whose text output is produced at run time (LLM/VLM) and thus not
    stored in the graph. Tracing into its inputs would return instructions,
    not the prompt, so it resolves to None (or a cached ShowText value)."""
    if any(h.lower() in ct.lower() for h in _GENERATOR_CLASSES):
        return True
    return any(k in inp or k.split(".")[0] in inp for k in _GENERATOR_INPUT_HINTS) and (
        "prompt" in inp or "text" in inp)


def _cached_display(graph: dict, link: Any) -> str | None:
    """Cached widget text of any node that consumes `link` (ShowText|pysssss
    `text_0`, easy showAnything `text`, …)."""
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        inp = node.get("inputs") or {}
        if not any(v == link for v in inp.values()):
            continue
        for key in _CACHED_KEYS + ("text",):
            v = inp.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return None


def _get_text_recursive(graph: dict, value: Any, depth: int = 0) -> str | None:
    """Walk a link to the text that ultimately feeds a conditioning input.

    Handles literal strings, CLIPTextEncode `text`, Flux dual encoders
    (`t5xxl`/`clip_l`), string primitives (`value`/`string`/`prompt`),
    pass-through nodes (PreviewAny `source`), boolean routers (ComfySwitchNode /
    Crystools `switch`/`boolean` — literal or linked to a PrimitiveBoolean),
    rgthree Any Switch (`any_NN`), Text/String Concatenate (joined with the
    node's delimiter), ShowText-style cached widgets, the Ideogram 4 builder /
    bbox editor, conditioning pass-throughs with (positive, negative) in and
    out (slot picks the side), and nested conditioning. ConditioningZeroOut →
    empty (zeroed branch). LLM/VLM generators (TextGenerate, Florence2Run…)
    resolve to a cached ShowText value or None, so a router falls back to its
    other branch — usually the user's raw prompt."""
    if depth > _MAX_TRACE_DEPTH:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    node = _resolve(graph, value)
    if node is None:
        return None
    ct = node.get("class_type", "")
    if ct == "ConditioningZeroOut":
        return None
    inp = node.get("inputs") or {}
    if ct == "Ideogram4PromptBuilderKJ":
        return _reconstruct_ideogram4(inp)
    if ct == "Ideogram4BboxEditor":
        # The bbox editor assembles its caption in the frontend and stores it
        # in the `caption_json` widget, so that string *is* the prompt.
        cj = inp.get("caption_json")
        if isinstance(cj, str) and cj.strip() and cj.strip() != "{}":
            return cj
        return None
    if _is_generator(ct, inp):
        return _cached_display(graph, value)
    # conditioning pass-throughs with (positive, negative) in AND out
    # (LTXVConditioning, WanImageToVideo, …): output slot picks the side
    if "positive" in inp and "negative" in inp:
        side = "negative" if value[1] == 1 else "positive"
        return _get_text_recursive(graph, inp[side], depth + 1)
    # boolean routers: prefer the active branch (switch literal or linked
    # PrimitiveBoolean), but fall back to the other one if it yields no text
    if "on_true" in inp or "on_false" in inp:
        sw = inp.get("switch", inp.get("boolean"))
        if not isinstance(sw, bool):
            sw_node = _resolve(graph, sw)
            sw = (sw_node.get("inputs") or {}).get("value") if sw_node else None
        order = ("on_false", "on_true") if sw is False else ("on_true", "on_false")
        for b in order:
            if b in inp:
                t = _get_text_recursive(graph, inp[b], depth + 1)
                if t:
                    return t
        return None
    # concatenation nodes: join every part that resolves
    for part_keys, delim_key in _CONCAT_KEYS:
        if any(k in inp for k in part_keys):
            parts = [_get_text_recursive(graph, inp[k], depth + 1) for k in part_keys if k in inp]
            parts = [t.strip() for t in parts if t and t.strip()]
            if parts:
                delim = inp.get(delim_key)
                return (delim if isinstance(delim, str) else " ").join(parts)
            return None
    lower = {k.lower(): k for k in inp}
    any_keys = sorted(k for k in lower if k.startswith("any_"))
    for key in list(_TEXT_KEYS) + any_keys:
        if key in lower:
            t = _get_text_recursive(graph, inp[lower[key]], depth + 1)
            if t:
                return t
    for key in _CACHED_KEYS:
        v = inp.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return None


# ---- sampler settings -------------------------------------------------------
_SCALAR_KEYS = ("value", "seed", "noise_seed", "Number", "number", "int", "float", "String", "string")


def _scalar(graph: dict, v: Any, depth: int = 0) -> Any:
    """Literal value of a sampler input; follows a link into a primitive node
    (PrimitiveFloat `value`, Seed (rgthree) `seed`, Float `Number`, …)."""
    if not isinstance(v, list):
        return v
    node = _resolve(graph, v)
    if node is None or depth > 4:
        return None
    inp = node.get("inputs") or {}
    for k in _SCALAR_KEYS:
        if k in inp:
            return _scalar(graph, inp[k], depth + 1)
    return None


def _fill(out: dict, key: str, value: Any) -> None:
    if out.get(key) is None and value is not None:
        out[key] = value


def _sampler_conditioning(graph: dict) -> tuple[Any, Any]:
    """(positive, negative) link refs feeding the first sampler, following a
    guider node for SamplerCustom* graphs (BasicGuider has one `conditioning`)."""
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        if "KSampler" in ct or "SamplerCustom" in ct:
            inp = node.get("inputs") or {}
            pos, neg = inp.get("positive"), inp.get("negative")
            if pos is None and neg is None and "guider" in inp:
                guider = _resolve(graph, inp.get("guider"))
                if guider:
                    gin = guider.get("inputs") or {}
                    pos, neg = gin.get("positive"), gin.get("negative")
                    if pos is None:
                        pos = gin.get("conditioning")
            return pos, neg
    return None, None


def _sampler_fields(graph: dict, out: dict) -> None:
    """Sampler settings: first from sampler nodes, then from the helper nodes
    of SamplerCustom* graphs (RandomNoise / BasicScheduler / *Guider /
    KSamplerSelect). Linked values are followed into primitive nodes."""
    nodes = [n for n in graph.values() if isinstance(n, dict)]

    def take(node: dict) -> None:
        inp = node.get("inputs") or {}
        s = _scalar(graph, inp.get("sampler_name"))
        _fill(out, "sampler", s if isinstance(s, str) and s else None)
        _fill(out, "steps", _int_or_none(_scalar(graph, inp.get("steps"))))
        _fill(out, "cfg", _float_or_none(_scalar(graph, inp.get("cfg"))))
        _fill(out, "seed", _int_or_none(_scalar(graph, inp.get("seed", inp.get("noise_seed")))))

    for node in nodes:
        ct = node.get("class_type", "")
        if "Sampler" in ct or ct.startswith("KSampler"):
            take(node)
    for node in nodes:
        ct = node.get("class_type", "")
        if ct in ("RandomNoise", "BasicScheduler") or "Scheduler" in ct or "Guider" in ct:
            take(node)


# ---- model / LoRA -----------------------------------------------------------
def _model_name(graph: dict) -> str | None:
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        if ("Checkpoint" in ct or "UNetLoader" in ct or "UNETLoader" in ct
                or "ModelLoader" in ct):
            inp = node.get("inputs") or {}
            for k in ("ckpt_name", "unet_name", "model_name", "model"):
                v = inp.get(k)
                if isinstance(v, str):
                    return v
    return None


def _lora_name(v: Any) -> str | None:
    """LoRA file name from a literal string or a {content: ...} widget dict.
    Placeholders ('None', '') → None."""
    if isinstance(v, dict):
        v = v.get("content") or v.get("lora") or v.get("name")
    if not isinstance(v, str) or not v.strip() or v.strip().lower() == "none":
        return None
    return v


def _loras(graph: dict) -> list[dict]:
    """LoRAs from every loader flavour: plain LoraLoader*, rgthree Power Lora
    Loader (`lora_N` dicts with on/lora/strength), rgthree Lora Loader Stack
    (`lora_NN` + `strength_NN`), CR LoRA Stack (`lora_name_N` + `switch_N` +
    `model_weight_N`), LoraLoaderStackedAdvanced (`lora_name` dict +
    `lora_weight`). Only enabled entries, no duplicates."""
    out: list[dict] = []
    seen: set[str] = set()

    def add(name: Any, strength: Any) -> None:
        n = _lora_name(name)
        if n and n not in seen:
            seen.add(n)
            out.append({"name": n, "strength": _float_or_none(strength)})

    for node in graph.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        if "lora" not in ct.lower():
            continue
        inp = node.get("inputs") or {}
        for k, v in inp.items():                       # rgthree Power Lora Loader
            if isinstance(v, dict) and "lora" in v and v.get("on", True):
                add(v.get("lora"), v.get("strength"))
        for k, v in inp.items():                       # rgthree Lora Loader Stack
            m = re.fullmatch(r"lora_(\d+)", k)
            if m and isinstance(v, str):
                add(v, inp.get(f"strength_{m.group(1)}"))
        for k, v in inp.items():                       # CR LoRA Stack & friends
            m = re.fullmatch(r"lora_name_(\d+)", k)
            if m and str(inp.get(f"switch_{m.group(1)}", "On")).lower() != "off":
                add(v, inp.get(f"model_weight_{m.group(1)}", inp.get(f"lora_wt_{m.group(1)}",
                                inp.get(f"model_str_{m.group(1)}"))))
        if "lora_name" in inp or "name" in inp:        # single loaders
            add(inp.get("lora_name", inp.get("name")),
                inp.get("strength_model", inp.get("strength", inp.get("lora_weight"))))
    return out


def extract_canonical(graph: dict, width: int, height: int) -> dict:
    """Walk the execution graph and pull out clean fields."""
    out: dict = {
        "version": 1,
        "source": "comfyui-save-image-rich-metadata",
        "prompt": None,
        "negative": None,
        "model_name": None,
        "sampler": None,
        "steps": None,
        "cfg": None,
        "seed": None,
        "loras": [],
        "width": width,
        "height": height,
        "generated_at": int(time.time()),
    }
    if not isinstance(graph, dict):
        return out
    pos, neg = _sampler_conditioning(graph)
    p = _get_text_recursive(graph, pos) if pos is not None else None
    n = _get_text_recursive(graph, neg) if neg is not None else None
    # Same text feeding both inputs (one Flux encoder wired to positive and
    # negative) is not a negative prompt.
    if p and n and p.strip() == n.strip():
        n = None
    out["prompt"], out["negative"] = p, n
    _sampler_fields(graph, out)
    out["model_name"] = _model_name(graph)
    out["loras"] = _loras(graph)
    return out


def _apply_overrides(meta: dict, *, prompt_text: str | None, negative_text: str | None) -> dict:
    """Explicit `prompt_text` / `negative_text` node inputs beat graph
    extraction — the only way to record text produced at run time (LLM
    prompt expanders, wildcards, captioners). Blank values are ignored."""
    if isinstance(prompt_text, str) and prompt_text.strip():
        meta["prompt"] = prompt_text
    if isinstance(negative_text, str) and negative_text.strip():
        meta["negative"] = negative_text
    return meta


# ---------- CivitAI resource hashes (AutoV2 = first 12 hex of SHA256) --------

_HASH_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hash_cache.json")
_HASH_CACHE: dict | None = None


def _autov2(sha_hex: Any) -> str:
    return sha_hex[:12] if isinstance(sha_hex, str) else ""


def _lora_key(name: str) -> str:
    """The lora name as used in <lora:NAME:..> / Hashes: basename, no extension."""
    base = os.path.basename(str(name).replace("\\", "/"))
    return os.path.splitext(base)[0]


def _lora_hashes_str(loras: list) -> str:
    parts = [f"{_lora_key(l['name'])}: {l['hash']}" for l in (loras or []) if l.get("hash")]
    return ", ".join(parts)


def _hashes_json(meta: dict) -> dict:
    out: dict = {}
    if meta.get("model_hash"):
        out["model"] = meta["model_hash"]
    for l in meta.get("loras") or []:
        if l.get("hash"):
            out[f"lora:{_lora_key(l['name'])}"] = l["hash"]
    return out


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_file(path: str, cache: dict) -> str | None:
    """Full SHA256 of `path`, cached by path+size+mtime. None if missing/unreadable."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    if key in cache:
        return cache[key]
    try:
        digest = _sha256_file(path)
    except OSError:
        return None
    cache[key] = digest
    return digest


def _load_hash_cache() -> dict:
    global _HASH_CACHE
    if _HASH_CACHE is None:
        try:
            with open(_HASH_CACHE_FILE, "r", encoding="utf-8") as f:
                _HASH_CACHE = json.load(f)
        except (OSError, ValueError):
            _HASH_CACHE = {}
    return _HASH_CACHE


def _save_hash_cache() -> None:
    try:
        with open(_HASH_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_HASH_CACHE or {}, f)
    except OSError:
        pass


def _resolve_model_path(name: str, folders: tuple) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    for folder in folders:
        try:
            p = folder_paths.get_full_path(folder, name)
        except Exception:
            p = None
        if p and os.path.exists(p):
            return p
    return None


def _augment_with_hashes(meta: dict) -> dict:
    """Fill meta['model_hash'] and per-lora 'hash' (AutoV2), using a persistent
    cache so big model files are hashed only once."""
    cache = _load_hash_cache()
    before = len(cache)

    mp = _resolve_model_path(meta.get("model_name"), ("checkpoints", "diffusion_models", "unet"))
    if mp:
        full = _hash_file(mp, cache)
        if full:
            meta["model_hash"] = _autov2(full)

    for lora in meta.get("loras") or []:
        lp = _resolve_model_path(lora.get("name"), ("loras",))
        if lp:
            full = _hash_file(lp, cache)
            if full:
                lora["hash"] = _autov2(full)

    if len(cache) != before:
        _save_hash_cache()
    return meta


# ---------- A1111 parameters formatting ----------

def _format_a1111_parameters(meta: dict) -> str:
    parts: list[str] = []
    pos = meta.get("prompt") or ""
    for lora in meta.get("loras") or []:
        s = lora.get("strength")
        if s is None:
            s = 1.0
        pos = pos.rstrip() + f" <lora:{lora['name']}:{s}>"
    parts.append(pos.strip())
    if meta.get("negative"):
        parts.append(f"Negative prompt: {meta['negative']}")
    kv: list[str] = []
    if meta.get("steps") is not None:
        kv.append(f"Steps: {meta['steps']}")
    if meta.get("sampler"):
        kv.append(f"Sampler: {meta['sampler']}")
    if meta.get("cfg") is not None:
        kv.append(f"CFG scale: {meta['cfg']}")
    if meta.get("seed") is not None:
        kv.append(f"Seed: {meta['seed']}")
    if meta.get("width") and meta.get("height"):
        kv.append(f"Size: {meta['width']}x{meta['height']}")
    if meta.get("model_hash"):
        kv.append(f"Model hash: {meta['model_hash']}")
    if meta.get("model_name"):
        kv.append(f"Model: {meta['model_name']}")
    lora_hashes = _lora_hashes_str(meta.get("loras") or [])
    if lora_hashes:
        kv.append(f'Lora hashes: "{lora_hashes}"')
    # `Hashes` is JSON and must be LAST so its commas aren't read as kv separators.
    hashes = _hashes_json(meta)
    if hashes:
        kv.append("Hashes: " + json.dumps(hashes))
    if kv:
        parts.append(", ".join(kv))
    return "\n".join(parts)


# ---------- ComfyUI v3 node ----------

class SaveImageRichMetadata(io.ComfyNode):
    """Save Image (Rich Metadata) — saves PNG with rich, multi-format metadata.

    Has one Autogrow image input — connect as many image batches as you need
    (framework cap: 100). Each connected batch is saved with the shared
    metadata extracted from the workflow; per-slot filenames auto-suffix
    `_2`, `_3`, ... onto the main `filename_prefix`.
    """

    @classmethod
    def define_schema(cls):
        autogrow_template = _io.Autogrow.TemplatePrefix(
            input=io.Image.Input("img"),
            prefix="img_",
            min=1,
            max=100,
        )
        return io.Schema(
            node_id="SaveImageRichMetadata",
            display_name="Save Image (Rich Metadata)",
            category="image",
            description=(
                "Saves images with clean canonical JSON metadata + standard "
                "ComfyUI prompt/workflow + A1111-compatible parameters "
                "(CivitAI-ready). Unlimited image input slots (up to "
                "framework cap 100)."
            ),
            inputs=[
                io.String.Input(
                    "filename_prefix",
                    default="AIGal",
                    tooltip=(
                        "Filename prefix for the first image batch. Extra "
                        "batches auto-suffix '_2', '_3', ..."
                    ),
                ),
                io.Boolean.Input(
                    "embed_workflow",
                    default=True,
                    tooltip="Also embed standard ComfyUI 'prompt'+'workflow' chunks.",
                    optional=True,
                ),
                io.Boolean.Input(
                    "embed_a1111",
                    default=True,
                    tooltip="Also embed A1111-compatible 'parameters' chunk.",
                    optional=True,
                ),
                io.String.Input(
                    "prompt_text",
                    optional=True,
                    force_input=True,
                    tooltip=(
                        "Optional. Connect the STRING that actually conditioned "
                        "the image (e.g. the output of an LLM prompt expander, "
                        "wildcard processor or switch). Overrides the prompt "
                        "recovered from the workflow graph — use it whenever the "
                        "prompt is generated at run time."
                    ),
                ),
                io.String.Input(
                    "negative_text",
                    optional=True,
                    force_input=True,
                    tooltip="Optional. Explicit negative prompt STRING; overrides graph extraction.",
                ),
                _io.Autogrow.Input("images", template=autogrow_template),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        filename_prefix: str,
        embed_workflow: bool,
        embed_a1111: bool,
        images: _io.Autogrow.Type,
        prompt_text: str | None = None,
        negative_text: str | None = None,
    ) -> io.NodeOutput:
        # images is dict {img_0: batch_tensor, img_1: batch_tensor, ...}
        prompt = cls.hidden.prompt if cls.hidden else None
        extra_pnginfo = cls.hidden.extra_pnginfo if cls.hidden else None

        all_results: list[dict] = []
        slot_idx = 0
        for slot_name in sorted(images.keys()):
            batch = images.get(slot_name)
            if batch is None or len(batch) == 0:
                slot_idx += 1
                continue
            slot_prefix = filename_prefix if slot_idx == 0 else f"{filename_prefix}_{slot_idx + 1}"
            results = cls._save_batch(
                batch, slot_prefix,
                embed_workflow=embed_workflow, embed_a1111=embed_a1111,
                prompt=prompt, extra_pnginfo=extra_pnginfo,
                prompt_text=prompt_text, negative_text=negative_text,
            )
            all_results.extend(results)
            slot_idx += 1

        return io.NodeOutput(ui={"images": all_results})

    @classmethod
    def _save_batch(
        cls,
        images,
        filename_prefix: str,
        *,
        embed_workflow: bool,
        embed_a1111: bool,
        prompt: dict | None,
        extra_pnginfo: dict | None,
        prompt_text: str | None = None,
        negative_text: str | None = None,
    ) -> list[dict]:
        output_dir = folder_paths.get_output_directory()
        h, w = images[0].shape[0], images[0].shape[1]
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(filename_prefix, output_dir, w, h)
        )
        # Same for every image in the batch; hash resource files once (cached).
        meta = extract_canonical(prompt or {}, w, h)
        _apply_overrides(meta, prompt_text=prompt_text, negative_text=negative_text)
        _augment_with_hashes(meta)

        out: list[dict] = []
        for image in images:
            arr = 255.0 * image.cpu().numpy()
            img = PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

            png_info = PngInfo()
            png_info.add_text("ai_gallery_meta", json.dumps(meta, ensure_ascii=False))

            if embed_workflow:
                if prompt is not None:
                    png_info.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo:
                    for k, v in extra_pnginfo.items():
                        png_info.add_text(k, json.dumps(v))

            if embed_a1111:
                params_str = _format_a1111_parameters(meta)
                if params_str.strip():
                    png_info.add_text("parameters", params_str)

            file = f"{filename}_{counter:05d}_.png"
            path = os.path.join(full_output_folder, file)
            img.save(path, pnginfo=png_info, compress_level=4)
            out.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1
        return out


# ---------- v3 extension entrypoint ----------

class SaveImageRichMetadataExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveImageRichMetadata]


async def comfy_entrypoint() -> SaveImageRichMetadataExtension:
    return SaveImageRichMetadataExtension()


