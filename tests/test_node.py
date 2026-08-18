import sys
import types
import hashlib
import importlib.util
from pathlib import Path

# Stub the ComfyUI-only imports so nodes.py loads standalone; the helpers under
# test only use json/os/hashlib.
for name in ("folder_paths", "numpy"):
    sys.modules.setdefault(name, types.ModuleType(name))
pil = types.ModuleType("PIL"); pil.Image = types.SimpleNamespace(); sys.modules.setdefault("PIL", pil)
pilp = types.ModuleType("PIL.PngImagePlugin"); pilp.PngInfo = object; sys.modules.setdefault("PIL.PngImagePlugin", pilp)
ca = types.ModuleType("comfy_api"); cal = types.ModuleType("comfy_api.latest")
cal.ComfyExtension = object
cal.io = types.SimpleNamespace(ComfyNode=object)
cal._io = types.SimpleNamespace()
sys.modules.setdefault("comfy_api", ca); sys.modules.setdefault("comfy_api.latest", cal)
te = types.ModuleType("typing_extensions"); te.override = lambda f: f; sys.modules.setdefault("typing_extensions", te)

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("srm_nodes", ROOT / "nodes.py")
srm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srm)


# ---- AutoV2 ----------------------------------------------------------------
def test_autov2_first_12():
    assert srm._autov2("0123456789abcdef0123") == "0123456789ab"
    assert srm._autov2("") == ""
    assert srm._autov2(None) == ""


# ---- lora key + hashes string ---------------------------------------------
def test_lora_key_basename_no_ext():
    assert srm._lora_key("ideogram/plener.safetensors") == "plener"
    assert srm._lora_key("detail.safetensors") == "detail"
    assert srm._lora_key("sub\\dir\\x.pt") == "x"


def test_lora_hashes_str_skips_unhashed():
    loras = [
        {"name": "ideogram/plener.safetensors", "hash": "aaaaaaaaaaaa"},
        {"name": "no_hash.safetensors"},
        {"name": "b.safetensors", "hash": "bbbbbbbbbbbb"},
    ]
    assert srm._lora_hashes_str(loras) == "plener: aaaaaaaaaaaa, b: bbbbbbbbbbbb"
    assert srm._lora_hashes_str([]) == ""


def test_hashes_json():
    meta = {"model_hash": "deadbeef0000",
            "loras": [{"name": "x/plener.safetensors", "hash": "aaaaaaaaaaaa"},
                      {"name": "no.safetensors"}]}
    assert srm._hashes_json(meta) == {"model": "deadbeef0000", "lora:plener": "aaaaaaaaaaaa"}


def test_hashes_json_empty_when_no_hashes():
    assert srm._hashes_json({"loras": [{"name": "a"}]}) == {}


# ---- A1111 parameters with / without hashes --------------------------------
def _base_meta():
    return {"prompt": "a cat", "negative": "", "steps": 20, "sampler": "euler",
            "cfg": 7.0, "seed": 42, "width": 512, "height": 512, "model_name": "realism",
            "loras": []}


def test_parameters_without_hashes_has_no_hash_fields():
    out = srm._format_a1111_parameters(_base_meta())
    assert out.startswith("a cat")
    assert "Model hash:" not in out and "Hashes:" not in out and "Lora hashes:" not in out


def test_parameters_with_hashes_includes_civitai_fields():
    meta = _base_meta()
    meta["model_hash"] = "deadbeef0000"
    meta["loras"] = [{"name": "ideogram/plener.safetensors", "strength": 0.5, "hash": "aaaaaaaaaaaa"}]
    out = srm._format_a1111_parameters(meta)
    assert out.startswith("a cat")
    assert "<lora:ideogram/plener.safetensors:0.5>" in out
    assert "Model hash: deadbeef0000" in out
    assert 'Lora hashes: "plener: aaaaaaaaaaaa"' in out
    assert '"model": "deadbeef0000"' in out and '"lora:plener": "aaaaaaaaaaaa"' in out
    # Hashes is the last field (its JSON commas must not be parsed as kv separators)
    assert out.rstrip().rfind("Hashes:") > out.rfind("Model hash:")


# ---- file hashing + cache --------------------------------------------------
def test_hash_file_matches_hashlib_and_caches(tmp_path):
    p = tmp_path / "model.bin"
    data = b"hello world" * 1000
    p.write_bytes(data)
    cache = {}
    h1 = srm._hash_file(str(p), cache)
    assert h1 == hashlib.sha256(data).hexdigest()
    assert len(cache) == 1
    # poison the cache to prove the 2nd call is a cache hit, not a recompute
    for k in list(cache):
        cache[k] = "POISONED"
    assert srm._hash_file(str(p), cache) == "POISONED"


def test_hash_file_invalidates_on_change(tmp_path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"aaaa")
    cache = {}
    srm._hash_file(str(p), cache)
    for k in list(cache):
        cache[k] = "POISONED"
    p.write_bytes(b"aaaabbbb")  # size + mtime change -> new key -> recompute
    h = srm._hash_file(str(p), cache)
    assert h == hashlib.sha256(b"aaaabbbb").hexdigest()


def test_hash_file_missing_returns_none():
    assert srm._hash_file("/no/such/file.bin", {}) is None


# ---- graph extraction: real-world ComfyUI patterns ------------------------
def _canon(graph: dict) -> dict:
    return srm.extract_canonical(graph, 8, 8)


def test_prompt_through_preview_any_and_llm_switch_falls_back_to_user_text():
    """PreviewAny(source) → ComfySwitchNode(switch linked to PrimitiveBoolean=True)
    → on_true TextGenerate (LLM, output not in graph). Must pass through
    PreviewAny and fall back to the on_false branch (the user's raw prompt);
    seed linked to Seed (rgthree) must resolve; Power Lora Loader honoured."""
    g = {
        "53": {"class_type": "KSampler", "inputs": {
            "seed": ["76", 0], "steps": 13, "cfg": 1.0, "sampler_name": "euler",
            "positive": ["79", 0], "negative": ["58", 0]}},
        "58": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["79", 0]}},
        "60": {"class_type": "TextGenerate", "inputs": {"prompt": ["61", 0], "max_length": 512}},
        "61": {"class_type": "StringConcatenate",
               "inputs": {"string_a": ["62", 0], "string_b": ["63", 0], "delimiter": ""}},
        "62": {"class_type": "PrimitiveStringMultiline",
               "inputs": {"value": "You are an expert prompt engineer. Expand the prompt."}},
        "63": {"class_type": "PrimitiveStringMultiline",
               "inputs": {"value": "portrait of a man by a rooftop infinity pool"}},
        "65": {"class_type": "ComfySwitchNode",
               "inputs": {"switch": ["68", 0], "on_false": ["63", 0], "on_true": ["60", 0]}},
        "68": {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
        "76": {"class_type": "Seed (rgthree)", "inputs": {"seed": 876576531857229}},
        "78": {"class_type": "PreviewAny", "inputs": {"source": ["65", 0]}},
        "79": {"class_type": "CLIPTextEncode", "inputs": {"text": ["78", 0]}},
        "83": {"class_type": "Power Lora Loader (rgthree)", "inputs": {
            "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
            "lora_1": {"on": True, "lora": "krea2/bart.safetensors", "strength": 1},
            "lora_2": {"on": False, "lora": "krea2/off.safetensors", "strength": 0.7},
            "➕ Add Lora": ""}},
    }
    m = _canon(g)
    assert m["prompt"] == "portrait of a man by a rooftop infinity pool"
    assert m["negative"] is None
    assert m["seed"] == 876576531857229
    assert m["steps"] == 13 and m["cfg"] == 1.0 and m["sampler"] == "euler"
    assert m["loras"] == [{"name": "krea2/bart.safetensors", "strength": 1.0}]


def test_sampler_custom_advanced_helper_nodes_and_basic_guider():
    g = {
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["25", 0], "guider": ["22", 0], "sampler": ["16", 0], "sigmas": ["17", 0]}},
        "25": {"class_type": "RandomNoise", "inputs": {"noise_seed": 4242}},
        "22": {"class_type": "BasicGuider", "inputs": {"conditioning": ["6", 0]}},
        "16": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "dpmpp_2m"}},
        "17": {"class_type": "BasicScheduler", "inputs": {"scheduler": "beta", "steps": 28}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["7", 0]}},
        "7": {"class_type": "CR Prompt Text", "inputs": {"prompt": "a dog on a beach"}},
    }
    m = _canon(g)
    assert m["prompt"] == "a dog on a beach"
    assert m["seed"] == 4242 and m["steps"] == 28 and m["sampler"] == "dpmpp_2m"


def test_flux_dual_encoder_and_same_node_for_negative():
    g = {
        "6": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 3.5,
              "sampler_name": "euler", "positive": ["4", 0], "negative": ["4", 0]}},
        "4": {"class_type": "CLIPTextEncodeFlux",
              "inputs": {"clip_l": "man", "t5xxl": "man in a black polo shirt", "guidance": 3.5}},
    }
    m = _canon(g)
    assert m["prompt"] == "man in a black polo shirt"
    assert m["negative"] is None


def test_showtext_cached_output_of_generator_and_text_concatenate():
    g = {
        "S": {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["G", 0]}},
        "G": {"class_type": "BasicGuider", "inputs": {"conditioning": ["E", 0]}},
        "E": {"class_type": "CLIPTextEncode", "inputs": {"text": ["C", 0]}},
        "C": {"class_type": "Text Concatenate", "inputs": {
            "delimiter": ", ", "text_a": ["F", 2], "text_b": "cinematic"}},
        "F": {"class_type": "Florence2Run", "inputs": {"max_new_tokens": 1024}},
        "X": {"class_type": "ShowText|pysssss",
              "inputs": {"text": ["F", 2], "text_0": "The image shows a woman in a suit"}},
    }
    assert _canon(g)["prompt"] == "The image shows a woman in a suit, cinematic"


def test_conditioning_passthrough_slot_and_rgthree_switches():
    g = {
        "S": {"class_type": "SamplerCustom", "inputs": {"positive": ["C", 0], "negative": ["C", 1]}},
        "C": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["P", 0], "negative": ["N", 0], "frame_rate": 25}},
        "P": {"class_type": "CLIPTextEncode", "inputs": {"text": ["A", 0]}},
        "A": {"class_type": "Any Switch (rgthree)", "inputs": {"any_01": ["W", 0], "any_03": ["L", 0]}},
        "W": {"class_type": "Switch any [Crystools]",
              "inputs": {"boolean": True, "on_true": ["L", 0], "on_false": ["X", 0]}},
        "L": {"class_type": "String", "inputs": {"String": "storm waves and lightning"}},
        "X": {"class_type": "String", "inputs": {"String": ""}},
        "N": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality, worst quality"}},
    }
    m = _canon(g)
    assert m["prompt"] == "storm waves and lightning"
    assert m["negative"] == "low quality, worst quality"


def test_lora_loader_stack_rgthree_and_stacked_advanced_dict_name():
    g = {
        "5": {"class_type": "Lora Loader Stack (rgthree)", "inputs": {
            "lora_01": "a.safetensors", "strength_01": 1.0,
            "lora_02": "None", "strength_02": 1.0}},
        "9": {"class_type": "LoraLoaderStackedAdvanced", "inputs": {
            "lora_name": {"content": "zavy.safetensors", "type": "loras"}, "lora_weight": 0.63}},
        "4": {"class_type": "CR LoRA Stack", "inputs": {
            "switch_1": "On", "lora_name_1": "c.safetensors", "model_weight_1": 0.9,
            "switch_2": "Off", "lora_name_2": "d.safetensors", "model_weight_2": 1.0}},
    }
    assert _canon(g)["loras"] == [
        {"name": "a.safetensors", "strength": 1.0},
        {"name": "zavy.safetensors", "strength": 0.63},
        {"name": "c.safetensors", "strength": 0.9},
    ]


def test_ideogram4_builder_still_reconstructed():
    g = {
        "S": {"class_type": "KSampler", "inputs": {"positive": ["E", 0], "negative": ["Z", 0]}},
        "Z": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["E", 0]}},
        "E": {"class_type": "CLIPTextEncode", "inputs": {"text": ["B", 0]}},
        "B": {"class_type": "Ideogram4PromptBuilderKJ", "inputs": {
            "high_level_description": "A portrait of quz0", "style": "none",
            "background": "urban bokeh", "elements_data": "[]"}},
    }
    m = _canon(g)
    assert '"high_level_description": "A portrait of quz0"' in m["prompt"]
    assert m["negative"] is None


# ---- explicit overrides -----------------------------------------------------
def test_explicit_prompt_inputs_override_graph_extraction():
    meta = {"prompt": "from graph", "negative": None}
    srm._apply_overrides(meta, prompt_text="LLM generated text", negative_text="  ")
    assert meta["prompt"] == "LLM generated text"
    assert meta["negative"] is None          # blank override is ignored
    srm._apply_overrides(meta, prompt_text=None, negative_text="blurry")
    assert meta["prompt"] == "LLM generated text" and meta["negative"] == "blurry"
