"""Structural identification heuristics. No inference is a provenance guarantee."""
from collections import Counter
import math
import re
import struct


def adapter_parts(name):
    match = re.match(r"^(.*?)[._](lora_down|lora_up|lora_a|lora_b)(?:\.[^.]+)?\.weight$", name, re.I)
    if not match:
        return None
    return match[1], "down" if match[2].lower() in ("lora_down", "lora_a") else "up"


def pairs_for(tensors):
    pairs = {}
    for tensor in tensors:
        part = adapter_parts(tensor["name"])
        if part:
            pairs.setdefault(part[0], {})[part[1]] = tensor
    return pairs


def identify(tensors, metadata):
    findings = []
    def add(category, label, confidence, evidence):
        findings.append(dict(category=category, label=label, confidence=confidence, evidence=evidence))
    def hits(*terms):
        return [t for t in tensors if any(term in t["name"].lower() for term in terms)]
    def examples(items):
        return "; ".join(f"{t['name']} {t['shape']}" for t in items[:3])
    pairs = pairs_for(tensors)
    complete = {name: p for name, p in pairs.items() if "up" in p and "down" in p}
    valid = {name: p for name, p in complete.items()
             if len(p["up"]["shape"]) >= 2 and len(p["down"]["shape"]) >= 2
             and p["up"]["shape"][1] == p["down"]["shape"][0] > 0}
    if pairs:
        ranks = Counter(p["down"]["shape"][0] for p in valid.values())
        add("File type", "LoRA adapter" if valid else "Possible LoRA / incomplete adapter", "High" if valid else "Low",
            f"{len(valid)} dimension-consistent up/down pairs; {len(pairs) - len(complete)} incomplete pairs; "
            f"{len(complete) - len(valid)} inconsistent pairs. Ranks (module counts): {dict(ranks)}. " +
            "; ".join(list(valid)[:3]))
    lyco = hits("hada_w1", "hada_w2", "lokr_w1", "lokr_w2", "oft_blocks", "boft")
    if lyco:
        add("File type", "LyCORIS / structured adapter", "Medium", examples(lyco))
    adapter = bool(pairs or lyco)
    # Exclude adapter tensors when deciding which full components are bundled.
    full = [t for t in tensors if not adapter_parts(t["name"])]
    full_names = [t["name"].lower() for t in full]
    def full_has(*terms):
        return any(any(term in name for term in terms) for name in full_names)
    encoder = full_has("encoder.conv_in.weight")
    decoder = full_has("decoder.conv_out.weight")
    vae = encoder and decoder
    unet = (full_has("input_blocks.", "down_blocks.") and full_has("output_blocks.", "up_blocks.")
            and full_has("time_embed.", "time_embedding."))
    flux_blocks = bool(hits("double_blocks.", "double_blocks_")) and bool(hits("single_blocks.", "single_blocks_"))
    flux_diffusers = bool(hits("single_transformer_blocks.", "single_transformer_blocks_")) and bool(hits("transformer_blocks.", "transformer_blocks_"))
    sd3 = bool(hits("joint_blocks.", "joint_blocks_")) and bool(hits("x_block", "context_block"))
    # Newer diffusion transformers, matched on layer groups seen in the released weights.
    qwen_image = bool(hits("transformer_blocks.", "transformer_blocks_")) and bool(hits("img_mod.", "img_mod_")) and not flux_diffusers
    # patch_embedding separates Wan from ViT-style models (e.g. SAM) that also have cross/self attention.
    wan = (bool(hits("cross_attn.", "cross_attn_")) and bool(hits("self_attn.", "self_attn_")) and bool(hits("blocks.", "blocks_"))
           and (adapter or bool(hits("patch_embedding"))))
    hunyuan = bool(hits("individual_token_refiner"))
    krea2 = bool(hits("attn.qknorm", "attn_qknorm")) and bool(hits("mod.lin", "mod_lin"))
    zimage = bool(hits("cap_embedder", "noise_refiner", "context_refiner"))
    ltx = bool(hits("adaln_single", "audio_adaln", "av_ca_")) and bool(hits("transformer_blocks.", "transformer_blocks_"))
    new_family = qwen_image or wan or krea2 or zimage or ltx or hunyuan
    transformer = not adapter and (flux_blocks or flux_diffusers or sd3 or new_family)
    text = full_has("text_model.embeddings.token_embedding", "cond_stage_model.transformer", "conditioner.embedders", "text_encoder.")
    t5 = not transformer and full_has("encoder.block.") and full_has("selfattention", "shared.weight")
    llm = not transformer and full_has("model.layers.") and full_has("embed_tokens")
    text = text or t5 or llm
    control = hits("controlnet_cond_embedding", "control_model.", "zero_convs.", "controlnet_down_blocks", "controlnet_blocks", "controlnet_single_blocks")
    if control:
        add("File type", "ControlNet / conditioning network", "Medium", examples(control))
    components = []
    if unet:
        components.append("diffusion UNet")
    if transformer:
        components.append("diffusion transformer")
    if vae:
        components.append("VAE")
    if text:
        components.append("text encoder")
    if components:
        add("Components", ", ".join(components), "High", "Independent layer groups found: " + ", ".join(components))
    if (unet or transformer) and not control:
        label = "Bundled diffusion checkpoint" if vae or text else "Standalone diffusion backbone / model weights"
        add("File type", label, "High", "Contains " + ", ".join(components) + ". Completeness of the model is not guaranteed.")
    elif vae:
        add("File type", "VAE / autoencoder weights", "High", "Both encoder.conv_in.weight and decoder.conv_out.weight are present.")
    elif text and not adapter:
        add("File type", "Text encoder weights", "Medium", "Text-encoder layers without a recognized diffusion backbone.")
    # Read cross-attention input width, including the down/A side of a LoRA.
    attention = []
    for t in tensors:
        n = t["name"].lower().replace("_", ".")
        part = adapter_parts(t["name"])
        if ("attn2.to.k" in n or "attn2.to.v" in n) and len(t["shape"]) == 2:
            if part is None or part[1] == "down":
                attention.append(t)
    families = {768: "Stable Diffusion 1.x", 1024: "Stable Diffusion 2.x",
                2048: "SDXL base", 1280: "SDXL refiner"}
    widths = sorted({t["shape"][1] for t in attention})
    for width in widths:
        group = [t for t in attention if t["shape"][1] == width]
        family = families.get(width, f"Unknown cross-attention family (width {width})")
        confidence = "Medium" if len(widths) > 1 or len(group) < 2 else "High"
        add("Base family", family, confidence, f"Cross-attention K/V input width {width} across {len(group)} tensors. " + examples(group))
    if len(widths) > 1:
        add("Caution", "Multiple conditioning widths", "High", f"Observed {widths}; this could be a mixed file or an unsupported architecture. Do not assume a single base.")
    if flux_blocks or flux_diffusers:
        projection = hits("img_in.weight", "x_embedder.weight", "txt_in.weight", "context_embedder.weight")
        signature = any(t["shape"] == [3072, 64] or t["shape"] == [3072, 4096] for t in projection)
        add("Base family", "FLUX.1" if signature else "FLUX-style (variant unresolved)", "High" if signature else "Medium",
            "Both dual-stream and single-stream block groups found. " + examples(projection) +
            " LoRA-only files may not distinguish dev, schnell, or other derivatives.")
    if sd3:
        add("Base family", "Stable Diffusion 3 / 3.5 family", "Medium",
            "Joint blocks contain image/context streams. Exact version needs more evidence. " + examples(hits("joint_blocks.", "joint_blocks_")))
    for found, label, terms in ((qwen_image, "Qwen-Image family", ("img_mod.", "img_mod_")),
                                (wan, "Wan 2.x video family", ("cross_attn.", "cross_attn_")),
                                (krea2, "Krea 2 family", ("attn.qknorm", "attn_qknorm")),
                                (zimage, "Z-Image / Lumina-style family", ("cap_embedder", "noise_refiner", "context_refiner")),
                                (ltx, "LTX video family", ("adaln_single", "audio_adaln", "av_ca_")),
                                (hunyuan, "HunyuanVideo family", ("individual_token_refiner",))):
        if found:
            add("Base family", label, "Medium", "Layer-group signature (tentative). " + examples(hits(*terms)))
    if t5:
        add("Base family", "T5-style text encoder", "Medium", "encoder.block self-attention stack. " + examples(hits("encoder.block.")))
    if llm:
        add("Base family", "LLM-style text encoder (Qwen / Llama / Gemma layout)", "Medium",
            "model.layers stack with token embeddings. " + examples(hits("embed_tokens", "model.layers.0.")))
    if vae:
        latent = [t for t in full if t["name"].endswith("decoder.conv_in.weight") and len(t["shape"]) == 4]
        for t in latent:
            channels = t["shape"][1]
            add("Compatibility", f"VAE with {channels} latent channels", "High", examples([t]) +
                (". Shared by SD 1.x, SD 2.x and SDXL-style VAEs; weights alone do not establish scaling or exact compatibility." if channels == 4 else
                 ". Sixteen-channel VAEs occur in FLUX and SD3 families; channel count alone cannot choose between them." if channels == 16 else
                 ". Consult the model configuration for latent scaling and architecture."))
    if not adapter:
        inputs = [t for t in tensors if any(t["name"].endswith(s) for s in ("input_blocks.0.0.weight", "unet.conv_in.weight")) or t["name"] == "conv_in.weight"]
        for t in inputs:
            if len(t["shape"]) == 4 and t["shape"][1] == 9:
                add("Variant", "Inpainting-style UNet", "Medium", "Nine input channels (latent + mask + masked latent). " + examples([t]))
    for key, value in metadata.items():
        if any(s in key.lower() for s in ("architecture", "base_model", "model_version", "sd_model", "network_module", "modelspec", "ss_v2")):
            add("Declared metadata", key, "Unverified", value)
    if not findings:
        add("File type", "Unknown tensor collection", "Low", "No supported structural signature. Inspect tensor names and dimensions manually.")
    add("Limits", "Exact base checkpoint / training history unresolved", "High",
        "A full checkpoint may be original, fine-tuned, or merged. Similar shapes or value statistics cannot prove which. "
        "Family inference is not a guarantee of compatibility with every model in that family.")
    return findings


def canonical(name):
    name = name.lower()
    for prefix in ("base_model.model.", "model.diffusion_model.", "diffusion_model.",
                   "lora_unet_", "lora_transformer_", "unet.", "transformer."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return re.sub(r"[^a-z0-9]", "", name.removesuffix(".weight"))


def compare_adapter(adapter, base):
    index = {}
    for t in base["tensors"]:
        index.setdefault(canonical(t["name"]), []).append(t)
    result = {"base_path": base["path"], "matched": [], "mismatched": [], "unresolved": []}
    pairs = pairs_for(adapter["tensors"])
    for name, pair in pairs.items():
        if "down" not in pair or "up" not in pair:
            result["unresolved"].append({"module": name, "reason": "Missing up/down pair"})
            continue
        down, up = pair["down"]["shape"], pair["up"]["shape"]
        candidates = index.get(canonical(name), [])
        if len(candidates) != 1 or len(down) < 2 or len(up) < 2:
            result["unresolved"].append({"module": name, "reason": "No unique corresponding weight or unsupported shape/naming convention"})
            continue
        target = candidates[0]
        # Ordinary linear and LoCon with a 1x1 up projection only.
        if len(up) != len(down) or (len(up) > 2 and any(n != 1 for n in up[2:])):
            result["unresolved"].append({"module": name, "reason": "Unsupported convolution factorization"})
            continue
        expected = [up[0], down[1]] + down[2:]
        item = {"module": name, "base_tensor": target["name"], "adapter_target_shape": expected, "base_shape": target["shape"]}
        result["matched" if expected == target["shape"] and up[1] == down[0] else "mismatched"].append(item)
    result["summary"] = (f"{len(result['matched'])} shape matches, {len(result['mismatched'])} mismatches, "
                         f"{len(result['unresolved'])} unresolved modules. Matching shapes are necessary, not sufficient for compatibility. "
                         "Cross-format name conversions (e.g. original UNet to Diffusers) are not automatically assumed.")
    if not pairs:
        result["summary"] = "No conventional LoRA up/down pairs to compare in the current file."
    return result


def sample_tensor(path, tensor, limit=4096):
    """Read up to 4096 values in three windows; bounded memory, no torch."""
    formats = {"F64": "d", "F32": "f", "F16": "e", "I64": "q", "U64": "Q", "I32": "i",
               "U32": "I", "I16": "h", "U16": "H", "I8": "b", "U8": "B", "BOOL": "?", "BF16": "H"}
    dtype = tensor["dtype"]
    if dtype not in formats:
        raise ValueError(f"Value decoding for {dtype} is not implemented. Descriptor inspection remains available.")
    count = tensor["elements"]
    if count == 0:
        return {"sampled": 0, "note": "Empty tensor"}
    fmt = "<" + formats[dtype]
    width = struct.calcsize(fmt)
    windows = [(0, count)] if count <= limit else [(0, limit // 3), (count // 2, limit // 3), (count - limit // 3, limit // 3)]
    values = []
    with open(path, "rb") as f:
        for start, length in windows:
            f.seek(tensor["file_offsets"][0] + start * width)
            raw = f.read(length * width)
            if len(raw) != length * width:
                raise ValueError("File changed or was truncated after inspection.")
            for (value,) in struct.iter_unpack(fmt, raw):
                if dtype == "BF16":
                    value = struct.unpack("<f", struct.pack("<I", value << 16))[0]
                values.append(value)
    finite = [v for v in values if math.isfinite(v)]
    # Scaled moments avoid overflow for legitimate F64 tensors.
    scale = max((abs(v) for v in finite), default=0) or 1
    mean_scaled = math.fsum(v / scale for v in finite) / len(finite) if finite else None
    std = scale * math.sqrt(math.fsum((v / scale - mean_scaled) ** 2 for v in finite) / len(finite)) if finite else None
    return {"sampled": len(values), "total_elements": count, "windows": windows,
            "min_finite": min(finite) if finite else None, "max_finite": max(finite) if finite else None,
            "mean_finite": mean_scaled * scale if finite else None, "std_finite": std,
            "zero_fraction": sum(v == 0 for v in values) / len(values),
            "nan_count": sum(math.isnan(v) for v in values), "inf_count": sum(math.isinf(v) for v in values),
            "first_values": [v if math.isfinite(v) else str(v) for v in values[:16]],
            "note": "Sample statistics only; unsampled values may differ. Value distributions do not prove base-model identity."}
