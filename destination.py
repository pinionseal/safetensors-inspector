"""Best-guess ComfyUI model folder for an inspected file, plus no-overwrite moves and folder sorting."""
import json
import os
import re
import shutil
from pathlib import Path

SETTINGS = Path(__file__).with_name("inspector_settings.json")
DEFAULTS = {"models_root": "", "large_models_root": "", "large_file_gb": 4}

# Checked in order: the first match wins, so specific names come before general ones.
# Each entry: (text to look for, loras subfolder, diffusion_models subfolder).
FAMILIES = [
    (("hunyuan",), None, None),
    (("qwen",), "qwen", "QWEN"),
    (("krea2", "krea 2", "krea_2"), "Krea2", None),
    (("flux.2", "flux2", "flux 2", "flux-2"), "flux2", "FLUX"),
    (("flux",), "flux1", "FLUX"),
    (("wan",), "wan", "WAN"),
    (("ltx",), "ltx", None),
    (("z-image", "z_image", "zimage", "lumina"), "zimage", None),
    (("sdxl", "stable-diffusion-xl", "xl base", "xl refiner"), "sdxl", None),
]

METADATA_KEYS = ("ss_base_model_version", "modelspec.architecture", "base_model", "ss_sd_model_name", "model_version")


def load_settings():
    settings = dict(DEFAULTS)
    try:
        settings.update(json.loads(SETTINGS.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings):
    SETTINGS.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def family(report):
    """Return (loras subfolder, diffusion_models subfolder, evidence) or (None, None, None)."""
    declared = " ".join(str(report["metadata"].get(k, "")) for k in METADATA_KEYS).lower()
    found = " ".join(f["label"] for f in report["identification"] if f["category"] == "Base family").lower()
    filename = Path(report.get("path", "")).name.lower()
    for source, text in (("declared metadata", declared), ("structure", found), ("filename only", filename)):
        for terms, lora_dir, model_dir in FAMILIES:
            if any(term in text for term in terms):
                return lora_dir, model_dir, f"{source}: {text.strip()[:80]}"
    return None, None, None


# Last resort when the structure is not recognised: (words in the filename, folder).
FILENAME_HINTS = [
    (("lora", "lokr", "locon", "lycoris"), "loras"),
    (("controlnet", "control_"), "controlnet"),
    (("vae", "taesd", "taef1", "taeltx", "taehv", "taew"), "vae"),
    (("clip", "t5xxl", "umt5", "text_encoder", "embeddings_connector", "gemma", "llama"), "text_encoders"),
    (("esrgan", "upscale", "4x", "2x_", "x4", "swinir"), "upscale_models"),
]


def looks_like_upscaler(report):
    names = [t["name"].lower() for t in report["tensors"]]
    return (any(n.startswith("conv_first") for n in names) or
            (any(".rrdb" in n or "body." in n for n in names) and any("conv_up" in n or "upconv" in n for n in names)))


def guess(report):
    """Best guess only. Returns {'folder': 'loras/qwen' or None, 'reason': str, 'confidence': str}."""
    labels = {f["label"] for f in report["identification"] if f["category"] == "File type"}
    lora_dir, model_dir, why = family(report)
    def result(folder, reason, confidence):
        return {"folder": folder, "reason": reason, "confidence": confidence}
    if any("ControlNet" in l for l in labels):
        return result("controlnet", "ControlNet layer groups", "High")
    if any(l.startswith(("LoRA", "Possible LoRA", "LyCORIS")) for l in labels):
        if lora_dir:
            return result(f"loras/{lora_dir}", f"LoRA/LyCORIS; family from {why}", "Medium")
        return result("loras", "LoRA/LyCORIS; base family not recognised, so the loras root", "Low")
    if "Bundled diffusion checkpoint" in labels:
        return result("checkpoints", "Diffusion model bundled with VAE and/or text encoder", "High")
    if any(l.startswith("Standalone diffusion") for l in labels):
        folder = f"diffusion_models/{model_dir}" if model_dir else "diffusion_models"
        return result(folder, "Diffusion model without VAE/text encoder" + (f"; family from {why}" if why else ""), "Medium")
    if any(l.startswith("VAE") for l in labels):
        return result("vae", "Encoder and decoder of an autoencoder", "High")
    if any(l.startswith("Text encoder") for l in labels):
        return result("text_encoders", "Text-encoder layers without a diffusion model", "Medium")
    if looks_like_upscaler(report):
        return result("upscale_models", "ESRGAN-style upscaler layers", "Low")
    learned = learned_guess(report)
    if learned:
        return learned
    name = Path(report.get("path", "")).name.lower()
    for words, folder in FILENAME_HINTS:
        if any(w in name for w in words):
            if folder == "loras" and lora_dir:
                folder = f"loras/{lora_dir}"
            return result(folder, f"Structure not recognised; guessed from the filename ({name})", "Low")
    return result(None, "No confident match. Choose the folder yourself.", "None")


def needs_manual_choice(report):
    """True when the destination is missing, weakly inferred, or conflicted."""
    if report["destination_guess"]["confidence"] in ("None", "Low"):
        return True
    findings = report["identification"]
    if any(f["category"] == "Caution" and f["label"] == "Multiple conditioning widths" for f in findings):
        return True
    diffusion_families = {f["label"] for f in findings
                          if f["category"] == "Base family" and
                          not f["label"].endswith("text encoder") and
                          not f["label"].startswith("LLM-style text encoder")}
    return len(diffusion_families) > 1


def target_root(report, settings):
    big = report["file_bytes"] >= float(settings.get("large_file_gb", 4)) * 1024 ** 3
    root = settings.get("large_models_root") if big else settings.get("models_root")
    if not root or not os.path.isdir(root):
        root = settings.get("models_root")
    return root or None


def configured(settings):
    return bool(settings.get("models_root")) and os.path.isdir(settings["models_root"])


def native_copy(source, target, progress=None):
    """Copy with the operating system's own copy call. On Windows this is CopyFileExW, which lets an
    SMB server copy between its own shares without sending the data through this PC."""
    if os.name != "nt":
        shutil.copyfile(source, target)
        return
    import ctypes
    from ctypes import wintypes
    ROUTINE = ctypes.WINFUNCTYPE(wintypes.DWORD, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
                                 ctypes.c_longlong, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
                                 wintypes.HANDLE, wintypes.LPVOID)
    def routine(total, done, *_):
        if progress:
            progress(done, total)
        return 0  # PROGRESS_CONTINUE
    callback = ROUTINE(routine)
    copy = ctypes.windll.kernel32.CopyFileExW
    copy.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ROUTINE, wintypes.LPVOID, ctypes.POINTER(wintypes.BOOL), wintypes.DWORD]
    if not copy(str(source), str(target), callback, None, None, 0x1):  # COPY_FILE_FAIL_IF_EXISTS
        raise ctypes.WinError()


def copy_file(source, folder, progress=None):
    """Copy into folder without overwriting. Writes a .partial file, checks size, then renames."""
    source, folder = Path(source), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / source.name
    total = source.stat().st_size
    if target.exists():
        if target.stat().st_size == total:
            return {"target": str(target), "status": "same-size file already there, nothing copied"}
        raise FileExistsError(f"A different file already exists at {target}. Nothing was overwritten.")
    partial = target.with_name(target.name + ".partial")
    try:
        native_copy(source, partial, progress)
        if partial.stat().st_size != total:
            raise OSError("Copied size does not match the source.")
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"target": str(target), "status": "copied"}


# ---- Learning from folders the user already sorted -------------------------------------
INDEX = Path(__file__).with_name("library_index.json")
STRIP = ("model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer.", "model.", "unet.", "lora_unet_", "lora_te_")


def fingerprint(tensor_names, limit=400):
    """Layer-pattern set: prefixes stripped, numbers generalised, first three name parts kept."""
    parts = set()
    for name in tensor_names:
        name = name.lower()
        for prefix in STRIP:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        name = re.sub(r"\d+", "#", name.replace("_", "."))
        parts.add(".".join(name.split(".")[:3]))
    return sorted(parts)[:limit]


def build_index(roots, read_names, progress=None):
    """read_names(path) -> tensor names. Only headers are read. Returns entry count."""
    entries, files = [], []
    for root in roots:
        if root and os.path.isdir(root):
            files += [(root, p) for p in Path(root).rglob("*.safetensors") if p.is_file()]
    for i, (root, path) in enumerate(files, 1):
        try:
            entries.append({"folder": path.parent.relative_to(root).as_posix(), "file": path.name,
                            "fingerprint": fingerprint(read_names(path))})
        except Exception:
            pass
        if progress:
            progress(i, len(files))
    INDEX.write_text(json.dumps({"roots": [r for r in roots if r], "entries": entries}), encoding="utf-8")
    return len(entries)


def learned_guess(report, minimum=0.5):
    """Nearest already-sorted file by fingerprint overlap (Jaccard). None if no index or no close match."""
    try:
        entries = json.loads(INDEX.read_text(encoding="utf-8"))["entries"]
    except (OSError, ValueError, KeyError):
        return None
    mine = set(fingerprint(t["name"] for t in report["tensors"]))
    name = Path(report.get("path", "")).name
    best = None
    for e in entries:
        if e["file"] == name:
            continue
        theirs = set(e["fingerprint"])
        score = len(mine & theirs) / max(len(mine | theirs), 1)
        if not best or score > best[0]:
            best = (score, e)
    if best and best[0] >= minimum:
        return {"folder": best[1]["folder"], "reason": f"Layer pattern matches {best[1]['file']} already in {best[1]['folder']} ({best[0]:.0%} overlap)",
                "confidence": "Medium" if best[0] >= 0.8 else "Low"}
    return None


# ---- Sorting a whole folder ------------------------------------------------------------
LOGS = Path(__file__).with_name("sort_logs")


def inside(path, roots):
    path = Path(path).resolve()
    for root in roots:
        if root and os.path.isdir(root):
            root = Path(root).resolve()
            if path == root or root in path.parents:
                return True
    return False


def plan_folder(folder, inspect, settings):
    """One entry per .safetensors file directly in folder: where it would go, or why it stays."""
    plan = []
    for path in sorted(Path(folder).iterdir()):
        if not (path.is_file() and path.suffix.lower() in (".safetensors", ".safetensor")):
            continue
        item = {"source": str(path), "size": path.stat().st_size, "target_dir": None, "confidence": "None", "reason": ""}
        try:
            report = inspect(path)
        except Exception as exc:
            item["reason"] = f"Cannot read: {exc}"
            plan.append(item)
            continue
        guess = report["destination_guess"]
        root = target_root(report, settings)
        item.update(confidence=guess["confidence"], reason=guess["reason"])
        conflicted = needs_manual_choice(report) and guess["confidence"] not in ("None", "Low")
        if not guess["folder"] or guess["confidence"] == "None":
            item["reason"] = "No guess. " + guess["reason"]
        elif conflicted:
            item["reason"] = "Conflicting family clues, so not moved. " + guess["reason"]
        elif not root or not os.path.isdir(root):
            item["reason"] = f"Models folder {root} not reachable"
        else:
            item["target_dir"] = str(Path(root) / guess["folder"])
        plan.append(item)
    return plan


def move_file(source, folder, progress=None):
    """Move without overwriting. Same drive: rename. Other drive: verified copy, then delete the original."""
    source, folder = Path(source), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / source.name
    if target.exists():
        raise FileExistsError(f"{target} already exists; left {source.name} where it is.")
    try:
        os.rename(source, target)  # instant when source and target are on the same drive/share
        return str(target)
    except OSError:
        pass
    copy_file(source, folder, progress)
    if target.stat().st_size != source.stat().st_size:
        raise OSError(f"Size check failed for {target}; original kept.")
    source.unlink()
    return str(target)


def sort_folder(plan, include_low=True, progress=None):
    """Carry out a plan. Writes a CSV log for undo. Returns (moved, skipped, failed, log_path)."""
    import csv, time
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / time.strftime("sort_%Y%m%d_%H%M%S.csv")
    moved = skipped = failed = 0
    todo = [p for p in plan if p["target_dir"] and (include_low or p["confidence"] != "Low")]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        log.writerow(["status", "source", "target", "confidence", "reason"])
        for item in plan:
            if item not in todo:
                skipped += 1
                why = item["reason"] if not item["target_dir"] else "Low confidence, left in place"
                log.writerow(["left in place", item["source"], "", item["confidence"], why])
        for i, item in enumerate(todo, 1):
            name = Path(item["source"]).name
            try:
                target = move_file(item["source"], item["target_dir"],
                                   (lambda d, t, n=name, i=i: progress(f"{i}/{len(todo)} {n}: {d * 100 // max(t, 1)}%")) if progress else None)
                moved += 1
                log.writerow(["moved", item["source"], target, item["confidence"], item["reason"]])
            except Exception as exc:
                failed += 1
                log.writerow(["failed", item["source"], "", item["confidence"], str(exc)])
            if progress:
                progress(f"{i}/{len(todo)} done: {name}")
    return moved, skipped, failed, str(log_path)


def undo_sort(log_path, progress=None):
    """Move files from a sort log back to where they came from. Returns (restored, failed)."""
    import csv
    restored = failed = 0
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "moved"]
    for row in rows:
        try:
            move_file(row["target"], Path(row["source"]).parent, None)
            restored += 1
        except Exception:
            failed += 1
        if progress:
            progress(f"Undo: {restored + failed}/{len(rows)}")
    return restored, failed


def latest_log():
    logs = sorted(LOGS.glob("sort_*.csv")) if LOGS.is_dir() else []
    return str(logs[-1]) if logs else None
