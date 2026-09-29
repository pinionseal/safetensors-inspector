"""Offline Safetensors Inspector. Python 3.10+ with Tk; no pip packages needed."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import queue
import struct
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tensor_detective import identify, compare_adapter, sample_tensor
import destination

MAX_HEADER = 100_000_000
BITS = {"BOOL": 8, "U8": 8, "I8": 8, "I16": 16, "U16": 16,
        "I32": 32, "U32": 32, "I64": 64, "U64": 64, "F16": 16,
        "BF16": 16, "F32": 32, "F64": 64, "F8_E4M3": 8, "F8_E5M2": 8,
        "F8_E8M0": 8, "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6}


def size_text(n):
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:,.2f} {unit}"
        value /= 1024


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        obj[key] = value
    return obj


def inspect_file(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("File is too short to contain a Safetensors header.")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= MAX_HEADER:
            raise ValueError("Invalid header length or header exceeds the 100 MB inspection limit.")
        if header_size > file_size - 8:
            raise ValueError("Header extends beyond the end of the file.")
        raw = f.read(header_size)
    if not raw.startswith(b"{"):
        raise ValueError("Safetensors JSON header must start with '{'.")
    header = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                        parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {s}")))
    if not isinstance(header, dict):
        raise ValueError("Header must be a JSON object.")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("Metadata must be a string-to-string object.")
    tensors, warnings = [], []
    payload = file_size - 8 - header_size
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Invalid tensor descriptor: {name!r}")
        dtype, shape, offsets = item.get("dtype"), item.get("shape"), item.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"Invalid dtype: {name!r}")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid shape: {name!r}")
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(n) is not int for n in offsets) or not 0 <= offsets[0] <= offsets[1] <= payload):
            raise ValueError(f"Invalid or out-of-file data offsets: {name!r}")
        count = math.prod(shape)
        length = offsets[1] - offsets[0]
        if dtype in BITS:
            bits = count * BITS[dtype]
            if bits % 8 or bits // 8 != length:
                raise ValueError(f"Shape/dtype byte length disagrees with offsets: {name!r}")
        else:
            warnings.append(f"{name}: unknown dtype {dtype}; byte size could not be verified.")
        tensors.append({"name": name, "dtype": dtype, "shape": shape, "rank": len(shape),
                        "elements": count, "bytes": length, "data_offsets": offsets,
                        "file_offsets": [8 + header_size + n for n in offsets]})
    cursor = 0
    for t in sorted(tensors, key=lambda t: (t["data_offsets"][0], t["data_offsets"][1])):
        start, end = t["data_offsets"]
        if start != cursor:
            raise ValueError(f"Payload contains an overlap or gap at {t['name']!r}.")
        cursor = end
    if cursor != payload:
        raise ValueError("File contains unindexed trailing data.")
    report = {"path": str(path), "file_bytes": file_size, "header_bytes": header_size,
              "payload_bytes": payload, "tensor_count": len(tensors),
              "stored_elements": sum(t["elements"] for t in tensors),
              "dtypes": dict(Counter(t["dtype"] for t in tensors)), "metadata": metadata,
              "identification": identify(tensors, metadata), "warnings": warnings, "tensors": tensors}
    report["destination_guess"] = destination.guess(report)
    return report


def header_names(path):
    """Tensor names only, for building the library index quickly."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if not 2 <= size <= MAX_HEADER:
            raise ValueError("bad header")
        return [k for k in json.loads(f.read(size)) if k != "__metadata__"]


def folders_text(settings):
    small, big = settings.get("models_root"), settings.get("large_models_root")
    if not small:
        return "Models folder: not set (Model folders...)"
    if not big or big == small:
        return f"Models folder: {small}  (all files)"
    return f"Models folder: {small}    Files of {settings.get('large_file_gb', 4)} GB and over: {big}"


def library_roots():
    settings = destination.load_settings()
    roots = [settings.get("models_root"), settings.get("large_models_root")]
    return list(dict.fromkeys(r for r in roots if r))


class Inspector(tk.Tk):
    PAGE_SIZE = 500

    def __init__(self, initial=None):
        super().__init__()
        self.title("Safetensors Inspector")
        self.geometry("1180x780")
        self.minsize(850, 560)
        self.report = None
        self.loading = False
        self.events = queue.Queue()
        self.generation = 0
        self.filtered = []
        self.page = 0
        self.sort_key = "name"
        self.reverse = False
        self.filter_job = None
        self.copying = False
        self.manual_prompted_generation = -1
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Treeview", rowheight=26)
        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")
        ttk.Button(top, text="Open file...", command=self.choose).pack(side="left")
        self.export_button = ttk.Button(top, text="Export JSON report...", command=self.export, state="disabled")
        self.export_button.pack(side="left", padx=8)
        self.compare_button = ttk.Button(top, text="Compare LoRA to base...", command=self.compare, state="disabled")
        self.compare_button.pack(side="left", padx=(0, 8))
        self.copy_button = ttk.Button(top, text="Move to best-guess folder...", command=self.copy_to_guess, state="disabled")
        self.copy_button.pack(side="left", padx=(0, 8))
        library = ttk.Frame(self, padding=(12, 0, 12, 8))
        library.pack(fill="x")
        ttk.Label(library, text="Library:").pack(side="left", padx=(0, 8))
        ttk.Button(library, text="Sort a folder...", command=self.sort_folder).pack(side="left", padx=(0, 4))
        self.recursive = tk.BooleanVar(value=False)
        ttk.Checkbutton(library, text="Include subfolders", variable=self.recursive).pack(side="left", padx=(0, 8))
        ttk.Button(library, text="Undo last sort", command=self.undo_sort).pack(side="left", padx=(0, 8))
        ttk.Button(library, text="Learn from my folders", command=self.learn).pack(side="left", padx=(0, 8))
        ttk.Button(library, text="Model folders...", command=self.set_folders).pack(side="left", padx=(0, 8))
        self.path_label = ttk.Label(top, text="Choose a .safetensors or .safetensor file", anchor="w")
        self.path_label.pack(side="left", fill="x", expand=True)
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=12)
        self.overview = self.text_tab("Overview")
        self.metadata = self.text_tab("Metadata")
        pane = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(pane, text="Tensors")
        search = ttk.Frame(pane)
        search.pack(fill="x", pady=(0, 8))
        ttk.Label(search, text="Filter name / shape / dtype:").pack(side="left")
        self.query = tk.StringVar()
        ttk.Entry(search, textvariable=self.query).pack(side="left", fill="x", expand=True, padx=8)
        self.query.trace_add("write", self.schedule_filter)
        self.dtype = tk.StringVar(value="All dtypes")
        self.dtype_box = ttk.Combobox(search, textvariable=self.dtype, state="readonly", width=15, values=["All dtypes"])
        self.dtype_box.pack(side="left")
        self.dtype_box.bind("<<ComboboxSelected>>", lambda e: self.filter_rows())
        table = ttk.Frame(pane)
        table.pack(fill="both", expand=True)
        columns = ("name", "dtype", "shape", "elements", "bytes")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse")
        for col, label, width in zip(columns, ("Tensor name", "Dtype", "Shape", "Elements", "Stored bytes"), (530, 90, 230, 125, 125)):
            self.tree.heading(col, text=label, command=lambda c=col: self.sort(c))
            self.tree.column(col, width=width, minwidth=70, stretch=(col == "name"))
        vs = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        hs = ttk.Scrollbar(table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self.select_tensor)
        nav = ttk.Frame(pane)
        nav.pack(fill="x", pady=8)
        self.previous = ttk.Button(nav, text="Previous", command=lambda: self.change_page(-1))
        self.previous.pack(side="left")
        self.next = ttk.Button(nav, text="Next", command=lambda: self.change_page(1))
        self.next.pack(side="left", padx=8)
        self.page_label = ttk.Label(nav)
        self.page_label.pack(side="left")
        ttk.Button(nav, text="Copy selected details", command=self.copy_details).pack(side="right")
        ttk.Button(nav, text="Sample weight values", command=self.sample).pack(side="right", padx=8)
        self.details = self.make_text(pane, height=7)
        self.raw = self.text_tab("Header JSON")
        self.analysis_text = self.text_tab("Weight samples / Base comparison")
        self.status = tk.StringVar(value=folders_text(destination.load_settings()) + "   |   Ctrl+O to open a file")
        ttk.Label(self, textvariable=self.status, padding=10).pack(fill="x")
        self.set_text(self.overview, "Open a file to inspect its metadata and tensor structure.\n\n"
                      "The app suggests model components from names and declared metadata. "
                      "These are clues, not definitive identification.\n\n"
                      "Use Sample weight values for bounded reads of actual tensor data. "
                      "Use Compare LoRA to base to check target shapes against a candidate model.\n\n"
                      "Files are never uploaded or modified.")
        self.bind("<Control-o>", lambda e: self.choose())
        self.poll_job = self.after(100, self.poll)
        if initial:
            self.after(150, lambda: self.load(initial))

    def make_text(self, parent, height=20):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="word", font=("Consolas", 10), padx=12, pady=12, height=height)
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set, state="disabled")
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return text

    def text_tab(self, label):
        frame = ttk.Frame(self.tabs)
        self.tabs.add(frame, text=label)
        return self.make_text(frame)

    @staticmethod
    def set_text(widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def choose(self):
        path = filedialog.askopenfilename(filetypes=[("Safetensors", "*.safetensors *.safetensor"), ("All files", "*.*")])
        if path:
            self.load(path)

    def load(self, path):
        if self.copying:
            messagebox.showinfo("Move in progress", "Wait for the current move to finish before opening another file.")
            return
        self.loading = True
        self.generation += 1
        generation = self.generation
        self.status.set(f"Reading header: {path}")
        self.export_button.configure(state="disabled")
        self.compare_button.configure(state="disabled")
        self.copy_button.configure(state="disabled")
        def work():
            try:
                result = inspect_file(path)
                self.events.put((generation, result, None))
            except Exception as exc:
                self.events.put((generation, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                generation, result, error = self.events.get_nowait()
                if generation != self.generation:
                    continue
                if error:
                    self.loading = False
                    self.status.set("Inspection or analysis failed. Previous results remain displayed." if self.report else "Open failed.")
                    messagebox.showerror("Cannot complete inspection or analysis", error)
                    self.copying = False
                    if self.report:
                        self.export_button.configure(state="normal")
                        self.compare_button.configure(state="normal")
                        self.copy_button.configure(state="normal")
                elif "progress" in result:
                    self.status.set(result["progress"])
                elif "learned" in result:
                    self.status.set(f"Learned layer patterns from {result['learned']:,} files. Unrecognised files now match against them.")
                    if self.report:
                        self.report["destination_guess"] = destination.guess(self.report)
                        self.display(self.report)
                elif "copy_result" in result:
                    self.copying = False
                    self.copy_button.configure(state="normal")
                    self.status.set(f"{result['copy_result']['status']}: {result['copy_result']['target']}")
                    messagebox.showinfo("Move finished", f"{result['copy_result']['status']}\n\n{result['copy_result']['target']}")
                    self.path_label.configure(text=result['copy_result']['target'])
                    if self.report:
                        self.report["path"] = result['copy_result']['target']
                elif "sort_plan" in result:
                    self.confirm_sort(result["sort_plan"], result["folder"])
                elif "sort_result" in result:
                    self.copying = False
                    moved, skipped, failed, log = result["sort_result"]
                    self.status.set(f"Sort done: {moved} moved, {skipped} left in place, {failed} failed. Log: {log}")
                    messagebox.showinfo("Folder sorted", f"Moved: {moved}\nLeft in place: {skipped}\nFailed: {failed}\n\nLog (used by Undo last sort):\n{log}")
                elif "undo_result" in result:
                    self.copying = False
                    restored, failed = result["undo_result"]
                    self.status.set(f"Undo done: {restored} moved back, {failed} failed.")
                    messagebox.showinfo("Undo finished", f"Moved back: {restored}\nFailed: {failed}")
                elif "analysis_result" in result:
                    self.report.setdefault("extra_analysis", []).append(result["analysis_result"])
                    self.set_text(self.analysis_text, json.dumps(self.report["extra_analysis"], indent=2, ensure_ascii=True))
                    self.tabs.select(4)
                    self.status.set("Analysis complete. Results are included in JSON export.")
                else:
                    self.display(result)
        except queue.Empty:
            pass
        self.poll_job = self.after(100, self.poll)

    def destroy(self):
        for job in (self.poll_job, self.filter_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
        super().destroy()

    def display(self, report):
        self.loading = False
        self.report = report
        self.path_label.configure(text=report["path"])
        summary = [Path(report["path"]).name, "=" * 60,
                   f"File: {size_text(report['file_bytes'])}    Header: {size_text(report['header_bytes'])}",
                   f"Tensors: {report['tensor_count']:,}    Stored elements: {report['stored_elements']:,}",
                   "Dtypes (tensor counts): " + ", ".join(f"{k}: {v:,}" for k, v in report["dtypes"].items()),
                   "", "IDENTIFICATION (structural evidence + declared metadata)", ""]
        for clue in report["identification"]:
            summary.extend([f"{clue['category']}: {clue['label']}  [{clue['confidence']}]", "  " + clue["evidence"], ""])
        g = report["destination_guess"]
        summary.extend(["SUGGESTED COMFYUI FOLDER (best guess)",
                        f"  {g['folder'] or 'none - choose it yourself'}  [{g['confidence']}]", "  " + g["reason"], ""])
        if destination.needs_manual_choice(report):
            summary.extend(["Destination uncertain: choose a folder when prompted. Cancel leaves the file in place.", ""])
        summary.extend(["Names can be changed, metadata can be missing or incorrect, and this file may be one shard.",
                        "Shape and dtype describe storage; they cannot prove model identity, training data, or provenance.",
                        "Stored elements include all tensors, including buffers and quantized values; this is not necessarily a model parameter count.", "",
                        "HEADER CHECKS", "Offsets, payload coverage, shapes, and known dtype sizes checked. Tensor values were not read."])
        summary.extend(report["warnings"])
        self.set_text(self.overview, "\n".join(summary))
        pretty_meta = {}
        for key, value in report["metadata"].items():
            try:
                pretty_meta[key] = json.loads(value)
            except (ValueError, RecursionError):
                pretty_meta[key] = value
        self.set_text(self.metadata, "Metadata values containing JSON are expanded for readability. Export preserves original strings.\n\n" +
                      (json.dumps(pretty_meta, indent=2, ensure_ascii=True) if pretty_meta else "No metadata stored in this file."))
        header = {"__metadata__": report["metadata"]}
        header.update({t["name"]: {k: t[k] for k in ("dtype", "shape", "data_offsets")} for t in report["tensors"]})
        raw = json.dumps(header, indent=2, ensure_ascii=True)
        if len(raw) > 2_000_000:
            raw = raw[:2_000_000] + "\n[Display truncated. Export JSON for all tensor descriptors.]"
        self.set_text(self.raw, raw)
        self.set_text(self.details, "Select a tensor to see its full name and offsets.")
        self.set_text(self.analysis_text, "Select a tensor and click Sample weight values, or compare the open LoRA to a candidate base file.")
        self.dtype_box.configure(values=["All dtypes"] + sorted(report["dtypes"]))
        self.dtype.set("All dtypes")
        self.query.set("")
        self.filter_rows()
        self.export_button.configure(state="normal")
        self.compare_button.configure(state="normal")
        self.copy_button.configure(state="normal", text="Choose destination folder..." if destination.needs_manual_choice(report) else "Move to best-guess folder...")
        self.tabs.select(0)
        self.status.set(f"Inspected {report['tensor_count']:,} tensors | {len(report['warnings'])} warnings | No tensor payload loaded")
        if destination.needs_manual_choice(report) and self.manual_prompted_generation != self.generation:
            self.manual_prompted_generation = self.generation
            self.after_idle(lambda generation=self.generation: self.offer_manual_destination(generation))

    def schedule_filter(self, *_):
        if self.filter_job:
            self.after_cancel(self.filter_job)
        self.filter_job = self.after(200, self.filter_rows)

    def filter_rows(self):
        if self.filter_job:
            self.after_cancel(self.filter_job)
            self.filter_job = None
        if not self.report:
            return
        query = self.query.get().lower().strip()
        dtype = self.dtype.get()
        self.filtered = [t for t in self.report["tensors"] if
                         (dtype == "All dtypes" or t["dtype"] == dtype) and
                         (not query or query in f"{t['name']} {t['dtype']} {t['shape']}".lower())]
        self.filtered.sort(key=lambda t: t[self.sort_key], reverse=self.reverse)
        self.page = 0
        self.render_rows()

    def sort(self, key):
        self.reverse = not self.reverse if key == self.sort_key else False
        self.sort_key = key
        self.filter_rows()

    def change_page(self, delta):
        self.page = max(0, min(self.page + delta, max(0, (len(self.filtered) - 1) // self.PAGE_SIZE)))
        self.render_rows()

    def render_rows(self):
        self.tree.delete(*self.tree.get_children())
        start = self.page * self.PAGE_SIZE
        for i, t in enumerate(self.filtered[start:start + self.PAGE_SIZE], start):
            self.tree.insert("", "end", iid=str(i), values=(t["name"], t["dtype"], str(t["shape"]), f"{t['elements']:,}", f"{t['bytes']:,}"))
        end = min(start + self.PAGE_SIZE, len(self.filtered))
        self.page_label.configure(text=f"{start + 1 if self.filtered else 0:,}–{end:,} of {len(self.filtered):,} matches")
        self.previous.configure(state="normal" if self.page else "disabled")
        self.next.configure(state="normal" if end < len(self.filtered) else "disabled")
        self.set_text(self.details, "Select a tensor to see details. Shape [] is a scalar; a zero dimension indicates an empty tensor.")

    def select_tensor(self, _=None):
        selection = self.tree.selection()
        if selection:
            tensor = self.filtered[int(selection[0])]
            self.set_text(self.details, json.dumps(tensor, indent=2, ensure_ascii=True))

    def copy_details(self):
        if self.tree.selection():
            self.clipboard_clear()
            self.clipboard_append(self.details.get("1.0", "end-1c"))

    def analyze_async(self, operation):
        generation = self.generation
        self.status.set("Analyzing...")
        def work():
            try:
                self.events.put((generation, {"analysis_result": operation()}, None))
            except Exception as exc:
                self.events.put((generation, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def sample(self):
        if self.loading:
            return
        if self.report and self.tree.selection():
            tensor = self.filtered[int(self.tree.selection()[0])]
            path = self.report["path"]
            self.analyze_async(lambda: {"tensor": tensor["name"], "values": sample_tensor(path, tensor)})
        else:
            messagebox.showinfo("Select a tensor", "Open the Tensors tab and select a row first.")

    def compare(self):
        if not self.report:
            return
        path = filedialog.askopenfilename(title="Choose the candidate base model or backbone", filetypes=[("Safetensors", "*.safetensors *.safetensor"), ("All files", "*.*")])
        if path:
            current = self.report
            self.analyze_async(lambda: compare_adapter(current, inspect_file(path)))

    def set_folders(self):
        settings = destination.load_settings()
        small = filedialog.askdirectory(title="Your ComfyUI 'models' folder (the one containing loras, vae, checkpoints...)",
                                        initialdir=settings["models_root"] if os.path.isdir(settings["models_root"]) else None)
        if not small:
            return False
        big = small
        if messagebox.askyesno("Large files", f"Put files of {settings['large_file_gb']} GB and over in a DIFFERENT folder?\n\n"
                               f"No = everything goes to\n{small}", default="no"):
            big = filedialog.askdirectory(title=f"Folder for files of {settings['large_file_gb']} GB and over",
                                          initialdir=small) or small
        settings.update(models_root=small, large_models_root=big)
        destination.save_settings(settings)
        self.status.set(folders_text(settings))
        return True

    def ensure_folders(self):
        """First run: ask for the models folder before anything is moved or learned."""
        if destination.configured(destination.load_settings()):
            return True
        messagebox.showinfo("Set your models folder", "First, choose your ComfyUI models folder (the folder that contains loras, vae, checkpoints...).\n\n"
                            "You will then be asked for an optional second folder for large files (Cancel = same folder).")
        return bool(self.set_folders())

    def sort_folder(self):
        if self.copying or self.loading or not self.ensure_folders():
            return
        recursive = self.recursive.get()
        folder = filedialog.askdirectory(title="Folder of downloaded models to sort" + (" (and its subfolders)" if recursive else " (top level only)"))
        if not folder:
            return
        if destination.inside(folder, library_roots()):
            messagebox.showerror("Not that folder", "That folder is inside your models folders. Point this at a downloads folder, not the library itself.")
            return
        settings, generation = destination.load_settings(), self.generation
        self.copying = True
        self.status.set(f"Inspecting files in {folder}...")
        def work():
            try:
                self.events.put((generation, {"sort_plan": destination.plan_folder(folder, inspect_file, settings, recursive), "folder": folder}, None))
            except Exception as exc:
                self.events.put((generation, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def confirm_sort(self, plan, folder):
        ready = [p for p in plan if p["target_dir"]]
        low = [p for p in ready if p["confidence"] == "Low"]
        stay = [p for p in plan if not p["target_dir"]]
        if not ready:
            self.copying = False
            messagebox.showinfo("Nothing to move", f"{len(plan)} model files found in {folder}; none had a usable guess." +
                                "".join(f"\n| {Path(p['source']).name}: {p['reason'][:90]}" for p in stay[:10]))
            return
        lines = [f"{Path(p['source']).name}  →  {p['target_dir']}  [{p['confidence']}]" for p in ready[:15]]
        more = f"\n... and {len(ready) - 15} more" if len(ready) > 15 else ""
        answer = messagebox.askyesnocancel("Sort folder",
            f"{len(ready)} files will be moved, {len(stay)} stay where they are (no guess or conflicting clues).\n\n" +
            "\n".join(lines) + more +
            f"\n\nYes = move all {len(ready)}    No = move only Medium/High ({len(ready) - len(low)}), leave the {len(low)} Low-confidence ones    Cancel = do nothing")
        if answer is None:
            self.copying = False
            self.status.set("Sort cancelled. Nothing moved.")
            return
        generation = self.generation
        def progress(text):
            self.events.put((generation, {"progress": "Sorting: " + text}, None))
        def work():
            try:
                self.events.put((generation, {"sort_result": destination.sort_folder(plan, include_low=answer, progress=progress)}, None))
            except Exception as exc:
                self.events.put((generation, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def undo_sort(self):
        if self.copying:
            return
        log = destination.latest_log()
        if not log:
            messagebox.showinfo("Nothing to undo", "No sort log found.")
            return
        if not messagebox.askyesno("Undo last sort", f"Move every file listed as moved in\n\n{log}\n\nback to where it came from?"):
            return
        generation = self.generation
        self.copying = True
        def work():
            try:
                self.events.put((generation, {"undo_result": destination.undo_sort(log, lambda t: self.events.put((generation, {"progress": t}, None)))}, None))
            except Exception as exc:
                self.events.put((generation, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def learn(self):
        if not self.ensure_folders():
            return
        roots = [r for r in library_roots() if os.path.isdir(r)]
        if not roots:
            messagebox.showinfo("Models folder not found", "Set a reachable models folder with Model folders... first.")
            return
        generation = self.generation
        def progress(done, total):
            if done % 25 == 0 or done == total:
                self.events.put((generation, {"progress": f"Learning from your folders: {done:,} of {total:,} files"}, None))
        def work():
            try:
                self.events.put((generation, {"learned": destination.build_index(roots, header_names, progress)}, None))
            except Exception as exc:
                self.events.put((generation, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def copy_to_guess(self):
        if not self.report or self.copying or self.loading:
            return
        if not destination.needs_manual_choice(self.report) and not self.ensure_folders():
            return
        if destination.needs_manual_choice(self.report):
            self.offer_manual_destination(self.generation)
            return
        settings = destination.load_settings()
        root = destination.target_root(self.report, settings)
        guess = self.report["destination_guess"]
        name = Path(self.report["path"]).name
        folder = None
        if guess["folder"] and root and os.path.isdir(root):
            folder = str(Path(root) / guess["folder"])
            answer = messagebox.askyesnocancel("Move to best-guess folder",
                f"Move {name} ({size_text(self.report['file_bytes'])}) to\n\n{folder}\n\n"
                f"Why: {guess['reason']}  [{guess['confidence']}]\n\n"
                "Yes = move there    No = pick a different folder    Cancel = do nothing")
            if answer is None:
                return
            if answer is False:
                folder = None
        elif not (root and os.path.isdir(root)):
            messagebox.showinfo("Models folder not found", f"{root or 'No models folder'} is not reachable. Pick the destination folder yourself, or set it with Model folders...")
        if folder is None:
            folder = filedialog.askdirectory(title=f"Move {name} into...", initialdir=root if root and os.path.isdir(root) else None)
            if not folder:
                return
        self.start_copy(folder)

    def offer_manual_destination(self, generation):
        if (generation != self.generation or self.loading or self.copying or not self.report or
                not destination.needs_manual_choice(self.report)):
            return
        root = destination.target_root(self.report, destination.load_settings())
        name = Path(self.report["path"]).name
        folder = filedialog.askdirectory(
            parent=self,
            title=f"Destination uncertain: select a folder for {name} (Cancel to skip)",
            initialdir=root if root and os.path.isdir(root) else None)
        if not folder or generation != self.generation:
            self.status.set(f"No destination selected for {name}. Use Choose destination folder... when ready.")
            return
        self.report["manual_destination"] = folder
        self.start_copy(folder)

    def start_copy(self, folder):
        if not self.report or self.copying:
            return
        source, generation = self.report["path"], self.generation
        name = Path(source).name
        self.copying = True
        self.copy_button.configure(state="disabled")
        def progress(done, total):
            self.events.put((generation, {"progress": f"Moving {name}: {size_text(done)} of {size_text(total)} ({done * 100 // max(total, 1)}%)"}, None))
        def work():
            try:
                self.events.put((generation, {"copy_result": {"status": "moved", "target": destination.move_file(source, folder, progress)}}, None))
            except Exception as exc:
                self.events.put((generation, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def export(self):
        if not self.report:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile=Path(self.report["path"]).name + ".report.json",
                                          filetypes=[("JSON report", "*.json")])
        if path:
            if (Path(path).resolve() == Path(self.report["path"]) or
                    (Path(path).exists() and os.path.samefile(path, self.report["path"]))):
                messagebox.showerror("Cannot export", "Choose a different path from the inspected file.")
                return
            try:
                Path(path).write_text(json.dumps(self.report, indent=2, ensure_ascii=True), encoding="utf-8")
                self.status.set(f"Exported complete report: {path}")
            except OSError as exc:
                messagebox.showerror("Export failed", str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help="Optional file to open")
    parser.add_argument("--json", action="store_true", help="Print report without opening the GUI (requires file)")
    parser.add_argument("--suggest", action="store_true", help="Print the best-guess ComfyUI folder (requires file)")
    parser.add_argument("--sort", metavar="FOLDER", help="Move every model file in FOLDER (top level) to its best-guess folder")
    parser.add_argument("--recursive", action="store_true", help="With --sort: also sort model files in every subfolder of FOLDER")
    parser.add_argument("--skip-low", action="store_true", help="With --sort: leave Low-confidence files in place")
    parser.add_argument("--dry-run", action="store_true", help="With --sort: show the plan without moving anything")
    parser.add_argument("--models-root", help="ComfyUI models folder for this run (overrides the saved setting)")
    parser.add_argument("--large-models-root", help="Folder for large files for this run (default: --models-root)")
    parser.add_argument("--learn", nargs="*", metavar="ROOT", help="Index layer patterns of files already in your model folders (default: saved folders)")
    parser.add_argument("--move", "--copy", dest="copy", metavar="MODELS_ROOT", nargs="?", const="", help="Move into MODELS_ROOT/<best guess> (default: saved models folder)")
    args = parser.parse_args()
    settings = destination.load_settings()
    if args.models_root:
        settings.update(models_root=args.models_root, large_models_root=args.large_models_root or args.models_root)
    elif args.large_models_root:
        settings["large_models_root"] = args.large_models_root
    if (args.sort or args.copy is not None) and not (args.copy or destination.configured(settings)):
        parser.exit(1, "No models folder set. Pass --models-root, or set it once in the app with Model folders....\n")
    if args.sort:
        if destination.inside(args.sort, [settings.get("models_root"), settings.get("large_models_root")]):
            parser.exit(1, "That folder is inside your models folders; refusing to sort the library itself.\n")
        plan = destination.plan_folder(args.sort, inspect_file, settings, recursive=args.recursive)
        for p in plan:
            print(f"{'MOVE' if p['target_dir'] else 'STAY'} [{p['confidence']}] {Path(p['source']).name} -> {p['target_dir'] or p['reason']}")
        if not args.dry_run:
            moved, skipped, failed, log = destination.sort_folder(plan, include_low=not args.skip_low)
            print(f"{moved} moved, {skipped} left in place, {failed} failed. Log: {log}")
        return
    if args.learn is not None:
        roots = args.learn or [r for r in (settings.get("models_root"), settings.get("large_models_root")) if r]
        if not roots:
            parser.exit(1, "No folders to learn from. Pass them (--learn PATH ...) or --models-root.\n")
        count = destination.build_index(roots, header_names)
        print(f"Indexed {count} files from {', '.join(roots)} -> {destination.INDEX}")
        if not args.file:
            return
    if (args.suggest or args.copy is not None) and not args.file:
        parser.error("--suggest and --copy require a file")
    if args.suggest or args.copy is not None:
        try:
            report = inspect_file(args.file)
        except Exception as exc:
            parser.exit(1, f"Cannot inspect file: {exc}\n")
        g = report["destination_guess"]
        print(f"{g['folder'] or 'no guess'}  [{g['confidence']}]  {g['reason']}")
        if args.copy is not None:
            root = args.copy or destination.target_root(report, settings)
            if not g["folder"] or not root or not os.path.isdir(root):
                parser.exit(1, "No confident folder guess or models folder not found; nothing copied.\n")
            print("moved to " + destination.move_file(args.file, Path(root) / g["folder"]))
    elif args.json:
        if not args.file:
            parser.error("--json requires a file")
        try:
            print(json.dumps(inspect_file(args.file), indent=2))
        except Exception as exc:
            parser.exit(1, f"Cannot inspect file: {exc}\n")
    else:
        Inspector(args.file).mainloop()


if __name__ == "__main__":
    main()
