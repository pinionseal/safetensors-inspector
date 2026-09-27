import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from safetensor_inspector import inspect_file, Inspector
from tensor_detective import identify, compare_adapter, sample_tensor
import destination


def tensor(name, shape, dtype="F32"):
    return {"name": name, "shape": shape, "dtype": dtype}


def write_file(path, tensors, metadata=None):
    header = {} if metadata is None else {"__metadata__": metadata}
    payload = bytearray()
    for name, shape, values in tensors:
        start = len(payload)
        payload.extend(struct.pack("<" + "f" * len(values), *values))
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(payload)]}
    raw = json.dumps(header).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "test.safetensors"

    def tearDown(self):
        self.temp.cleanup()

    def test_scalar_empty_and_values(self):
        write_file(self.path, [("empty", [0, 3], []), ("scalar", [], [7]), ("weight", [4], [0, -2, 4, float("nan")])], {"name": "demo"})
        report = inspect_file(self.path)
        self.assertEqual(report["stored_elements"], 5)
        sample = sample_tensor(self.path, report["tensors"][-1])
        self.assertEqual(sample["nan_count"], 1)
        self.assertEqual(sample["min_finite"], -2)
        self.assertEqual(sample["zero_fraction"], .25)

    def test_truncated_header(self):
        self.path.write_bytes(struct.pack("<Q", 200) + b"{}")
        with self.assertRaisesRegex(ValueError, "beyond"):
            inspect_file(self.path)

    def test_duplicate_keys(self):
        raw = b'{"a":{},"a":{}}'
        self.path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            inspect_file(self.path)

    def test_bad_length(self):
        write_file(self.path, [("bad", [99], [1])])
        with self.assertRaisesRegex(ValueError, "byte length"):
            inspect_file(self.path)

    def test_trailing_payload(self):
        write_file(self.path, [("x", [1], [1])])
        with self.path.open("ab") as f:
            f.write(b"bad")
        with self.assertRaisesRegex(ValueError, "trailing"):
            inspect_file(self.path)

    def test_metadata_types(self):
        write_file(self.path, [], {"invalid": 1})
        with self.assertRaisesRegex(ValueError, "string-to-string"):
            inspect_file(self.path)

    def test_lora_sdxl_without_metadata(self):
        ts = [tensor("lora_unet_down_blocks_0_attentions_0_attn2_to_k.lora_down.weight", [4, 2048]),
              tensor("lora_unet_down_blocks_0_attentions_0_attn2_to_k.lora_up.weight", [320, 4])]
        findings = identify(ts, {})
        self.assertIn("LoRA adapter", [f["label"] for f in findings])
        self.assertIn("SDXL base", [f["label"] for f in findings])

    def test_attention_output_width_not_base_width(self):
        ts = [tensor("lora_unet_attn2_to_k.lora_up.weight", [2048, 4])]
        self.assertNotIn("SDXL base", [f["label"] for f in identify(ts, {})])

    def test_standalone_vae_ambiguous_family(self):
        ts = [tensor("encoder.conv_in.weight", [128, 3, 3, 3]),
              tensor("decoder.conv_out.weight", [3, 128, 3, 3]),
              tensor("decoder.conv_in.weight", [512, 4, 3, 3])]
        findings = identify(ts, {})
        self.assertIn("VAE / autoencoder weights", [f["label"] for f in findings])
        self.assertFalse(any(f["category"] == "Base family" for f in findings))

    def test_bundled_checkpoint(self):
        ts = [tensor(n, [2, 2]) for n in ("model.diffusion_model.input_blocks.0.0.weight",
              "model.diffusion_model.output_blocks.0.0.weight", "model.diffusion_model.time_embed.0.weight",
              "first_stage_model.encoder.conv_in.weight", "first_stage_model.decoder.conv_out.weight")]
        self.assertIn("Bundled diffusion checkpoint", [f["label"] for f in identify(ts, {})])

    def test_unknown(self):
        self.assertEqual(identify([tensor("foo", [4, 4])], {})[0]["label"], "Unknown tensor collection")

    def test_compare_and_mismatch(self):
        a = {"tensors": [tensor("lora_unet_down_blocks_0_attn2_to_k.lora_down.weight", [4, 2048]),
                         tensor("lora_unet_down_blocks_0_attn2_to_k.lora_up.weight", [320, 4])]}
        b = {"path": "base.safetensors", "tensors": [tensor("unet.down_blocks.0.attn2.to_k.weight", [320, 2048])]}
        self.assertEqual(len(compare_adapter(a, b)["matched"]), 1)
        b["tensors"][0]["shape"] = [320, 768]
        self.assertEqual(len(compare_adapter(a, b)["mismatched"]), 1)

    def test_bounded_sampling(self):
        write_file(self.path, [("w", [20000], list(range(20000)))])
        report = inspect_file(self.path)
        result = sample_tensor(self.path, report["tensors"][0])
        self.assertLessEqual(result["sampled"], 4096)
        self.assertEqual(result["max_finite"], 19999)

    def test_bf16_sample(self):
        self.path.write_bytes(struct.pack("<HH", 0x3f80, 0xc000))
        result = sample_tensor(self.path, {"dtype": "BF16", "elements": 2, "file_offsets": [0, 4]})
        self.assertEqual(result["first_values"], [1.0, -2.0])

    def guess_for(self, ts, metadata=None):
        metadata = metadata or {}
        return destination.guess({"metadata": metadata, "identification": identify(ts, metadata), "tensors": ts})["folder"]

    def test_guess_lora_family_from_metadata(self):
        ts = [tensor("diffusion_model.transformer_blocks.0.attn.add_k_proj.lora_A.weight", [16, 3072]),
              tensor("diffusion_model.transformer_blocks.0.attn.add_k_proj.lora_B.weight", [3072, 16])]
        self.assertEqual(self.guess_for(ts, {"ss_base_model_version": "qwen_image"}), "loras/qwen")
        self.assertEqual(self.guess_for(ts), "loras")

    def test_guess_components(self):
        vae = [tensor("encoder.conv_in.weight", [128, 3, 3, 3]), tensor("decoder.conv_out.weight", [3, 128, 3, 3])]
        self.assertEqual(self.guess_for(vae), "vae")
        t5 = [tensor("encoder.block.0.layer.0.SelfAttention.q.weight", [4, 4]), tensor("shared.weight", [4, 4])]
        self.assertEqual(self.guess_for(t5), "text_encoders")
        wan = [tensor("blocks.0.cross_attn.k.weight", [4, 4]), tensor("blocks.0.self_attn.q.weight", [4, 4]),
               tensor("patch_embedding.weight", [4, 4])]
        sam = [tensor("blocks.0.cross_attn.k.weight", [4, 4]), tensor("blocks.0.self_attn.q.weight", [4, 4])]
        self.assertNotEqual(self.guess_for(sam), "diffusion_models/WAN")
        self.assertEqual(self.guess_for(wan), "diffusion_models/WAN")
        self.assertIsNone(self.guess_for([tensor("foo", [4, 4])]))

    def test_learned_guess_from_sorted_library(self):
        index = destination.INDEX
        backup = index.read_bytes() if index.exists() else None
        try:
            root = Path(self.temp.name) / "models"
            (root / "unet").mkdir(parents=True)
            write_file(root / "unet" / "band.safetensors", [("band_split.0.weight", [1], [1]), ("mask_estimators.0.weight", [1], [1])])
            from safetensor_inspector import header_names
            destination.build_index([str(root)], header_names)
            write_file(self.path, [("band_split.0.weight", [1], [1]), ("mask_estimators.1.weight", [1], [1])])
            guess = inspect_file(self.path)["destination_guess"]
            self.assertEqual(guess["folder"], "unet")
            self.assertIn("band.safetensors", guess["reason"])
        finally:
            if backup is None:
                index.unlink(missing_ok=True)
            else:
                index.write_bytes(backup)

    def test_copy_never_overwrites(self):
        write_file(self.path, [("x", [1], [1])])
        folder = Path(self.temp.name) / "models" / "loras"
        self.assertEqual(destination.copy_file(self.path, folder)["status"], "copied")
        self.assertIn("already there", destination.copy_file(self.path, folder)["status"])
        (folder / self.path.name).write_bytes(b"different")
        with self.assertRaises(FileExistsError):
            destination.copy_file(self.path, folder)
        self.assertEqual((folder / self.path.name).read_bytes(), b"different")

    def test_manual_choice_rules(self):
        unknown = {"destination_guess": {"folder": None, "confidence": "None"}, "identification": []}
        self.assertTrue(destination.needs_manual_choice(unknown))
        unknown["destination_guess"] = {"folder": "loras", "confidence": "Low"}
        self.assertTrue(destination.needs_manual_choice(unknown))
        unknown["destination_guess"]["confidence"] = "High"
        self.assertFalse(destination.needs_manual_choice(unknown))
        unknown["identification"] = [{"category": "Caution", "label": "Multiple conditioning widths"}]
        self.assertTrue(destination.needs_manual_choice(unknown))
        unknown["identification"] = [{"category": "Base family", "label": "SDXL base"},
                                     {"category": "Base family", "label": "Stable Diffusion 1.x"}]
        self.assertTrue(destination.needs_manual_choice(unknown))
        unknown["identification"] = [{"category": "Base family", "label": "SDXL base"}]
        self.assertFalse(destination.needs_manual_choice(unknown))

    def test_unknown_prompts_and_moves_to_selected_folder(self):
        write_file(self.path, [("unrecognized.weight", [1], [1])])
        original = self.path.read_bytes()
        folder = Path(self.temp.name) / "chosen"
        folder.mkdir()
        app = Inspector()
        app.withdraw()
        try:
            with patch("safetensor_inspector.filedialog.askdirectory", return_value=str(folder)) as chooser, \
                    patch("safetensor_inspector.messagebox.showinfo"):
                app.display(inspect_file(self.path))
                for _ in range(50):
                    app.update()
                    if (folder / self.path.name).exists() and not app.copying:
                        break
                self.assertEqual(chooser.call_count, 1)
                self.assertEqual((folder / self.path.name).read_bytes(), original)
                self.assertFalse(self.path.exists())
                self.assertEqual(app.report["manual_destination"], str(folder))
        finally:
            app.destroy()

    def test_cancel_manual_choice_does_not_copy_or_reprompt(self):
        write_file(self.path, [("unknown", [1], [1])])
        app = Inspector()
        app.withdraw()
        try:
            with patch("safetensor_inspector.filedialog.askdirectory", return_value="") as chooser:
                app.display(inspect_file(self.path))
                app.update()
                app.display(inspect_file(self.path))
                app.update()
                self.assertEqual(chooser.call_count, 1)
                self.assertNotIn("manual_destination", app.report)
        finally:
            app.destroy()

    def test_sort_folder_moves_skips_logs_and_undoes(self):
        downloads = Path(self.temp.name) / "downloads"
        models = Path(self.temp.name) / "models"
        downloads.mkdir(); models.mkdir()
        write_file(downloads / "known_vae.safetensors", [("encoder.conv_in.weight", [1], [1]), ("decoder.conv_out.weight", [1], [1])])
        write_file(downloads / "mystery.safetensors", [("unrecognized", [1], [1])])
        (models / "vae").mkdir()
        write_file(models / "vae" / "taken.safetensors", [("x", [1], [1])])
        write_file(downloads / "taken.safetensors", [("encoder.conv_in.weight", [1], [1]), ("decoder.conv_out.weight", [1], [1])])
        settings = {"models_root": str(models), "large_models_root": str(models), "large_file_gb": 4}
        index = destination.INDEX
        backup = index.read_bytes() if index.exists() else None
        index.unlink(missing_ok=True)
        try:
            plan = destination.plan_folder(downloads, inspect_file, settings)
        finally:
            if backup is not None:
                index.write_bytes(backup)
        by_name = {Path(p["source"]).name: p for p in plan}
        self.assertIsNone(by_name["mystery.safetensors"]["target_dir"])
        moved, skipped, failed, log = destination.sort_folder(plan)
        self.assertEqual((moved, skipped, failed), (1, 1, 1))
        self.assertTrue((models / "vae" / "known_vae.safetensors").exists())
        self.assertFalse((downloads / "known_vae.safetensors").exists())
        self.assertTrue((downloads / "mystery.safetensors").exists())
        self.assertTrue((downloads / "taken.safetensors").exists())
        self.assertEqual(destination.undo_sort(log), (1, 0))
        self.assertTrue((downloads / "known_vae.safetensors").exists())
        Path(log).unlink()

    def test_recursive_sort_finds_subfolders_and_skips_library(self):
        downloads = Path(self.temp.name) / "downloads"
        models = downloads / "models"   # library nested inside the folder being sorted
        (downloads / "a" / "b").mkdir(parents=True); (models / "vae").mkdir(parents=True)
        vae = [("encoder.conv_in.weight", [1], [1]), ("decoder.conv_out.weight", [1], [1])]
        write_file(downloads / "top.safetensors", vae)
        write_file(downloads / "a" / "b" / "deep.safetensors", vae)
        write_file(models / "vae" / "already_sorted.safetensors", vae)
        (downloads / "a" / "notes.txt").write_text("x")
        roots = [str(models)]
        top = [p.name for p in destination.model_files(downloads, False, roots)]
        deep = [p.name for p in destination.model_files(downloads, True, roots)]
        self.assertEqual(top, ["top.safetensors"])
        self.assertEqual(sorted(deep), ["deep.safetensors", "top.safetensors"])
        settings = {"models_root": str(models), "large_models_root": str(models), "large_file_gb": 4}
        plan = destination.plan_folder(downloads, inspect_file, settings, recursive=True)
        self.assertEqual(sorted(Path(p["source"]).name for p in plan), ["deep.safetensors", "top.safetensors"])

    def test_refuses_to_sort_library(self):
        models = Path(self.temp.name) / "models"
        (models / "loras").mkdir(parents=True)
        self.assertTrue(destination.inside(models / "loras", [str(models)]))
        self.assertFalse(destination.inside(self.temp.name, [str(models)]))

    def test_gui_filter_sort_selection_and_paging(self):
        write_file(self.path, [(f"w{i}", [1], [i]) for i in range(510)])
        app = Inspector()
        app.withdraw()
        try:
            app.display(inspect_file(self.path))
            with patch("safetensor_inspector.filedialog.askdirectory", return_value=""):
                app.update()
            self.assertEqual(len(app.tree.get_children()), 500)
            app.change_page(1)
            self.assertEqual(len(app.tree.get_children()), 10)
            app.query.set("w509")
            app.filter_rows()
            self.assertEqual(len(app.tree.get_children()), 1)
            app.tree.selection_set("0")
            app.select_tensor()
            self.assertIn("w509", app.details.get("1.0", "end"))
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
