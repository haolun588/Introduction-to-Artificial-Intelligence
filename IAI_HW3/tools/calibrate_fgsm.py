"""Train-only FGSM scale and real BERT gradient audit; no optimizer updates."""
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model_source import file_sha256, verify_base_model
from src.training import ROOT, class_weights, fgsm_backward, load_split, loader, smoke_subset, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Choose a new calibration output directory")
    source = verify_base_model(ROOT / "outputs/base_model")
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    train_rows, _, manifest = load_split(ROOT / "dataset", ROOT / "outputs/eda/split.json")
    scale_rows = smoke_subset(train_rows, 42)
    audit_rows = smoke_subset(train_rows, 42, per_class=2)
    tokenizer = AutoTokenizer.from_pretrained(source["directory"], local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(source["directory"], local_files_only=True,
                                                              num_labels=3).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    weights = class_weights(train_rows, "sqrt-inverse").to(device)
    square_sum, coordinates = 0., 0
    with torch.no_grad():
        for batch in loader(scale_rows, tokenizer, 64, 2):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device).bool()
            for token_id in tokenizer.all_special_ids:
                mask &= ids != token_id
            embeddings = model.get_input_embeddings()(ids)[mask]
            square_sum += embeddings.double().square().sum().item()
            coordinates += embeddings.numel()
    rms = (square_sum / coordinates)**.5
    epsilons = [float(f"{rms * fraction:.6g}") for fraction in (.05, .1)]
    args.output_dir.mkdir(parents=True)
    result = {"completed": False, "seed": 42, "base_model_source": source,
              "dataset_sha256": manifest["dataset_sha256"],
              "split_sha256": file_sha256(ROOT / "outputs/eda/split.json"),
              "code_sha256": {name: file_sha256(ROOT / name) for name in
                              ("src/training.py", "tools/calibrate_fgsm.py", "main.py")},
              "scale_train_ids": [r["id"] for r in scale_rows], "audit_train_ids": [r["id"] for r in audit_rows],
              "embedding_rms": rms, "epsilon_rule": "0.05 and 0.10 of train-only word embedding RMS; 6 significant digits",
              "epsilons": epsilons, "checks": [], "optimizer_updates": 0,
              "gradient_audit_mode": "eval_dropout_disabled_for_precision_comparison",
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name()}}
    write_json(args.output_dir / "summary.json", result)
    started = time.perf_counter()
    for batch_index, raw in enumerate(loader(audit_rows, tokenizer, 64, 2)):
        batch = {k: v.to(device) for k, v in raw.items()}
        labels = batch.pop("labels")
        model.eval()
        with torch.no_grad():
            before = model(**batch).logits.clone()
        versions = [p._version for p in model.parameters()]
        for epsilon in epsilons:
            gradients = {}
            for amp in (False, True):
                # Same RNG seed does not guarantee identical dropout masks across
                # FP32/FP16 kernels. Disable dropout to isolate numerical drift.
                # Train-mode/checkpointing gradients are exercised by smoke/tests.
                model.eval()
                model.zero_grad(set_to_none=True)
                torch.manual_seed(42 + batch_index)
                torch.cuda.manual_seed_all(42 + batch_index)
                scaler = torch.amp.GradScaler("cuda", enabled=amp)
                loss, detail = fgsm_backward(model, batch, labels, weights, 16, scaler,
                                             epsilon, tokenizer.all_special_ids, amp, diagnostics=True)
                mask = batch["attention_mask"].bool().clone()
                for token_id in tokenizer.all_special_ids:
                    mask &= batch["input_ids"] != token_id
                delta = detail["delta"]
                assert torch.count_nonzero(delta[~mask]) == 0
                assert delta.abs().max() <= epsilon * 1.000001
                assert not delta.requires_grad
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                assert versions == [p._version for p in model.parameters()]
                gradients[amp] = detail["gradient"][mask].cpu()
                entry = {"batch": batch_index, "epsilon": epsilon, "amp": amp,
                         "scaler_scale": scaler.get_scale(), "loss": loss.item(),
                         "clean_loss": detail["clean_loss"].item(), "adversarial_loss": detail["adversarial_loss"].item(),
                         "delta_linf": delta.abs().max().item(), "masked_nonzero": 0,
                         "finite_parameter_gradients": True, "parameter_versions_unchanged": True}
                result["checks"].append(entry)
            a, b = gradients[False], gradients[True]
            result["checks"][-1]["amp_fp32_input_gradient_sign_agreement"] = (a.sign() == b.sign()).float().mean().item()
            result["checks"][-1]["amp_fp32_input_gradient_relative_l2"] = ((a - b).norm() / a.norm()).item()
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(**batch).logits, before, atol=0, rtol=0)
        print(f"Audited train batch {batch_index + 1}/3", flush=True)
    result.update(completed=True, clean_inference_unchanged=True,
                  seconds=time.perf_counter() - started,
                  peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20)
    result["precision_relative_l2_under_2pct"] = all(
        row.get("amp_fp32_input_gradient_relative_l2", 0) < .02 for row in result["checks"])
    write_json(args.output_dir / "summary.json", result)
    print({k: result[k] for k in ("completed", "embedding_rms", "epsilons", "seconds", "peak_allocated_mb")})
    print([{k: v for k, v in row.items() if k in (
        "batch", "epsilon", "amp_fp32_input_gradient_relative_l2", "amp_fp32_input_gradient_sign_agreement")}
           for row in result["checks"] if row["amp"]])


if __name__ == "__main__":
    main()
