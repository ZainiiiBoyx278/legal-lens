"""Run one local, greedy completion per case using LegalLens's evidence contract.

This tests the generator component, not the production provider or classifier.
It has no API client and requires already-downloaded model files. Preserve raw
responses, including failures; semantic support is scored by a separate review.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from scripts.evaluate_retrieval import sha256
from scripts.evaluation_integrity import code_sha256
from app.services.evidence_contract import build_messages, parse_answer
from app.services.question_policy import source_scope_reason

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SETTINGS = {"do_sample": False, "max_new_tokens": 1600, "max_input_tokens": 6144, "top_k_sources": 6, "seed": 20261113, "dtype": "bfloat16", "attention": "sdpa"}


def generation_hashes():
    return {p: code_sha256(BACKEND / p) for p in ("scripts/evaluate_local_answers.py", "app/services/evidence_contract.py", "app/services/question_policy.py", "scripts/evaluation_integrity.py", "evaluation/retrieval_v2/local-model.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--case-ids", nargs="+", help="Development smoke cases only")
    args = parser.parse_args()
    partial = args.output.with_suffix(".partial.json")
    if args.output.exists() or partial.exists() or not args.model_path.is_dir():
        parser.error("Use a new output path and an existing model snapshot")
    if args.case_ids and not args.development:
        parser.error("Holdout evaluation must run all cases")
    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    if retrieval["split"] != ("development" if args.development else "holdout"):
        raise ValueError("Wrong retrieval split")
    protocol = json.loads((BACKEND / "evaluation/retrieval_v2/protocol.json").read_text())
    if retrieval["questions_sha256"] != protocol["development_sha256" if args.development else "holdout_sha256"]:
        raise ValueError("Retrieval questions differ from the protocol")
    if not args.development:
        candidate = json.loads((BACKEND / "evaluation/retrieval_v2/candidate.json").read_text())
        if candidate["generation_code_sha256"] != generation_hashes() or candidate["generation_settings"] != SETTINGS:
            raise ValueError("Generator differs from frozen candidate")
        if candidate["retrieval_code_sha256"] != retrieval["retrieval_code_sha256"]:
            raise ValueError("Retrieval differs from frozen candidate")
    cases = [r for r in retrieval["queries"] if r["system"] == "v2_hybrid" and (not r["answerable"] or r["variant"] == "paraphrase")]
    if args.case_ids:
        cases = [r for r in cases if r["id"] in args.case_ids]
        if len(cases) != len(set(args.case_ids)):
            raise ValueError("Unknown smoke case ID")
    if not args.development and len(cases) != 60:
        raise ValueError("Expected all 60 frozen generation cases")
    manifest = json.loads((BACKEND / "evaluation/retrieval_v2/local-model.json").read_text())
    model_hashes = {name: sha256(args.model_path / name) for name in manifest["files_sha256"]}
    if model_hashes != manifest["files_sha256"] or manifest["model_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
        raise ValueError("Local files do not match the pinned generator snapshot")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OPENAI_API_KEY="")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("This measured configuration requires CUDA; choose a separately recorded CPU experiment if needed")
    set_seed(SETTINGS["seed"])
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, use_safetensors=True,
                                               torch_dtype=torch.bfloat16, attn_implementation=SETTINGS["attention"]).to("cuda").eval()
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "split": retrieval["split"],
              "scope": "Local generator component with shared production prompt/validator and source-scope guard. Classifier and HTTP routing bypassed; not OpenAI production performance.",
              "retrieval_sha256": sha256(args.retrieval), "questions_sha256": retrieval["questions_sha256"], "source_sha256": retrieval["source_sha256"],
              "generation_code_sha256": generation_hashes(), "settings": SETTINGS, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "code_hash_normalization": "UTF-8 with LF newlines; data and model hashes use raw bytes",
              "model_files_sha256": model_hashes,
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "gpu": torch.cuda.get_device_name(),
                              "versions": {p: version(p) for p in ("torch", "transformers", "pydantic", "numpy")}}, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for number, case in enumerate(cases, 1):
        row = {k: case[k] for k in ("id", "query", "answerable", "kind")}
        row.update(sources=case["results"][:SETTINGS["top_k_sources"]], claims=[], raw_response=None, generation_seconds=None)
        reason = source_scope_reason(case["query"])
        if reason:
            row.update(mode="scope_abstention", reason=reason)
        elif not row["sources"]:
            row.update(mode="retrieval_abstention", reason="No matching excerpts")
        else:
            messages = build_messages(case["query"], [], row["sources"])
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
            row["prompt_tokens"] = ids.shape[-1]
            if ids.shape[-1] > SETTINGS["max_input_tokens"]:
                # No hidden prompt truncation, excerpt deletion, or retries.
                row.update(mode="context_failure", reason="Input exceeds the frozen token budget")
            else:
                torch.cuda.synchronize()
                start = perf_counter()
                with torch.inference_mode():
                    output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                            temperature=None, top_p=None, top_k=None, max_new_tokens=SETTINGS["max_new_tokens"],
                                            pad_token_id=tokenizer.pad_token_id, logits_to_keep=1)
                torch.cuda.synchronize()
                row["generation_seconds"] = perf_counter() - start
                generated = output[0, ids.shape[-1]:]
                row["output_tokens"] = len(generated)
                row["raw_response"] = tokenizer.decode(generated, skip_special_tokens=True)
                eos = model.generation_config.eos_token_id
                eos = {eos} if isinstance(eos, int) else set(eos or [])
                try:
                    if int(generated[-1]) not in eos:
                        raise ValueError("Completion reached its token limit without EOS")
                    answer = parse_answer(row["raw_response"], row["sources"])
                    row.update(mode="generated" if answer.sufficient_evidence else "model_abstention", claims=[c.model_dump() for c in answer.claims], reason=None)
                except ValueError as error:
                    row.update(mode="validation_failure", reason=str(error))
        result["cases"].append(row)
        partial.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"case": row["id"], "number": number, "total": len(cases), "mode": row["mode"], "seconds": row["generation_seconds"]}), flush=True)
    partial.replace(args.output)


if __name__ == "__main__":
    main()
