"""Recover an interrupted frozen run without regenerating completed cases.

The frozen evaluator stays byte-for-byte unchanged. This narrow wrapper changes
only its existing-output guard, restores its checkpoint, and skips the already
saved prefix. Decoding, prompts, validation, sources and settings stay unchanged.
The output records this helper's hash separately from the frozen evaluator hash.
"""
from pathlib import Path

ORIGINAL = Path(__file__).with_name("evaluate_local_answers.py")
source = ORIGINAL.read_text(encoding="utf-8")
replacements = {
    "if args.output.exists() or partial.exists() or not args.model_path.is_dir():":
        "if args.output.exists() or not partial.exists() or not args.model_path.is_dir():",
    "    args.output.parent.mkdir(parents=True, exist_ok=True)": '''    interrupted = json.loads(partial.read_text(encoding="utf-8"))
    for identity in ("generation_code_sha256", "questions_sha256", "retrieval_sha256", "source_sha256", "settings", "model_files_sha256", "environment"):
        if result[identity] != interrupted[identity]:
            raise ValueError("Resume identity changed: " + identity)
    completed = len(interrupted["cases"])
    if [c["id"] for c in interrupted["cases"]] != [c["id"] for c in cases[:completed]]:
        raise ValueError("Saved cases are not the exact expected prefix")
    if completed >= len(cases):
        raise ValueError("No unfinished cases")
    archive = partial.with_name("holdout-generation-interrupted.json")
    if archive.exists():
        raise ValueError("An interrupted-run archive already exists")
    archive.write_bytes(partial.read_bytes())
    result = interrupted
    result["interruption_recovery"] = {
        "reason": "Workspace relocation interrupted the process. Resume only unsaved cases; retain every completed output unchanged.",
        "preserved_completed_cases": completed,
        "interrupted_output_sha256": sha256(archive),
        "resume_helper_sha256": sha256(__resume_helper__),
        "resumed_utc": datetime.now(timezone.utc).isoformat(),
        "decoder_prompt_and_settings_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)''',
    "    for number, case in enumerate(cases, 1):":
        "    for number, case in enumerate(cases, 1):\n        if number <= completed:\n            continue",
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise ValueError("Frozen evaluator structure changed; refusing an ambiguous resume")
    source = source.replace(old, new)
exec(compile(source, str(ORIGINAL), "exec"), {
    "__name__": "__main__", "__file__": str(ORIGINAL),
    "__resume_helper__": str(Path(__file__).resolve()),
})
