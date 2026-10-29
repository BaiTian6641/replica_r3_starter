"""Collect bounded public text sources for a review; never fetch model weights.

Run from the repository root. This writes only this report's sources/ and
source_manifest.json. It neither imports downloaded code nor changes model pins.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
MAX_BYTES = 2 * 1024 * 1024
REVISIONS = {
    "bf16_pinned": ("Spark-X2.6-4B", "66e3ae7bee54e0a523d692cb8ef841f35409807b"),
    "bf16_current": ("Spark-X2.6-4B", "d1b5f26071101c27fcd07728e7a7770e428281f4"),
    "fp8_pinned": ("Spark-X2.6-4B-FP8", "0318d6f6ab070c7b5b43052db571489b2508006c"),
    "fp8_current": ("Spark-X2.6-4B-FP8", "15b16e794bf251505f68501294fce730cceaa4e9"),
    "x25": ("Spark-X2.5-4B", "0bcb35678590218655dff3765b9e61c83b35e9c4"),
    "base": ("Spark-X2.6-4B-Base", "88439533d38a0e6a81125c72ec9fad50d6f98692"),
}


def get_source(spec):
    name, url = spec
    entry = {"file": "sources/" + name, "url": url,
             "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Spark-Architecture-Review/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = response.read(MAX_BYTES + 1)
            entry["resolved_url"] = response.url
        if len(payload) > MAX_BYTES:
            raise ValueError("public text source exceeds 2 MiB cap")
        payload.decode("utf-8")
        (SOURCES / name).write_bytes(payload)
        entry.update(status="ok", bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    except Exception as exc:
        entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return entry


def main():
    SOURCES.mkdir(parents=True, exist_ok=True)
    specs = []
    for label, (repo, revision) in REVISIONS.items():
        files = ["config.json", "modeling_spark.py", "configuration_spark.py", "chat_template.jinja", "README.md"]
        if label in ("bf16_pinned", "bf16_current", "x25"):
            files += ["generation_config.json", "tokenizer_config.json", "model.safetensors.index.json"]
        for name in files:
            specs.append((label + "__" + name + ".txt",
                          f"https://huggingface.co/XHToken/{repo}/resolve/{revision}/{name}"))
        specs.append((label + "__hub_metadata.json",
                      f"https://huggingface.co/api/models/XHToken/{repo}/revision/{revision}?blobs=true"))
    for repo, label in [("XHToken/Spark-X2.6", "spark_github"),
                        ("XHToken/Spark-MLX-LLM", "mlx"),
                        ("XHToken/Spark-plugin", "vllm_plugin")]:
        specs.append((label + "__commits.json", f"https://api.github.com/repos/{repo}/commits?per_page=1"))
    # These immutable backend commits were resolved during the dated review.
    specs.extend([
        ("mlx__tree.json", "https://api.github.com/repos/XHToken/Spark-MLX-LLM/git/trees/079a6a328ef878ac3cc4884c07e830add0540007?recursive=1"),
        ("plugin__tree.json", "https://api.github.com/repos/XHToken/Spark-plugin/git/trees/3e1040e63a5907e4c748a485d6795d739dcd5bd6?recursive=1"),
        ("spark_github__README.md.txt", "https://raw.githubusercontent.com/XHToken/Spark-X2.6/27ef075943cd9af43c1d1138e2ee267f1a4f872c/README.md"),
        ("mlx__README.md.txt", "https://raw.githubusercontent.com/XHToken/Spark-MLX-LLM/079a6a328ef878ac3cc4884c07e830add0540007/README.md"),
        ("mlx__model.py.txt", "https://raw.githubusercontent.com/XHToken/Spark-MLX-LLM/079a6a328ef878ac3cc4884c07e830add0540007/spark_mlx_llm/model.py"),
        ("mlx__loader.py.txt", "https://raw.githubusercontent.com/XHToken/Spark-MLX-LLM/079a6a328ef878ac3cc4884c07e830add0540007/spark_mlx_llm/loader.py"),
        ("plugin__spark2_5.py.txt", "https://raw.githubusercontent.com/XHToken/Spark-plugin/3e1040e63a5907e4c748a485d6795d739dcd5bd6/src/vllm_spark2_5_plugin/spark2_5.py"),
        ("bf16_current__all_benchmark.svg.txt", "https://huggingface.co/XHToken/Spark-X2.6-4B/resolve/d1b5f26071101c27fcd07728e7a7770e428281f4/images/all_benchmark.svg"),
        ("spark_hf__LICENSE.txt", "https://huggingface.co/XHToken/Spark-X2.6-4B/resolve/66e3ae7bee54e0a523d692cb8ef841f35409807b/LICENSE"),
        ("spark_github__LICENSE.txt", "https://raw.githubusercontent.com/XHToken/Spark-X2.6/27ef075943cd9af43c1d1138e2ee267f1a4f872c/LICENSE"),
        ("mlx__LICENSE.txt", "https://raw.githubusercontent.com/XHToken/Spark-MLX-LLM/079a6a328ef878ac3cc4884c07e830add0540007/LICENSE"),
        ("plugin__LICENSE.txt", "https://raw.githubusercontent.com/XHToken/Spark-plugin/3e1040e63a5907e4c748a485d6795d739dcd5bd6/LICENSE"),
        ("plugin__NOTICE.txt", "https://raw.githubusercontent.com/XHToken/Spark-plugin/3e1040e63a5907e4c748a485d6795d739dcd5bd6/NOTICE"),
        ("plugin__THIRD_PARTY_LICENSES.md.txt", "https://raw.githubusercontent.com/XHToken/Spark-plugin/3e1040e63a5907e4c748a485d6795d739dcd5bd6/THIRD_PARTY_LICENSES.md"),
    ])
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(get_source, specs))
    manifest = {"review_date_local": "2026-10-28", "timezone": "America/Chicago",
                "weight_payloads_downloaded": False, "downloaded_code_executed": False,
                "model_source_pins_changed": False, "sources": entries}
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sources": len(entries), "successful": sum(e["status"] == "ok" for e in entries),
                      "bytes": sum(e.get("bytes", 0) for e in entries),
                      "errors": [e for e in entries if e["status"] != "ok"]}, indent=2))


if __name__ == "__main__":
    main()
