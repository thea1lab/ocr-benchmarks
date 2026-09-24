"""Run the local OCR models on corpus/manifest.json and write the results.

Usage:
    python run.py
    python run.py rapidocr lighton-ocr
"""

import html
import json
import os
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIMEOUT_S = 90 * 60


def normalize(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9.%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def without_loop(text):
    """Drop a tail that repeats itself. A nutrition label can repeat a short number; a loop repeats a lot."""
    lines = []
    last_line = None
    run = 0
    looped = False
    for line in text.splitlines():
        key = line.strip()
        if key and key == last_line:
            run += 1
            if run >= 3:
                looped = True
                continue
        else:
            last_line = key
            run = 0
        lines.append(line)
    blocks = re.split(r"\n\s*\n", "\n".join(lines))
    seen = {}
    kept = []
    for block in blocks:
        key = " ".join(block.split())
        key = re.sub(r"</?think>", "", key).strip()
        if not key:
            continue
        seen[key] = seen.get(key, 0) + 1
        if (len(key) >= 16 and seen[key] >= 2) or seen[key] >= 4:
            looped = True
            break
        kept.append(block.strip())
    return "\n\n".join(kept), looped


def levenshtein(a, b):
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def character_error(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return None
    return levenshtein(ref, hyp) / len(ref)


def field_hits(fields, hypothesis):
    hyp = normalize(hypothesis)
    hits = []
    for field in fields:
        needle = normalize(field)
        hits.append({"field": field, "hit": bool(needle) and needle in hyp})
    found = sum(item["hit"] for item in hits)
    return {"found": found, "total": len(hits), "fields": hits}


def load_pages():
    manifest = json.loads((ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    pages = []
    for page in manifest["pages"]:
        reference = ""
        if page.get("reference"):
            reference = (ROOT / page["reference"]).read_text(encoding="utf-8")
        pages.append({**page, "reference_text": reference})
    return pages


def parse_log(text):
    parsed = {"pages": {}, "device": None, "load_seconds": None, "vram_mib": None}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[0] == "DEVICE":
            parsed["device"] = parts[1]
        elif parts[0] == "LOADED":
            parsed["load_seconds"] = float(parts[1])
        elif parts[0] == "VRAM_MIB":
            parsed["vram_mib"] = float(parts[1])
        elif parts[0] == "PAGE" and len(parts) >= 3:
            parsed["pages"][parts[1]] = float(parts[2])
    return parsed


def score(page, output_path, seconds):
    scored = {"id": page["id"], "task": page["task"], "seconds": seconds}
    if not output_path.is_file():
        scored["missing"] = True
        return scored
    raw = output_path.read_text(encoding="utf-8")
    text, looped = without_loop(raw)
    scored["looped"] = looped
    scored["excerpt"] = text[:500]
    scored["output"] = str(output_path.relative_to(ROOT))
    if page.get("reference_text"):
        scored["cer"] = character_error(page["reference_text"], text)
    if page.get("fields"):
        scored["fields"] = field_hits(page["fields"], raw)
    return scored


def run_model(spec, pages):
    record = {
        "id": spec["id"],
        "name": spec["name"],
        "params": spec["params"],
        "license": spec["license"],
        "kind": spec["kind"],
        "url": spec["url"],
        "note": spec.get("note"),
        "prior": spec.get("prior"),
        "status": "not_run",
        "pages": [],
    }
    if not spec.get("run"):
        return record
    python = ROOT / spec["dir"] / ".venv" / "bin" / "python"
    if not python.is_file():
        record["status"] = "not_installed"
        record["error"] = f"No venv at {spec['dir']}/.venv"
        return record

    out_dir = ROOT / "results" / "outputs" / spec["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    suite = ROOT / "results" / "suites" / f"{spec['id']}.tsv"
    suite.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for page in pages:
        output = out_dir / f"{page['id']}.md"
        lines.append(f"{(ROOT / page['image']).resolve()}\t{output.resolve()}")
    suite.write_text("\n".join(lines) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    try:
        completed = subprocess.run(
            [str(python), "ocr.py", "--suite", str(suite)],
            cwd=ROOT / spec["dir"],
            env=env,
            text=True,
            capture_output=True,
            timeout=TIMEOUT_S,
        )
        code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimed out after {TIMEOUT_S}s"

    log = ROOT / "results" / "logs" / f"{spec['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
    parsed = parse_log(stdout)
    record["pages"] = [
        score(page, out_dir / f"{page['id']}.md", parsed["pages"].get(page["id"]))
        for page in pages
    ]
    record["status"] = "ok" if code == 0 else "error"
    record["device"] = parsed["device"]
    record["load_seconds"] = parsed["load_seconds"]
    record["vram_mib"] = parsed["vram_mib"]
    if code != 0:
        record["error"] = "\n".join(stderr.strip().splitlines()[-8:])
    fill_summary(record)
    return record


def pct(value):
    if value is None:
        return "—"
    return f"{max(0.0, 1 - value) * 100:.0f}%"


def num(value):
    return "—" if value is None else f"{value:.2f}"


def missed(page):
    fields = page.get("fields")
    if not fields:
        return ""
    names = [item["field"] for item in fields["fields"] if not item["hit"]]
    if not names:
        text = f"{fields['found']}/{fields['total']}"
    else:
        text = f"{fields['found']}/{fields['total']}, missed {', '.join(names)}"
    if page.get("looped") and page.get("cer") is not None:
        text += ". Then the model repeated itself"
    return text


def write_results(document):
    machine = document["machine"]
    lines = [
        "# OCR results",
        "",
        "Small open models, same pages, this machine.",
        f"{machine.get('cpu')}, {machine.get('ram_gib')} GiB RAM, {machine.get('name')}.",
        f"GPU memory in use when this run started: {machine.get('vram_used_mib')} of {machine.get('vram_mib')} MiB.",
        f"Generated {document['generated']}.",
        "",
        "Character accuracy ignores HTML and LaTeX markup, and it ignores a tail the model repeated. It is averaged over pages that have a transcript.",
        "Fields are the facts listed in `corpus/manifest.json`. A miss means that fact is not in the output.",
        "Seconds are warm extraction, after the weights load.",
        "",
        "| Model | Size | Result | Median s/page | Load s | Character accuracy | Fields |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in document["models"]:
        fields = "—"
        if model.get("fields_total"):
            fields = f"{model['fields_found']}/{model['fields_total']}"
        lines.append(
            f"| {model['name']} | {model['params']} | {model['status']} | "
            f"{num(model.get('median_seconds'))} | {num(model.get('load_seconds'))} | "
            f"{pct(model.get('mean_cer'))} | {fields} |"
        )
    lines += ["", "## Pages", ""]
    for page in document["pages"]:
        lines += [f"### {page['task']}", ""]
        if page.get("note"):
            lines += [page["note"], ""]
        lines += ["| Model | Seconds | Character accuracy | Fields |", "| --- | ---: | ---: | --- |"]
        for model in document["models"]:
            scored = next((item for item in model["pages"] if item["id"] == page["id"]), None)
            if not scored or scored.get("missing"):
                continue
            lines.append(
                f"| {model['name']} | {num(scored.get('seconds'))} | {pct(scored.get('cer'))} | {missed(scored)} |"
            )
        lines.append("")
        for model in document["models"]:
            scored = next((item for item in model["pages"] if item["id"] == page["id"]), None)
            if scored and scored.get("excerpt"):
                lines += [f"<details><summary>{model['name']} excerpt</summary>", "", "```", scored["excerpt"].rstrip(), "```", "", "</details>", ""]
    lines += ["## Notes", ""]
    for model in document["models"]:
        lines += [f"### {model['name']}", "", model.get("note") or "", ""]
        if model.get("error"):
            lines += ["```", model["error"], "```", ""]
        if model.get("prior"):
            prior = model["prior"]
            lines += [
                f"Earlier run on {prior.get('machine')}: {prior.get('seconds')}s, {prior.get('detail')}",
                "",
            ]
    lines.append("Re-run with `python run.py` from the repo root.")
    lines.append("")
    text = "\n".join(lines)
    (ROOT / "RESULTS.md").write_text(text, encoding="utf-8")
    write_html(document)


def write_html(document):
    rows = []
    for model in document["models"]:
        fields = "—"
        if model.get("fields_total"):
            fields = f"{model['fields_found']}/{model['fields_total']}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(model['name'])}<div class='muted'>{html.escape(model['params'])}</div></td>"
            f"<td>{html.escape(model['status'])}</td>"
            f"<td class='num'>{num(model.get('median_seconds'))}</td>"
            f"<td class='num'>{num(model.get('load_seconds'))}</td>"
            f"<td class='num'>{pct(model.get('mean_cer'))}</td>"
            f"<td class='num'>{fields}</td>"
            "</tr>"
        )
    page_blocks = []
    for page in document["pages"]:
        body = []
        for model in document["models"]:
            scored = next((item for item in model["pages"] if item["id"] == page["id"]), None)
            if not scored or scored.get("missing"):
                continue
            excerpt = html.escape(scored.get("excerpt") or "")
            body.append(
                "<tr>"
                f"<td>{html.escape(model['name'])}</td>"
                f"<td class='num'>{num(scored.get('seconds'))}</td>"
                f"<td class='num'>{pct(scored.get('cer'))}</td>"
                f"<td>{html.escape(missed(scored))}</td>"
                "</tr>"
                f"<tr><td colspan='4'><pre>{excerpt}</pre></td></tr>"
            )
        note = f"<p class='muted'>{html.escape(page['note'])}</p>" if page.get("note") else ""
        page_blocks.append(
            f"<h2>{html.escape(page['task'])}</h2>{note}"
            "<table><thead><tr><th>Model</th><th>Seconds</th><th>Character accuracy</th><th>Fields</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>"
        )
    machine = document["machine"]
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR results</title>
<style>
body {{ margin: 0; background: #f6f3ec; color: #1c1915; font: 16px/1.45 "Noto Sans", sans-serif; }}
main {{ max-width: 980px; margin: 0 auto; padding: 32px 18px 64px; }}
table {{ width: 100%; border-collapse: collapse; background: #fffcf7; }}
th, td {{ border-bottom: 1px solid #e3dcd0; padding: 8px 10px; text-align: left; vertical-align: top; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.muted {{ color: #5c564c; }}
pre {{ white-space: pre-wrap; background: #f3efe6; padding: 8px; }}
</style></head><body><main>
<h1>OCR results</h1>
<p class="muted">{html.escape(str(machine.get('cpu')))}. {html.escape(str(machine.get('name')))}.
{html.escape(str(machine.get('vram_used_mib')))} of {html.escape(str(machine.get('vram_mib')))} MiB already in use.
{html.escape(document['generated'])}.</p>
<table><thead><tr><th>Model</th><th>Result</th><th>Median s/page</th><th>Load s</th><th>Character accuracy</th><th>Fields</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
{''.join(page_blocks)}
<p class="muted">Same numbers as RESULTS.md. Re-run with <code>python run.py</code>.</p>
</main></body></html>
"""
    path = ROOT / "dashboard" / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def machine_info():
    cpu = "unknown"
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    ram = None
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            ram = round(int(line.split()[1]) / 1024 / 1024, 1)
    gpu = {"name": None, "vram_mib": None, "vram_used_mib": None}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        name, total, used = [part.strip() for part in out.split(",")]
        gpu = {"name": name, "vram_mib": int(float(total)), "vram_used_mib": int(float(used))}
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return {"cpu": cpu, "ram_gib": ram, **gpu}


def fill_summary(record):
    cers = [page["cer"] for page in record["pages"] if page.get("cer") is not None]
    record["mean_cer"] = sum(cers) / len(cers) if cers else None
    found = sum(page["fields"]["found"] for page in record["pages"] if page.get("fields"))
    total = sum(page["fields"]["total"] for page in record["pages"] if page.get("fields"))
    record["fields_found"] = found
    record["fields_total"] = total
    times = sorted(page["seconds"] for page in record["pages"] if page.get("seconds") is not None)
    record["median_seconds"] = times[len(times) // 2] if times else None


def rescore():
    latest = ROOT / "results" / "latest.json"
    document = json.loads(latest.read_text(encoding="utf-8"))
    pages = load_pages()
    by_id = {page["id"]: page for page in pages}
    specs = {
        spec["id"]: spec
        for spec in json.loads((ROOT / "bench" / "models.json").read_text(encoding="utf-8"))
    }
    for model in document["models"]:
        spec = specs.get(model["id"])
        if spec:
            model["note"] = spec.get("note")
        if not model.get("pages"):
            continue
        rescored = []
        for old in model["pages"]:
            page = by_id.get(old["id"])
            if page is None:
                continue
            rescored.append(score(page, ROOT / "results" / "outputs" / model["id"] / f"{page['id']}.md", old.get("seconds")))
        model["pages"] = rescored
        if model.get("status") == "ok":
            fill_summary(model)
    document["pages"] = [{key: value for key, value in page.items() if key != "reference_text"} for page in pages]
    latest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    write_results(document)
    print("Rescored RESULTS.md")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--rescore":
        rescore()
        return
    specs = json.loads((ROOT / "bench" / "models.json").read_text(encoding="utf-8"))
    wanted = set(sys.argv[1:])
    pages = load_pages()
    previous = {}
    latest = ROOT / "results" / "latest.json"
    if wanted and latest.is_file():
        previous = {model["id"]: model for model in json.loads(latest.read_text(encoding="utf-8"))["models"]}

    models = []
    for spec in specs:
        if wanted and spec["id"] not in wanted:
            models.append(previous.get(spec["id"]) or run_model({**spec, "run": False}, pages))
            continue
        print(f"== {spec['name']} ==", flush=True)
        record = run_model(spec, pages)
        print(record["status"], "cer", record.get("mean_cer"), "fields", record.get("fields_found"), flush=True)
        models.append(record)

    public_pages = [{key: value for key, value in page.items() if key != "reference_text"} for page in pages]
    document = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "machine": machine_info(),
        "pages": public_pages,
        "models": models,
    }
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    write_results(document)
    print("Wrote RESULTS.md and dashboard/index.html")


if __name__ == "__main__":
    main()
