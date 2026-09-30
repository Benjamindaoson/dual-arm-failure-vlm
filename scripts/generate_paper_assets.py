from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("Base", "SFT", "GRPO", "GSPO")


def _pct(value: float) -> str:
    return f"{100 * value:.1f}"


def render_v1_assets(reanalysis: dict, distributions: dict) -> dict[str, str]:
    runs = reanalysis["runs"]
    if set(runs) != set(ORDER) or set(distributions) != set(ORDER):
        raise ValueError("all four frozen V1 conditions are required")
    rows = []
    progression = []
    states = []
    colors = ("blue!70", "orange!80", "red!65")
    for index, name in enumerate(ORDER):
        item = runs[name]
        semantic = item["semantic"]
        strict = item["protocol"]
        distribution = distributions[name]
        rows.append(f"{name} & {_pct(strict['json_valid_rate'])} & {_pct(strict['state_macro_f1'])} & {_pct(semantic['state_macro_f1'])} & {_pct(semantic['failure_recall'])} & {_pct(distribution['collapse_ratio'])} " + r"\\")
        x = 1.0 + index * 1.55
        for shift, value, color in ((-0.34, semantic["failure_recall"], "red!65"),
                                    (-0.08, strict["json_valid_rate"], "blue!70"),
                                    (0.18, distribution["collapse_ratio"], "orange!80")):
            progression.append(f"\\fill[{color}] ({x + shift:.2f},0) rectangle ({x + shift + 0.20:.2f},{value:.4f});")
        progression.append(f"\\node[below,font=\\scriptsize] at ({x:.2f},0) {{{name}}};")
        bottom = 0.0
        for label, color in zip(("nominal", "failure", "recovery"), colors, strict=True):
            value = distribution["probabilities"][label]
            states.append(f"\\fill[{color}] ({x - 0.28:.2f},{bottom:.4f}) rectangle ({x + 0.28:.2f},{bottom + value:.4f});")
            bottom += value
        states.append(f"\\node[below,font=\\scriptsize] at ({x:.2f},0) {{{name}}};")
    table = "\n".join([
        "% Generated from artifacts/v2/v1_reanalysis; values are percentages.",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Model & Strict valid & Strict F1 & Semantic F1 & Failure recall & Collapse " + r"\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
    ]) + "\n"
    axes = "\\draw[->] (0.35,0) -- (6.9,0);\n\\draw[->] (0.45,0) -- (0.45,1.12);\n\\foreach \\y in {0,0.5,1} {\\draw (0.40,\\y) -- (0.50,\\y); \\node[left,font=\\tiny] at (0.40,\\y) {\\y};}\n"
    fig1 = "\n".join(["% Generated from frozen V1 prediction reanalysis.",
        "\\begin{tikzpicture}[x=1cm,y=3.4cm]", axes, *progression,
        "\\node[font=\\tiny,anchor=west] at (0.55,1.13) {\\textcolor{red!65}{\\rule{5pt}{5pt}} Failure recall (semantic) \\quad \\textcolor{blue!70}{\\rule{5pt}{5pt}} Strict valid \\quad \\textcolor{orange!80}{\\rule{5pt}{5pt}} Collapse};",
        "\\end{tikzpicture}"]) + "\n"
    fig2 = "\n".join(["% Generated from frozen V1 semantic state predictions.",
        "\\begin{tikzpicture}[x=1cm,y=3.4cm]", axes, *states,
        "\\node[font=\\tiny,anchor=west] at (0.55,1.13) {\\textcolor{blue!70}{\\rule{5pt}{5pt}} Nominal \\quad \\textcolor{orange!80}{\\rule{5pt}{5pt}} Failure \\quad \\textcolor{red!65}{\\rule{5pt}{5pt}} Recovery};",
        "\\end{tikzpicture}"]) + "\n"
    return {"tables/v1.tex": table, "figures/v1_progression.tex": fig1,
            "figures/state_distribution.tex": fig2}


def render_camera_table(metrics: dict[str, dict[str, dict]]) -> str:
    if set(metrics) != {"C0", "C1", "C2"}:
        raise ValueError("all three camera screens are required")
    rows = []
    for camera in ("C0", "C1", "C2"):
        base = metrics[camera]["base"]
        sft = metrics[camera]["sft"]
        rows.append(f"{camera} & {_pct(base['failure_recall'])} & {_pct(sft['failure_recall'])} & {_pct(sft['failure_precision'])} & {_pct(sft['state_macro_f1'])} & {_pct(sft['recovery_recall'])} " + r"\\")
    return "\n".join([
        "% Generated from V2 validation metrics, not final-test results.",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        "View & Base F-rec & SFT F-rec & SFT F-prec & SFT State F1 & SFT R-rec " + r"\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
    ]) + "\n"


def main() -> int:
    source = ROOT / "artifacts/v2/v1_reanalysis"
    paths = [source / "strict_vs_semantic.json", source / "predicted_state_distribution.json"]
    reanalysis, distributions = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assets = render_v1_assets(reanalysis, distributions)
    camera_paths = {camera: {stage: ROOT / f"artifacts/v2/runs/{camera.lower()}-{stage}-val/metrics.json"
                             for stage in ("base", "sft")} for camera in ("C0", "C1", "C2")}
    camera_metrics = {camera: {stage: json.loads(path.read_text(encoding="utf-8"))["semantic_diagnostic"]
                               for stage, path in stages.items()} for camera, stages in camera_paths.items()}
    assets["tables/camera_screen.tex"] = render_camera_table(camera_metrics)
    paths.extend(path for stages in camera_paths.values() for path in stages.values())
    for name, body in assets.items():
        destination = ROOT / "paper" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(body, encoding="utf-8")
    receipt = {"generator": "scripts/generate_paper_assets.py",
               "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
               "generated_files": sorted(assets)}
    (ROOT / "paper/asset_sources.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
