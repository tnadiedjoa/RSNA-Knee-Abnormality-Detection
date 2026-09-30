"""Grade the 12 findings in every training report with a local open-weights LLM (vLLM).

Report text never leaves the cluster (competition rule 4.b: no hosted LLM API).
Each finding gets an ordinal grade - absent / equivocal / mild / moderate / severe - or
not_mentioned, plus a short English evidence quote. Grades are turned into soft labels later
(knee/labels.py), so the extraction itself stays free of any tuning on the gold studies.

Run with the LLM venv on one RTX 3090:  .venv-llm/bin/python -m scripts.llm_labels --name qwen36_27b_v1
"""

import argparse
import json
import time

import pandas as pd

from knee.config import DATA_DIR, LABELS

GRADES = ["absent", "equivocal", "mild", "moderate", "severe", "not_mentioned"]

SYSTEM = "You are a senior musculoskeletal radiologist who extracts structured findings from knee MRI reports."

EVIDENCE_LINE = ("For each finding also give \"evidence\": the report's own words that support the grade, "
                 "translated to English, at most 15 words (\"\" if not_mentioned).\n\n")

RUBRIC = """Read the knee MRI report below (it may be written in any language) and grade each of the 12 findings.

Grades:
- "absent": the report explicitly states the structure is normal/intact or the finding is not present.
- "equivocal": minimal, questionable or borderline (e.g. trace fluid, grade 1 sprain, intrasubstance/mucoid degeneration without a tear, possible/suspected finding, minimal chondrosis).
- "mild": present but small or low grade.
- "moderate": clearly present, intermediate severity.
- "severe": large, high grade, complete or complex.
- "not_mentioned": the report says nothing that bears on this finding.

Findings and how to grade them:
1. ACL - anterior cruciate ligament injury. Sprain/grade 1 or mucoid degeneration -> equivocal; low-grade partial tear -> mild; high-grade partial or interstitial tear, grade 2 -> moderate; complete rupture/grade 3 -> severe. A reconstructed ACL with an intact graft -> absent; a torn graft -> grade the graft tear.
2. MCL - medial collateral ligament injury. Periligamentous edema or grade 1 sprain -> equivocal; low-grade partial tear -> mild; grade 2 / partial tear -> moderate; complete tear / grade 3 -> severe.
3. Medial Meniscus - tear of the medial meniscus. Intrasubstance (grade 1-2) signal without surface extension or a meniscectomy without re-tear -> equivocal; possible/suspected tear -> mild; definite tear -> moderate; complex, displaced, bucket-handle, flap, root tear or maceration -> severe.
4. Lateral Meniscus - same scale for the lateral meniscus.
5. Medial OA - osteoarthritis of the medial tibiofemoral compartment (chondrosis/cartilage loss, osteophytes, subchondral sclerosis/cysts, degenerative marrow edema, joint-space narrowing). Minimal/early chondrosis -> equivocal; mild/low-grade -> mild; moderate -> moderate; severe, full-thickness loss, bone on bone -> severe.
6. Lateral OA - same, lateral tibiofemoral compartment.
7. PF OA - same, patellofemoral compartment (patella and trochlea cartilage).
   If the report describes osteoarthritis/degenerative change of the knee without naming a compartment, apply that grade to all three OA findings.
8. Effusion - joint effusion (fluid in the joint itself, not in a cyst or bursa outside the joint). Physiologic/trace -> equivocal; small -> mild; moderate -> moderate; large or hemarthrosis/lipohemarthrosis -> severe.
9. Synovitis - synovial inflammation: synovitis, synovial thickening/hypertrophy/proliferation, Hoffa fat pad edema or impingement, inflamed or thickened plicae, lipoma arborescens, debris-laden or complex effusion. Grade by stated severity.
10. Baker's - popliteal (Baker's) cyst. Trace/tiny -> equivocal; small -> mild; moderate -> moderate; large or ruptured/leaking -> severe.
11. Contusion - bone contusion / bone bruise / post-traumatic bone marrow edema. Purely degenerative subchondral edema -> equivocal. Soft-tissue or muscle contusion alone is not a bone contusion.
12. Fracture - any fracture: avulsion (incl. Segond), osteochondral or impaction fracture, subchondral insufficiency fracture, stress fracture, tibial plateau fracture. Occult/suspected -> mild; definite -> moderate; displaced, depressed or comminuted -> severe.

{evidence_line}Report:
<<<
{report}
>>>"""
RUBRIC = RUBRIC.replace("{evidence_line}", EVIDENCE_LINE)


def schema(evidence: bool = True):
    if not evidence:  # grades only: ~3x fewer generated tokens
        return {"type": "object", "properties": {label: {"type": "string", "enum": GRADES} for label in LABELS},
                "required": LABELS, "additionalProperties": False}
    item = {"type": "object", "properties": {"grade": {"type": "string", "enum": GRADES},
                                             "evidence": {"type": "string", "maxLength": 160}},
            "required": ["grade", "evidence"], "additionalProperties": False}
    return {"type": "object", "properties": {label: item for label in LABELS}, "required": LABELS,
            "additionalProperties": False}


def parse(uid: str, text: str) -> dict:
    row = {"StudyInstanceUID": uid, "raw": text}
    try:
        parsed = json.loads(text[text.index("{"):])
        for label in LABELS:
            item = parsed[label]
            row[label] = item["grade"] if isinstance(item, dict) else item
            if isinstance(item, dict):
                row[f"{label}__evidence"] = item["evidence"]
    except (ValueError, KeyError) as e:
        row["error"] = repr(e)
    return row


def main(args):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    df = pd.read_csv(DATA_DIR / "train.csv")
    if args.gold_only:
        df = df.dropna(subset=LABELS)
    if args.limit:
        df = df.head(args.limit)
    out_dir = DATA_DIR / "labels"
    parts_dir = out_dir / f"{args.name}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_len,
              gpu_memory_utilization=args.gpu_mem, limit_mm_per_prompt={"image": 0, "video": 0}, seed=args.seed,
              enable_prefix_caching=True, enforce_eager=args.eager, max_num_seqs=args.max_seqs,
              max_num_batched_tokens=args.batched_tokens)
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed,
                            structured_outputs=StructuredOutputsParams(json=schema(args.evidence)))
    rubric = RUBRIC if args.evidence else RUBRIC.replace(EVIDENCE_LINE, "")
    # chunks saved as they finish, so an interrupted run resumes where it stopped
    for start in range(0, len(df), args.chunk):
        part = parts_dir / f"part{start // args.chunk:04d}.parquet"
        if part.exists():
            continue
        chunk = df.iloc[start : start + args.chunk]
        messages = [[{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": rubric.format(report=r.strip())}] for r in chunk["Report"]]
        t = time.time()
        outputs = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": args.thinking}, use_tqdm=False)
        rows = [parse(uid, o.outputs[0].text) for uid, o in zip(chunk["StudyInstanceUID"], outputs)]
        pd.DataFrame(rows).to_parquet(part, index=False)
        print(f"{start + len(chunk)}/{len(df)} reports ({time.time() - t:.0f}s for this chunk)", flush=True)

    res = pd.concat([pd.read_parquet(p) for p in sorted(parts_dir.glob("part*.parquet"))], ignore_index=True)
    res.to_parquet(out_dir / f"{args.name}.parquet", index=False)
    n_err = res["error"].notna().sum() if "error" in res else 0
    print(f"saved {out_dir / (args.name + '.parquet')} ({len(res)} reports, {n_err} parse errors)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Intel/Qwen3.6-27B-int4-AutoRound")
    p.add_argument("--name", default="qwen36_v1")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-len", type=int, default=6144)
    p.add_argument("--gpu-mem", type=float, default=0.95)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--eager", action=argparse.BooleanOptionalAction, default=True,
                   help="no CUDA graphs: they do not fit next to a 27B model on a 24 GB GPU")
    p.add_argument("--max-seqs", type=int, default=32)
    p.add_argument("--evidence", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batched-tokens", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--gold-only", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
