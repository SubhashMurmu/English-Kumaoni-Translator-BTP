import json
from kmt.infer import Translator

rows = [json.loads(l) for l in open("mqm_run/sample.jsonl", encoding="utf-8")]
tr = Translator("runs/base/final", "mbart")
hyp = tr.translate([r["src"] for r in rows], num_beams=5, n_best=5, alpha=0.0)
with open("mqm_run/s1_outputs.jsonl", "w", encoding="utf-8") as f:
    for r, h in zip(rows, hyp):
        f.write(json.dumps({"src_id": r["src_id"], "text": h}, ensure_ascii=False) + "\n")
print(f"wrote {len(hyp)} outputs")