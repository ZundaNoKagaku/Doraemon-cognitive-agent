"""反実仮想実験（方針B, 合成環境）。

M1 記憶消去（250日目に人物Aのエピソード記憶を消す）: B を独立に保持するモデル（feedback）と
   記憶から再構成するモデル（memory）で、B_A・長期選好・φ がどう変わるか。φ も消す「全消去」も比較。
M2 短期感情のショック（250日目に worry_A を +3）: 長期の B_A が対照（同じシード・ショックなし）からどれだけ変わるか。
注意: 合成環境での結果。
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agent import run_sim  # noqa: E402
from src.relationship_model import RelParams  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "configs/agent.yaml"))
P = CFG["planner"]
REL = RelParams(**CFG["relationship"])
T, D = CFG["intervention_day"], CFG["days"]


def m1():
    out = {}
    for model in ["feedback", "memory", "exposure"]:
        for cond, kw in [("control", {}), ("erase_M", dict(erase_at=T)), ("erase_M_and_phi", dict(erase_at=T, erase_phi=True))]:
            rows = []
            for s in CFG["seeds"]:
                h, ag = run_sim(s, model, D, P, REL, **kw)
                pr = dict(h["probe"])
                rows.append(dict(B_before=h["B"][T - 1, 0], B_after=h["B"][T, 0], B_end=h["B"][-1, 0],
                                 pref_after=pr[T], pref_end=pr[D - 10], brierA_after=float(np.mean(h["brier"][0][T:T + 30]))))
            out[f"{model}/{cond}"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return out


def m2():
    out = {}
    for model in ["feedback", "exposure", "none"]:
        rows = []
        for s in CFG["seeds"]:
            hc, _ = run_sim(s, model, D, P, REL)
            hs, _ = run_sim(s, model, D, P, REL, shock_at=T, shock=3.0)
            help_c = np.mean(np.array(hc["help"][T:T + 5]) == 0)
            help_s = np.mean(np.array(hs["help"][T:T + 5]) == 0)
            def rel(a, b):
                return float((b - a) / a) if a > 1e-9 else 0.0
            rows.append(dict(help_A_5d_control=help_c, help_A_5d_shock=help_s,
                             dB_10d=rel(hc["B"][T + 10, 0], hs["B"][T + 10, 0]),
                             dB_end=rel(hc["B"][-1, 0], hs["B"][-1, 0])))
        out[model] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return out


def main():
    r1, r2 = m1(), m2()
    L = ["# 反実仮想実験の結果（合成環境）", "", f"シード {CFG['seeds']} の平均。介入日 {T}、終了 {D} 日。人物0 = A。", "",
         "## M1 記憶消去", "", "| モデル / 条件 | B_A 消去直前 | B_A 消去直後 | B_A 終了時 | 長期選好 P(A) 直後 | 長期選好 P(A) 終了時 | φ_A の誤差（直後30日） |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, v in r1.items():
        L.append(f"| {k} | {v['B_before']:.2f} | {v['B_after']:.2f} | {v['B_end']:.2f} | {v['pref_after']:.2f} | {v['pref_end']:.2f} | {v['brierA_after']:.3f} |")
    L += ["", "## M2 短期感情のショック（worry_A +3）", "", "| モデル | Aを助けた割合（直後5日、対照） | 同（ショック） | B_A の相対変化（10日後） | B_A の相対変化（終了時） |",
          "| --- | --- | --- | --- | --- |"]
    for k, v in r2.items():
        L.append(f"| {k} | {v['help_A_5d_control']:.2f} | {v['help_A_5d_shock']:.2f} | {v['dB_10d']:+.3f} | {v['dB_end']:+.3f} |")
    (ROOT / "results/memory_ablation.md").write_text("\n".join(L), encoding="utf-8")
    json.dump({"m1": r1, "m2": r2}, open(ROOT / "results/memory_ablation.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
