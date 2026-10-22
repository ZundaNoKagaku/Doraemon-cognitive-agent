"""認知エージェント実験（方針B, 合成環境）。

E1 競合モデルの比較（任務あり／任務を250日目に除去）: B_A、長期選好、人物知識の精度、助けた割合
E2 人物理解と優先度の分離: φ の予測精度（Brier）と選好の関係
E3 正のフィードバックの安定性: パラメータ格子での解析的固定点と数値反復の一致、|f'(B*)|
E4 識別可能性: (η, ρ) を定常値だけから推定できるか、軌跡全体なら推定できるか
E5 知覚の改良が意思決定に寄与するか: 融合方式 full / nocc / vision_only × 矛盾率
注意: 合成環境での結果。ドラえもんや実在のロボットについての実証ではない。
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agent import EnvCfg, run_sim  # noqa: E402
from src.relationship_model import RelParams, steady_state, step_B  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "configs/agent.yaml"))
P = CFG["planner"]
REL = RelParams(**CFG["relationship"])


def mean_ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.min()), float(x.max())


def e1():
    out = {}
    T, D = CFG["mission_until"], CFG["days"]
    for model in CFG["models"]:
        for cond, mu in [("mission_kept", None), ("mission_removed", T)]:
            rows = []
            for s in CFG["seeds"]:
                h, ag = run_sim(s, model, D, P, REL, mission_until=mu)
                probe = dict(h["probe"])
                BA_T, BA_end = h["B"][T - 1, 0], h["B"][-1, 0]
                BC_end = h["B"][-1, 2]
                help_A = np.mean(np.array(h["help"][T:]) == 0)
                brier_A = np.mean(h["brier"][0][-100:]); brier_C = np.mean(h["brier"][2][-100:])
                rows.append(dict(BA_T=BA_T, BA_end=BA_end, BC_end=BC_end, pref_T=probe[T], pref_end=probe[D - D % 10 - 10],
                                 help_A_after=help_A, brier_A=brier_A, brier_C=brier_C,
                                 welfare=float(np.mean(h["welfare"])), persist=BA_end / BA_T if BA_T > 1e-9 else np.nan))
            out[f"{model}/{cond}"] = {k: mean_ci([r[k] for r in rows]) for k in rows[0]}
    return out


def e3():
    res = []
    for eta in CFG["sweep"]["eta"]:
        for rho in CFG["sweep"]["rho"]:
            for Bm in CFG["sweep"]["Bmax"]:
                for gain in CFG["sweep"]["gain"]:
                    p = RelParams(eta=eta, rho=rho, Bmax=Bm)
                    g0 = CFG["sweep"]["g0"]
                    Bs, fp = steady_state(p, gain, g0)
                    B = 0.0
                    traj = []
                    for _ in range(20000):
                        B = step_B(B, gain * B + g0, p)
                        traj.append(B)
                    tail = np.array(traj[-200:])
                    osc = float(tail.max() - tail.min())
                    linear_stable = eta * gain < rho
                    res.append(dict(eta=eta, rho=rho, Bmax=Bm, gain=gain, B_analytic=Bs, B_sim=float(tail.mean()),
                                    fprime=fp, osc=osc, linear_stable=linear_stable, in_bounds=bool(0 <= tail.min() and tail.max() <= Bm + 1e-9)))
    arr = res
    n_unstable = sum(1 for r in arr if abs(r["fprime"]) >= 1)
    n_bad = sum(1 for r in arr if abs(r["fprime"]) < 1 and abs(r["B_sim"] - r["B_analytic"]) > 1e-3)
    n_sat = sum(1 for r in arr if not r["linear_stable"])
    n_out = sum(1 for r in arr if not r["in_bounds"])
    return dict(n=len(arr), n_fprime_ge1=n_unstable, n_mismatch_stable=n_bad, n_linear_unstable_saturated=n_sat,
                n_out_of_bounds=n_out, max_osc=max(r["osc"] for r in arr),
                examples=[r for r in arr if abs(r["fprime"]) >= 1][:5], all=arr)


def e4():
    """真値 (η, ρ) = (0.05, 0.01) の開ループ軌跡（gain=0.3, g0=0.2）。観測は B に雑音。"""
    rng = np.random.default_rng(7)
    true = RelParams(eta=0.05, rho=0.01, Bmax=5.0)
    def traj(p, T=400):
        B, out = 0.0, []
        for _ in range(T):
            B = step_B(B, 0.3 * B + 0.2, p); out.append(B)
        return np.array(out)
    y = traj(true) + rng.normal(0, 0.05, 400)
    etas = np.geomspace(0.01, 0.5, 40); rhos = np.geomspace(0.002, 0.1, 40)
    sse_full = np.zeros((40, 40)); sse_ss = np.zeros((40, 40))
    for i, e in enumerate(etas):
        for k, r in enumerate(rhos):
            p = RelParams(eta=e, rho=r, Bmax=5.0)
            Bs, _ = steady_state(p, 0.3, 0.2)
            sse_ss[i, k] = (Bs - steady_state(true, 0.3, 0.2)[0]) ** 2
            sse_full[i, k] = np.sum((traj(p) - y) ** 2)
    near_ss = int(np.sum(sse_ss < 0.01))          # 定常値がほぼ同じになる格子点の数
    i, k = np.unravel_index(np.argmin(sse_full), sse_full.shape)
    ratio_ss = [float(etas[a] / rhos[b]) for a, b in zip(*np.where(sse_ss < 0.01))]
    return dict(grid=1600, n_same_steady_state=near_ss, ratio_range_same_ss=(min(ratio_ss), max(ratio_ss)),
                best_full=(float(etas[i]), float(rhos[k])), true=(0.05, 0.01))


def e5():
    out = {}
    for fusion in ["full", "nocc", "vision_only"]:
        for pc in [0.0, 0.3]:
            rows = []
            for s in CFG["seeds"]:
                env = EnvCfg(p_conflict=pc, p_missing=0.1)
                h, _ = run_sim(s, "feedback", 300, P, REL, env=env, fusion=fusion, log_B=True)
                rows.append(dict(welfare=float(np.mean(h["welfare"])), wrong=h["wrong_help"] / 300,
                                 BA=float(h["B"][-1, 0]), obs=h["obs"] / 300))
            out[f"{fusion}/conflict{pc}"] = {k: mean_ci([r[k] for r in rows]) for k in rows[0]}
    return out


def main():
    out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
    r1, r3, r4, r5 = e1(), e3(), e4(), e5()
    L = ["# 認知エージェント実験の結果（合成環境）", "", f"シード {CFG['seeds']}。値は平均（最小〜最大）。人物0 = 任務対象（A）、人物2 = 比較対象（C）。", "",
         "## E1・E2 競合モデルの比較", "",
         "| モデル / 条件 | B_A（250日） | B_A（400日） | B_C（400日） | 長期選好 P(A) 250日 | 長期選好 P(A) 400日 | 250日以降にAを助けた割合 | φ の誤差 A（Brier） | φ の誤差 C | 1日の幸福合計 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    f = lambda t: f"{t[0]:.2f} ({t[1]:.2f}–{t[2]:.2f})"
    for k, v in r1.items():
        L.append(f"| {k} | {f(v['BA_T'])} | {f(v['BA_end'])} | {f(v['BC_end'])} | {f(v['pref_T'])} | {f(v['pref_end'])} | {f(v['help_A_after'])} | {v['brier_A'][0]:.3f} | {v['brier_C'][0]:.3f} | {v['welfare'][0]:.3f} |")
    L += ["", "## E3 正のフィードバックの安定性（開ループ写像 B ← (1−ρ)B + η(1−B/Bmax)(gain·B + g0)）", "",
          f"格子 {r3['n']} 点（η×ρ×Bmax×gain）。|f'(B*)| ≥ 1（固定点が局所不安定）: {r3['n_fprime_ge1']} 点。"
          f"安定な点で数値反復と解析解が 1e-3 以上ずれた点: {r3['n_mismatch_stable']}。"
          f"線形近似では発散（η·gain ≥ ρ）だが飽和で有界になった点: {r3['n_linear_unstable_saturated']}。"
          f"B が [0, Bmax] の外に出た点: {r3['n_out_of_bounds']}。末尾200反復の最大振れ幅: {r3['max_osc']:.4f}。", ""]
    for ex in r3["examples"]:
        L.append(f"- 不安定例: η={ex['eta']}, ρ={ex['rho']}, Bmax={ex['Bmax']}, gain={ex['gain']} → f'={ex['fprime']:.3f}, 振れ幅 {ex['osc']:.3f}")
    L += ["", "## E4 識別可能性", "",
          f"真値 (η, ρ) = {r4['true']}。格子 {r4['grid']} 点のうち、定常値がほぼ同じ（二乗誤差 < 0.01）になる点: {r4['n_same_steady_state']}"
          f"（η/ρ の範囲 {r4['ratio_range_same_ss'][0]:.2f}〜{r4['ratio_range_same_ss'][1]:.2f}）。"
          f"軌跡全体（400日・雑音あり）での最小二乗推定: η={r4['best_full'][0]:.3f}, ρ={r4['best_full'][1]:.4f}。", "",
          "## E5 知覚の改良は意思決定に寄与するか（feedback モデル, 300日）", "",
          "| 融合 / 矛盾率 | 1日の幸福合計 | 困っていない人を助けた割合 | 追加観測の頻度 | B_A（300日） |", "| --- | --- | --- | --- | --- |"]
    for k, v in r5.items():
        L.append(f"| {k} | {f(v['welfare'])} | {f(v['wrong'])} | {f(v['obs'])} | {f(v['BA'])} |")
    (out_dir / "relationship_ablation.md").write_text("\n".join(L), encoding="utf-8")
    json.dump({"e1": r1, "e3": {k: v for k, v in r3.items() if k != "all"}, "e4": r4, "e5": r5},
              open(out_dir / "relationship_ablation.json", "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
