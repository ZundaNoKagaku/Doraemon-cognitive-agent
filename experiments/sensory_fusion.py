"""知覚実験（方針A, 合成データ）: H1/H2/H5 と査読 C-1〜C-4 の検証。

1) 融合方式の比較（8条件 × 9手法 × 5シード）
2) 化学センサーの応答遅延: 固定遅延法 vs 動的応答モデル（湿度で応答が変わる）
3) 能動的知覚: 情報利得のみ vs 安全制約つき
注意: 合成データでの結果。実センサー性能ではない。
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.async_fusion import AsyncFuser, SensorModel, information_gain  # noqa: E402
from src.synthetic_perception import CONDITIONS, TRUE, make_bursts, sample_object  # noqa: E402
from src.world_model import GaussianState  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "configs/perception.yaml"))
Z90 = 1.6449


# ---------------------------------------------------------------- 較正（訓練物体のみ）
def calibrate(rng, n_obj):
    """訓練物体から H, R, Σ_b, 劣化係数を推定する（真値 TRUE は使わない）。"""
    sensors = {}
    data = {m: [] for m in TRUE}
    deg = {m: [] for m in TRUE}
    for i in range(n_obj):
        x = sample_object(rng)
        for b in make_bursts(x, rng, "clean", oid=f"tr{i}"):
            data[b.modality].append((x, b.tokens))
        for b in make_bursts(x, rng, "all_degraded", oid=f"trd{i}"):
            deg[b.modality].append(b.tokens)
    for m, rows in data.items():
        X = np.array([r[0] for r in rows])
        Y = np.array([r[1].mean(0) for r in rows])
        H = np.linalg.lstsq(X, Y, rcond=None)[0].T
        within = np.mean([np.var(r[1], 0, ddof=1) for r in rows if len(r[1]) > 1], 0) if len(rows[0][1]) > 1 else None
        resid = Y - X @ H.T
        n = len(rows[0][1])
        tot = np.var(resid, 0, ddof=2)
        if within is None:            # トークン1つのセンサーは偏りと雑音を分離できない → 合計を R とし Σ_b=0
            R, Sb = np.diag(tot), np.zeros((2, 2))
        else:
            R = np.diag(within)
            Sb = np.diag(np.clip(tot - within / n, 1e-4, None))
        dw = np.mean([np.var(t, 0, ddof=1) for t in deg[m]], 0) if n > 1 else None
        factor = float(np.mean(dw / within)) if dw is not None else 25.0
        sensors[m] = SensorModel(H=H, R=R, Sb=Sb, pi_common=CFG["pi_common"], degrade_factor=factor)
    return sensors


def fit_linear(rng, n_obj, sensors):
    """late fusion（固定重み）と concat（特徴連結の線形回帰）を訓練物体で学習。"""
    rows = []
    for i in range(n_obj):
        x = sample_object(rng)
        cond = rng.choice(["clean", "missing_h"], p=[0.8, 0.2])
        bs = {b.modality: b for b in make_bursts(x, rng, cond, oid=f"lf{i}")}
        rows.append((x, bs))
    # concat: [各感覚のトークン平均(2) + 欠測フラグ(1)] × 3 + 定数
    def feat(bs, t):
        f = []
        for m in TRUE:
            b = bs.get(m)
            if b is not None and b.t_arrival <= t:
                f += list(b.tokens.mean(0)) + [0.0]
            else:
                f += [0.0, 0.0, 1.0]
        return np.array(f + [1.0])
    F = np.array([feat(bs, 9e9) for _, bs in rows])
    Y = np.array([x[0] for x, _ in rows])
    w = np.linalg.solve(F.T @ F + 1e-2 * np.eye(F.shape[1]), F.T @ Y)
    sd_concat = float(np.std(Y - F @ w))
    # late fusion: 各感覚単独の推定を、訓練で決めた固定重みで平均（欠測時は残りで正規化）
    singles = {m: AsyncFuser({m: sensors[m]}, np.zeros(2), np.eye(2), Q_of(), use_common_cause=False) for m in TRUE}
    est = {m: [] for m in TRUE}
    for x, bs in rows:
        for m in TRUE:
            if m in bs:
                est[m].append((singles[m].estimate([bs[m]], 9e9)[0].mean[0], x[0]))
    mse = {m: np.mean([(e - t) ** 2 for e, t in est[m]]) for m in TRUE}
    wl = {m: 1 / mse[m] for m in TRUE}
    return dict(concat_w=w, concat_sd=sd_concat, feat=feat, late_w=wl, late_mse=mse, singles=singles)


def Q_of():
    return np.eye(2) * CFG["Q"]


def run_methods(bs, t, sensors, lin):
    out = {}
    for m in TRUE:
        st, _ = lin["singles"][m].estimate([b for b in bs if b.modality == m], t)
        out[f"single_{m}"] = (st.mean[0], np.sqrt(st.cov[0, 0]), None)
    # late fusion
    num, den, var = 0.0, 0.0, 0.0
    present = [m for m in TRUE if any(b.modality == m and b.t_arrival <= t for b in bs)]
    if present:
        for m in present:
            num += lin["late_w"][m] * out[f"single_{m}"][0]
            den += lin["late_w"][m]
        out["late"] = (num / den, np.sqrt(1 / den), None)
    else:
        out["late"] = (0.0, 1.0, None)
    bsd = {b.modality: b for b in bs}
    f = lin["feat"](bsd, t)
    out["concat"] = (float(f @ lin["concat_w"]), lin["concat_sd"], None)
    variants = {"async_nobias": dict(use_bias=False), "async_nocc": dict(use_common_cause=False),
                "async_noquality": dict(use_quality=False), "full": {}}
    for name, kw in variants.items():
        fz = AsyncFuser(sensors, np.zeros(2), np.eye(2), Q_of(), **kw)
        st, r = fz.estimate(bs, t)
        out[name] = (st.mean[0], np.sqrt(st.cov[0, 0]), r)
    return out


def fusion_experiment(seed):
    rng = np.random.default_rng(seed)
    sensors = calibrate(rng, CFG["n_train"])
    lin = fit_linear(rng, CFG["n_train"], sensors)
    res = {}
    for cond in CONDITIONS:
        rows = {}
        rvals = {"h_true": [], "h_conflict": []}
        for i in range(CFG["n_test"]):
            x = sample_object(rng, cond)
            xo = sample_object(rng)
            bs = make_bursts(x, rng, cond, x_other=xo, oid=f"te{i}")
            out = run_methods(bs, CFG["t_decision"], sensors, lin)
            for k, (mu, sd, r) in out.items():
                rows.setdefault(k, []).append((mu, sd, x[0]))
            r = out["full"][2]
            key = f"te{i}-h"
            if key in r:
                rvals["h_conflict" if cond == "conflict_h" else "h_true"].append(r[key])
        res[cond] = {}
        for k, v in rows.items():
            mu, sd, y = map(np.array, zip(*v))
            err = np.abs(mu - y)
            nll = 0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * ((mu - y) / sd) ** 2
            cover = np.mean(err <= Z90 * sd)
            keep = sd <= CFG["abstain_sd"]
            res[cond][k] = dict(mae=float(err.mean()), nll=float(nll.mean()), cover90=float(cover),
                                abstain=float(1 - keep.mean()),
                                mae_answered=float(err[keep].mean()) if keep.any() else float("nan"),
                                err=err.tolist())
        res[cond]["_r_h"] = {k: float(np.mean(v)) if v else None for k, v in rvals.items()}
    return res


# ---------------------------------------------------------------- 化学センサーの応答遅延（C-3）
def chem_response(t, t_exp, s, d, tau):
    return s * (1 - np.exp(-np.clip(t - t_exp - d, 0, None) / tau))


def chem_experiment(seed):
    """湿度が高いほど立ち上がりが遅い（d, τ が増える）合成センサー。判断時刻 T で s を推定する。"""
    rng = np.random.default_rng(1000 + seed)
    ts = np.arange(0, 20.0, 0.5)
    T = CFG["chem"]["t_decision"]
    out = {}
    for hum_name, (lo, hi) in {"low_humidity": (0.0, 0.3), "high_humidity": (0.7, 1.0)}.items():
        e_fixed, e_dyn, e_dyn_sd = [], [], []
        for _ in range(CFG["chem"]["n"]):
            hum = rng.uniform(lo, hi)
            s = rng.uniform(0.5, 2.0)
            d = 1.0 + 3.0 * hum + rng.normal(0, 0.2)
            tau = 1.5 * (1 + 2.5 * hum) * rng.lognormal(0, 0.15)
            y = chem_response(ts, 0.0, s, d, tau) + rng.normal(0, 0.05, len(ts))
            m = ts <= T
            # 固定遅延法: 低湿度の訓練で決めた「平坦になる時刻」D0 以降の平均を s とみなす
            D0 = CFG["chem"]["fixed_delay"]
            sel = m & (ts >= D0)
            e_fixed.append(abs((y[sel].mean() if sel.any() else 0.0) - s))
            # 動的応答モデル: (s, d, τ) を格子上の最小二乗で推定。s は線形なので解析的に解く
            best = (np.inf, None)
            for dd in np.linspace(0.5, 5.0, 19):
                for tt in np.linspace(0.8, 8.0, 25):
                    g = chem_response(ts[m], 0.0, 1.0, dd, tt)
                    if g @ g < 1e-6:
                        continue
                    sh = (g @ y[m]) / (g @ g)
                    sse = np.sum((y[m] - sh * g) ** 2)
                    if sse < best[0]:
                        best = (sse, sh, g)
            e_dyn.append(abs(best[1] - s))
        out[hum_name] = dict(mae_fixed=float(np.mean(e_fixed)), mae_dynamic=float(np.mean(e_dyn)))
    return out


# ---------------------------------------------------------------- 能動的知覚（C-4）
def active_experiment(seed):
    """視覚だけ観測した後に、追加の観測行動を1つ選ぶ。
    行動: press(触覚, 情報大, 壊れやすい物体に危険), tap(接触音, 中), look(視覚, 小), stop(打ち切り)
    目的: 硬さの不確実性（事後分散の対数比）を減らす − 時間コスト。安全制約: P(壊れやすい) > p_max なら press 禁止。"""
    rng = np.random.default_rng(2000 + seed)
    sensors = calibrate(rng, CFG["n_train"])
    fz = AsyncFuser(sensors, np.zeros(2), np.eye(2), Q_of())
    A = CFG["active"]
    actions = {"press": ("h", A["cost_press"]), "tap": ("a", A["cost_tap"]), "look": ("v", A["cost_look"])}
    res = {"ig_only": dict(viol=0, err=[], chosen=[]), "constrained": dict(viol=0, err=[], chosen=[])}
    for i in range(CFG["n_test"]):
        x = sample_object(rng)
        fragile = rng.random() < 0.2
        # 壊れやすさの手がかり（視覚から得る確率的な推定、較正済みと仮定）
        p_frag = np.clip((0.75 if fragile else 0.08) + rng.normal(0, 0.05), 0, 1)
        bs = [b for b in make_bursts(x, rng, "clean", oid=f"ac{i}") if b.modality == "v"]
        st, _ = fz.estimate(bs, 1.0)
        for policy in res:
            best, best_u = "stop", 0.0
            for a, (m, cost) in actions.items():
                sm = sensors[m]
                S = sm.R / TRUE[m]["n"] + sm.Sb
                u = information_gain(st, sm.H, S) - cost
                if policy == "constrained" and a == "press" and p_frag > A["p_fragile_max"]:
                    continue                       # 情報利得が高くても危険な行動は候補から外す（トレードしない）
                if u > best_u:
                    best, best_u = a, u
            res[policy]["chosen"].append(best)
            if best == "press" and fragile:
                res[policy]["viol"] += 1
            extra = []
            if best != "stop":
                m = actions[best][0]
                extra = [b for b in make_bursts(x, rng, "clean", oid=f"ac{i}{policy}") if b.modality == m]
                for b in extra:
                    b.obs_id += "x"; b.t_capture = 1.1; b.t_arrival = 1.2
            st2, _ = fz.estimate(bs + extra, 2.0)
            res[policy]["err"].append(abs(st2.mean[0] - x[0]))
    return {k: dict(violations=v["viol"], mae=float(np.mean(v["err"])),
                    share={a: v["chosen"].count(a) / len(v["chosen"]) for a in ["press", "tap", "look", "stop"]})
            for k, v in res.items()}


def boot_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = (b[idx] - a[idx]).mean(1)
    return float((b - a).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
    seeds = CFG["seeds"]
    fus = [fusion_experiment(s) for s in seeds]
    chem = [chem_experiment(s) for s in seeds]
    act = [active_experiment(s) for s in seeds]
    methods = [k for k in fus[0]["clean"] if not k.startswith("_")]
    lines = ["# 知覚実験の結果（合成データ）", "",
             f"シード {seeds}、各条件 テスト物体 {CFG['n_test']} 個/シード。判断時刻 t = {CFG['t_decision']}。",
             "値は5シードの平均（括弧内はシード間の最小〜最大）。目標は硬さ（標準化尺度）。", ""]
    summary = {}
    for cond in CONDITIONS:
        lines += [f"## {cond}", "", "| 手法 | MAE | NLL | 90%区間の被覆率 | 判断保留率 | 回答時MAE |", "| --- | --- | --- | --- | --- | --- |"]
        summary[cond] = {}
        for m in methods:
            vals = {k: [f[cond][m][k] for f in fus] for k in ["mae", "nll", "cover90", "abstain", "mae_answered"]}
            summary[cond][m] = {k: float(np.nanmean(v)) for k, v in vals.items()}
            lines.append(f"| {m} | {np.mean(vals['mae']):.3f} ({min(vals['mae']):.3f}–{max(vals['mae']):.3f}) | "
                         f"{np.mean(vals['nll']):.2f} | {np.mean(vals['cover90']):.2f} | {np.nanmean(vals['abstain']):.2f} | "
                         f"{np.nanmean(vals['mae_answered']):.3f} |")
        a = sum((f[cond]["concat"]["err"] for f in fus), [])
        b = sum((f[cond]["full"]["err"] for f in fus), [])
        d, lo, hi = boot_diff(a, b)
        summary[cond]["full_minus_concat"] = (d, lo, hi)
        a2 = sum((f[cond]["async_nocc"]["err"] for f in fus), [])
        d2, lo2, hi2 = boot_diff(a2, b)
        summary[cond]["full_minus_nocc"] = (d2, lo2, hi2)
        rh = [f[cond]["_r_h"] for f in fus]
        lines += ["", f"- 主要比較 full − concat のMAE差: {d:+.3f}（95%区間 {lo:+.3f}〜{hi:+.3f}）",
                  f"- 共通原因の効果 full − async_nocc: {d2:+.3f}（{lo2:+.3f}〜{hi2:+.3f}）",
                  f"- 触覚観測の共通原因確率 r の平均: 正しい観測 {np.mean([x['h_true'] for x in rh if x['h_true'] is not None]) if any(x['h_true'] is not None for x in rh) else float('nan'):.2f}"
                  f" / 別物体の観測 {np.mean([x['h_conflict'] for x in rh if x['h_conflict'] is not None]) if any(x['h_conflict'] is not None for x in rh) else float('nan'):.2f}", ""]
    lines += ["## 化学センサーの応答遅延（固定遅延法 vs 動的応答モデル）", "", "| 湿度 | 固定遅延法 MAE | 動的応答モデル MAE |", "| --- | --- | --- |"]
    for h in ["low_humidity", "high_humidity"]:
        lines.append(f"| {h} | {np.mean([c[h]['mae_fixed'] for c in chem]):.3f} | {np.mean([c[h]['mae_dynamic'] for c in chem]):.3f} |")
    lines += ["", "## 能動的知覚（1回の追加観測）", "", "| 方策 | 安全違反（壊れやすい物体を押した回数／5シード合計） | 追加観測後の MAE | 行動の割合 |", "| --- | --- | --- | --- |"]
    for p in ["ig_only", "constrained"]:
        lines.append(f"| {p} | {sum(a[p]['violations'] for a in act)} | {np.mean([a[p]['mae'] for a in act]):.3f} | "
                     + ", ".join(f"{k} {np.mean([a[p]['share'][k] for a in act]):.2f}" for k in ["press", "tap", "look", "stop"]) + " |")
    (out_dir / "sensory_fusion.md").write_text("\n".join(lines), encoding="utf-8")
    json.dump({"summary": summary, "chem": chem, "active": act}, open(out_dir / "sensory_fusion.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
