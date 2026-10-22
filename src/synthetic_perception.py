"""合成（仮）知覚データ。実センサーの記録ではない。

物体の潜在状態 x = [硬さ, 表面の粗さ]（標準化）。
v 視覚: 粗さに強く・硬さに弱く反応、トークン多数（12/バースト）、センサー共通の偏りが大きい
h 触覚: 硬さに強く反応、トークン8、偏り小
a 接触音: 硬さと粗さの両方、トークン1
各バーストのトークン = H x + β(バースト共通の偏り) + ε(トークンごとの雑音)
"""
import numpy as np

TRUE = {
    "v": dict(H=np.array([[0.2, 0.0], [0.0, 1.0]]), sd=0.6, sd_b=0.25, n=12, tc=0.10, lag=0.05),
    "h": dict(H=np.array([[1.0, 0.0], [0.0, 0.6]]), sd=0.4, sd_b=0.08, n=8, tc=0.50, lag=0.05),
    "a": dict(H=np.array([[0.7, 0.0], [0.0, 0.4]]), sd=0.5, sd_b=0.12, n=1, tc=0.50, lag=0.02),
}
CONDITIONS = ["clean", "conflict_h", "missing_h", "delayed_h", "degraded_v_flagged",
              "degraded_v_unflagged", "rare_correct", "all_degraded"]


def make_bursts(x, rng, cond="clean", x_other=None, oid="o"):
    from .async_fusion import Burst
    out = []
    for m, s in TRUE.items():
        if cond == "missing_h" and m == "h":
            continue
        src = x_other if (cond == "conflict_h" and m == "h") else x
        sd, degraded = s["sd"], False
        if (cond.startswith("degraded_v") and m == "v") or cond == "all_degraded":
            sd, degraded = s["sd"] * 5, True
        if cond == "degraded_v_unflagged" and m == "v":
            degraded = False
        beta = rng.normal(0, s["sd_b"], 2)
        tok = (s["H"] @ src + beta)[None] + rng.normal(0, sd, (s["n"], 2))
        arr = s["tc"] + s["lag"] + (1.0 if (cond == "delayed_h" and m == "h") else 0.0)
        out.append(Burst(f"{oid}-{m}", m, s["tc"], arr, tok, degraded))
    return out


def sample_object(rng, cond="clean"):
    x = rng.normal(0, 1, 2)
    if cond == "rare_correct":
        x[0] = rng.choice([-1, 1]) * rng.uniform(2.6, 3.2)   # 母集団では珍しい（2.6σ以上）が正しい観測
    return x
