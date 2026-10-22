"""第6層（前半）：人物別の関係性重み B_j の更新則（競合モデル）。

すべて「仮定」であり、どれが正しいかは実験で比較する（査読 B-3, B-4, C-5）。

R0 none        : B_j = 0（任務と一般的配慮 c0 だけで行動）
R1 feedback    : B ← (1-ρ)B + η(1 - B/Bmax)(a_c C + a_r R + a_s q)
                 C = 実際に自分が行った世話（方策が B に依存 → 正のフィードバック）
R2 exposure    : 同じ式だが C の代わりに「割り当てられた接触量」E_exp（行動に依存しない → フィードバックなし）
R3 recip_only  : g = a_r R + a_s q（世話の投資を入力にしない。相互性と共有経験だけ）
R4 memory      : B_j = λ_B Σ_k ω_k(t) g_k を毎日エピソード記憶から再構成（記憶を消すと B も消える）
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RelParams:
    eta: float = 0.05
    rho: float = 0.01
    Bmax: float = 5.0
    a_c: float = 1.0
    a_r: float = 0.8
    a_s: float = 0.5
    lam_B: float = 0.05


def step_B(B, g, p: RelParams):
    """離散時間の更新。飽和項だけでは η·g > Bmax のとき1歩で Bmax を超える（tests で検出）ため、
    [0, Bmax] に切り詰める。本研究の設定（η = 0.05, g ≲ 3）では切り詰めは作動しない。"""
    return float(np.clip((1 - p.rho) * B + p.eta * (1 - B / p.Bmax) * g, 0.0, p.Bmax))


def input_g(model, care, exposure, recip, salience, p: RelParams):
    if model == "feedback":
        return p.a_c * care + p.a_r * recip + p.a_s * salience
    if model == "exposure":
        return p.a_c * exposure + p.a_r * recip + p.a_s * salience
    if model == "recip_only":
        return p.a_r * recip + p.a_s * salience
    return 0.0


def B_from_memory(memory, person, day, p: RelParams):
    eps = memory.retrieve(person, day)
    return min(p.Bmax, p.lam_B * sum(w * (p.a_c * e.care + p.a_r * e.reciprocity + p.a_s * e.salience) for w, e in eps))


def steady_state(p: RelParams, gain, g0):
    """開ループ入力 g = gain·B + g0（世話が B に比例する部分 gain と一定部分 g0）の固定点。
    ρB = η(1 - B/Bmax)(gain·B + g0) の非負根と、写像の微分 f'(B*) を返す（|f'|<1 なら局所安定）。"""
    a = -p.eta * gain / p.Bmax
    b = p.eta * gain - p.eta * g0 / p.Bmax - p.rho
    c = p.eta * g0
    if abs(a) < 1e-12:
        B = -c / b if b != 0 else np.inf
    else:
        disc = b * b - 4 * a * c
        roots = [(-b + s * np.sqrt(disc)) / (2 * a) for s in (1, -1)] if disc >= 0 else []
        roots = [r for r in roots if r >= -1e-9]
        B = min(roots) if roots else np.nan
    fprime = (1 - p.rho) + p.eta * ((-1 / p.Bmax) * (gain * B + g0) + (1 - B / p.Bmax) * gain)
    return float(B), float(fprime)
