"""第2層：非同期な観測融合。

査読 C-1, C-2 への対応:
- 観測は「バースト」（1つのセンサーが1つの観測窓で出したトークン群）単位で扱う。
- 判断時刻 t の推定には、到着時刻 t_arrival <= t のバーストだけを使う（未来情報の遮断）。
- 到着済みバーストを取得時刻順に並べ、事前分布から再フィルタする（予測→更新の繰返し）。
  遅れて届いた観測も、取得時刻の位置で一度だけ使われる（同じ観測を毎時刻加算しない）。
- 信頼度（観測雑音 S）と共通原因確率 r を別の量として扱う:
    S = R_m / n + Σ_b,m  （トークン数 n が増えても、センサー共通の偏り Σ_b で情報量が頭打ち）
    品質フラグが「劣化」なら S に較正済みの膨張係数を掛ける
    r_i = P(z_i = 1 | 全観測)。z の全組合せを列挙して厳密に計算（n ≤ 10）
      z_i = 1: 同じ物体に由来（同時周辺尤度）
      z_i = 0: 別の物体・事象に由来 = 母集団の事前予測分布 L0_i
    フィルタ更新では r_i を各観測の重みとして使う（組合せ間の相関は無視する近似）
  他の感覚がない場合 L1 = L0 となり r = π（矛盾を検出できない＝珍しいが正しい観測を捨てない）。
"""
from dataclasses import dataclass

import numpy as np

from .world_model import GaussianState, gauss_logpdf


@dataclass
class Burst:
    obs_id: str
    modality: str
    t_capture: float
    t_arrival: float
    tokens: np.ndarray            # (n, p)
    degraded: bool = False        # 品質フラグ（センサー自身が報告する劣化）


@dataclass
class SensorModel:
    H: np.ndarray                 # (p, d)
    R: np.ndarray                 # (p, p) トークン1つあたりの雑音
    Sb: np.ndarray                # (p, p) センサー共通の偏り（同一バースト内で相関）
    pi_common: float = 0.9        # 共通原因の事前確率
    degrade_factor: float = 25.0  # 品質フラグ時の雑音膨張（訓練データで較正）


class AsyncFuser:
    def __init__(self, sensors, prior_mean, prior_cov, Q, use_bias=True, use_common_cause=True,
                 use_quality=True, cc_iters=3, max_enum=10):
        self.sensors, self.m0, self.P0, self.Q = sensors, prior_mean, prior_cov, Q
        self.use_bias, self.use_cc, self.use_quality, self.cc_iters = use_bias, use_common_cause, use_quality, cc_iters
        self.max_enum = max_enum

    # --- バーストを十分統計量（平均と共分散）に要約
    def summarize(self, b: Burst):
        sm = self.sensors[b.modality]
        n = len(b.tokens)
        y = b.tokens.mean(0)
        S = sm.R / n + (sm.Sb if self.use_bias else 0.0)
        if b.degraded and self.use_quality:
            S = S * sm.degrade_factor
        return y, sm.H, S

    def responsibilities(self, bursts):
        """共通原因確率 r_i = P(z_i = 1 | 到着済みの全観測) を、z の全組合せ（2^n 通り）の厳密な列挙で求める。
        z_i = 1 の観測はすべて同じ物体 x に由来する（同時周辺尤度 N(H_S m0, H_S P0 H_S^T + diag S)）、
        z_i = 0 の観測は別の物体に由来する（母集団の事前予測分布 L0_i、互いに独立）。
        旧版の leave-one-out 反復（同時更新）は2周期で振動する不具合があったため置き換えた（tests で検出）。
        n > max_enum のときは減衰つき逐次更新で近似する。"""
        n = len(bursts)
        pis = np.array([self.sensors[b.modality].pi_common for b in bursts], float)
        if not self.use_cc:
            return np.ones(n)
        if n == 0:
            return pis
        summ = [self.summarize(b) for b in bursts]
        l0 = np.array([gauss_logpdf(y, H @ self.m0, H @ self.P0 @ H.T + S) for (y, H, S) in summ])
        if n <= self.max_enum:
            logw, Z = [], []
            for mask in range(2 ** n):
                z = np.array([(mask >> i) & 1 for i in range(n)], bool)
                lw = np.sum(np.where(z, np.log(pis), np.log1p(-pis))) + l0[~z].sum()
                if z.any():
                    idx = np.where(z)[0]
                    y = np.concatenate([summ[i][0] for i in idx])
                    H = np.vstack([summ[i][1] for i in idx])
                    S = np.zeros((len(y), len(y))); o = 0
                    for i in idx:
                        k = len(summ[i][0]); S[o:o + k, o:o + k] = summ[i][2]; o += k
                    lw += gauss_logpdf(y, H @ self.m0, H @ self.P0 @ H.T + S)
                logw.append(lw); Z.append(z)
            logw = np.array(logw); w = np.exp(logw - logw.max()); w /= w.sum()
            return np.array(Z, float).T @ w
        # 近似（観測が多い場合）: 逐次（Gauss-Seidel）・減衰つきの leave-one-out 更新
        r = pis.copy()
        L0 = np.linalg.inv(self.P0)
        J = [H.T @ np.linalg.solve(S, H) for (y, H, S) in summ]
        h = [H.T @ np.linalg.solve(S, y) for (y, H, S) in summ]
        for _ in range(self.cc_iters * 10):
            for i, (y, H, S) in enumerate(summ):
                Lam = L0 + sum(r[k] * J[k] for k in range(n) if k != i)
                eta = L0 @ self.m0 + sum(r[k] * h[k] for k in range(n) if k != i)
                C = np.linalg.inv(Lam); mu = C @ eta
                l1 = gauss_logpdf(y, H @ mu, H @ C @ H.T + S)
                new = 1.0 / (1.0 + (1 - pis[i]) / pis[i] * np.exp(np.clip(l0[i] - l1, -50, 50)))
                r[i] = 0.5 * r[i] + 0.5 * new
        return r

    def estimate(self, bursts, t, t0=0.0):
        """判断時刻 t の事後状態。到着済みのバーストだけを取得時刻順に一度ずつ取り込む。"""
        arrived = sorted([b for b in bursts if b.t_arrival <= t], key=lambda b: (b.t_capture, b.obs_id))
        r = self.responsibilities(arrived)
        st = GaussianState(self.m0.copy(), self.P0.copy(), t0)
        for b, rb in zip(arrived, r):
            st.predict(max(b.t_capture, st.t), self.Q)
            y, H, S = self.summarize(b)
            st.update(y, H, S, r=rb, obs_id=b.obs_id)
        st.predict(max(t, st.t), self.Q)
        return st, dict(zip([b.obs_id for b in arrived], r))


def information_gain(state, H, S, target=0):
    """観測行動の情報利得（目標変数の分散の対数比）。線形ガウスでは観測値に依存せず事前に計算できる。"""
    P = state.cov
    Sy = H @ P @ H.T + S
    K = P @ H.T @ np.linalg.inv(Sy)
    P1 = (np.eye(len(P)) - K @ H) @ P
    return 0.5 * np.log(P[target, target] / P1[target, target])
