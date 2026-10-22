"""第3層：世界状態（線形ガウス状態空間モデル）。

数式との対応（研究計画書 §4.2）:
  状態遷移（予測）  x_{t+Δ} = F x_t + w,  w ~ N(0, Q Δ)             -> GaussianState.predict
  観測更新          y = H x + v,  v ~ N(0, S)                        -> GaussianState.update
  共通原因つき更新  z ~ Bernoulli(r): z=1 なら上式、z=0 なら x と無関係
                    事後分布は2成分混合をモーメント一致で1つのガウスに近似 -> update(..., r)
事後状態 (mean, cov) が次の predict の入力になる（再帰性）。
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GaussianState:
    mean: np.ndarray
    cov: np.ndarray
    t: float = 0.0
    log: list = field(default_factory=list)   # 更新に使った観測IDの記録（同一観測の二重加算検査用）

    def copy(self):
        return GaussianState(self.mean.copy(), self.cov.copy(), self.t, list(self.log))

    def predict(self, t_new, Q):
        """状態遷移: 時刻 t から t_new へ。物体特性はほぼ一定なので F = I、Q は小さい拡散。"""
        dt = t_new - self.t
        if dt < -1e-12:
            raise ValueError("predict は時間を逆行できない（遅着観測は AsyncFuser が再フィルタで扱う）")
        self.cov = self.cov + Q * max(dt, 0.0)
        self.t = t_new
        return self

    def update(self, y, H, S, r=1.0, obs_id=None):
        """観測更新。r は共通原因の事後確率（信頼度 S とは別の量）。"""
        if obs_id is not None:
            if obs_id in self.log:
                raise RuntimeError(f"観測 {obs_id} を二重に取り込もうとした")
            self.log.append(obs_id)
        m0, P0 = self.mean, self.cov
        Sy = H @ P0 @ H.T + S
        K = P0 @ H.T @ np.linalg.inv(Sy)
        m1 = m0 + K @ (y - H @ m0)
        P1 = (np.eye(len(m0)) - K @ H) @ P0
        m = r * m1 + (1 - r) * m0
        d1, d0 = (m1 - m)[:, None], (m0 - m)[:, None]
        self.mean = m
        self.cov = r * (P1 + d1 @ d1.T) + (1 - r) * (P0 + d0 @ d0.T)
        self.cov = 0.5 * (self.cov + self.cov.T)
        return self


def gauss_logpdf(y, mu, S):
    d = y - mu
    sign, logdet = np.linalg.slogdet(S)
    return -0.5 * (d @ np.linalg.solve(S, d) + logdet + len(y) * np.log(2 * np.pi))
