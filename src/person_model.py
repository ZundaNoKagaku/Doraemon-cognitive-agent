"""第4層：人物知識 φ_j（相手を「知る」こと。相手を「大切にする」重み B とは独立）。

φ_j = (α_j, β_j): 人物 j が困りごとを起こす確率 p_j の Beta 事後分布。
     (μ_j, ν_j):   困りごとの深刻さの平均の Gaussian 推定（既知分散）。
更新: 毎日、知覚層が推定した「困っていたか・深刻さ」を観測として更新する（助けたかどうかに依存しない）。
出力: 予測確率 p̂_j = α/(α+β)、深刻さの予測 μ_j。意思決定の期待値 E[ΔW_j] に入る。
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PersonModel:
    alpha: float = 1.0
    beta: float = 1.0
    mu: float = 1.0
    prec: float = 1.0         # μ の精度
    obs_prec: float = 4.0     # 1回の深刻さ観測の精度

    def p_need(self):
        return self.alpha / (self.alpha + self.beta)

    def update(self, need_prob, severity=None, sev_prec=None):
        """need_prob: 知覚層が出す「困っていた」確率（確実な観測なら0/1）。期待値で Beta を更新。"""
        self.alpha += need_prob
        self.beta += 1 - need_prob
        if severity is not None and need_prob > 0.5:
            p = self.obs_prec if sev_prec is None else sev_prec
            self.mu = (self.prec * self.mu + p * severity) / (self.prec + p)
            self.prec += p

    def reset(self):
        self.__init__()
