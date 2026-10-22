"""第6層（後半）：短期の感情評価 E_j（関係性重み B とは時間尺度も役割も別の変数）。

worry_j  = (c0 + B_j) · p̂_need_j · μ_sev_j     … 予期される損失（心配）。保護的な「様子を見る・助ける」行動を押し上げる
frust_j  = 助けたのに結果が悪かったときの予測誤差（苛立ち）。助ける行動を一時的に押し下げる
更新: E ← (1-κ_E) E + 入力。κ_E ≫ ρ（日単位で減衰）。外部から E だけを操作できる（反実仮想用）。
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AffectiveState:
    n: int
    kappa: float = 0.5
    worry: np.ndarray = field(default=None)
    frust: np.ndarray = field(default=None)

    def __post_init__(self):
        self.worry = np.zeros(self.n) if self.worry is None else self.worry
        self.frust = np.zeros(self.n) if self.frust is None else self.frust

    def update(self, j_expected_loss, frust_input):
        self.worry = (1 - self.kappa) * self.worry + self.kappa * j_expected_loss
        self.frust = (1 - self.kappa) * self.frust + frust_input

    def shock(self, j, worry=0.0, frust=0.0):
        self.worry[j] += worry
        self.frust[j] += frust
