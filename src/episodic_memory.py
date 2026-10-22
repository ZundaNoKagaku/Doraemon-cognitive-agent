"""第5層：エピソード記憶 M。

保存: (日, 人物, 困りごとの推定, 自分の行動, 結果の幸福変化, 相手の相互的配慮, 顕著性 q)
顕著性 q = |予測誤差| + 相互性（強い出来事ほど強く符号化。McGaugh 2004 の知見の比喩的な転用＝仮定）
想起の重み ω_k(t) = q_k (1 + (t - τ_k)/λ)^(-d)（べき乗型の忘却）
消去: erase(person) で人物 j の全エピソードを削除（反実仮想用）
"""
from dataclasses import dataclass, field


@dataclass
class Episode:
    day: int
    person: int
    need_est: float
    helped: bool
    dW: float
    reciprocity: float
    salience: float
    care: float


@dataclass
class EpisodicMemory:
    lam: float = 30.0
    d: float = 0.6
    capacity: int = 5000
    episodes: list = field(default_factory=list)

    def store(self, ep: Episode):
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity:            # 容量超過時は最も弱い記憶から消す
            self.episodes.sort(key=lambda e: e.salience)
            self.episodes.pop(0)

    def weight(self, ep, day):
        return ep.salience * (1 + (day - ep.day) / self.lam) ** (-self.d)

    def retrieve(self, person, day, k=None):
        eps = [(self.weight(e, day), e) for e in self.episodes if e.person == person]
        eps.sort(key=lambda x: -x[0])
        return eps if k is None else eps[:k]

    def erase(self, person=None):
        self.episodes = [e for e in self.episodes if person is not None and e.person != person]
