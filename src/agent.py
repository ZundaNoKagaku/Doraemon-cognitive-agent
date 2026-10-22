"""第8層：行動と経験のフィードバックを含む統合エージェントと、合成の社会環境。

1日の流れ（推論時の情報フロー）:
 1. 環境: 各人物 j が確率 p_j で困る（深刻さ sev_j）。任務に関係する出来事かどうか dF_j。
 2. 知覚（第1–3層）: 各人物について視覚・音の観測バースト（雑音・欠測・矛盾あり）→ AsyncFuser
    → 融合観測 (ŷ_j, V_j) → 人物知識 φ_j を事前分布として P(need_j)、sev_j の事後推定
 3. 情報収集（第7層）: V_info が費用を上回れば observe_j（高品質の追加観測）
 4. 目的実行（第7層）: help_j / rest を softmax で選ぶ（安全規則つき）
 5. 結果: 幸福変化 ΔW、相互的配慮 R（助けた相手が確率 recip_j で返す）
 6. 更新: φ（知覚した need で）、M（エピソード保存）、B（モデルごとの更新則）、E（短期感情）
"""
from dataclasses import dataclass, field

import numpy as np

from .affective_state import AffectiveState
from .async_fusion import AsyncFuser, Burst, SensorModel
from .episodic_memory import Episode, EpisodicMemory
from .person_model import PersonModel
from .planner import choose_goal, choose_info, preference_probe
from .relationship_model import RelParams, B_from_memory, input_g, step_B
from .world_model import gauss_logpdf


@dataclass
class EnvCfg:
    p_need: tuple = (0.5, 0.3, 0.3)
    recip: tuple = (0.3, 0.3, 0.3)
    p_mission_event: float = 0.4
    sev_mean: float = 1.0
    sev_sd: float = 0.3
    p_missing: float = 0.1
    p_conflict: float = 0.1
    obs_sd: float = 0.5


def make_perception(fusion="full"):
    """1次元（その人物の困りごとの信号）用の融合器。fusion: full / nocc / vision_only"""
    H = np.array([[1.0]])
    sensors = {"v": SensorModel(H, np.array([[0.25]]), np.array([[0.01]]), 0.9),
               "a": SensorModel(H, np.array([[0.25]]), np.array([[0.01]]), 0.9),
               "chk": SensorModel(H, np.array([[0.02]]), np.array([[0.0]]), 0.99)}
    if fusion == "vision_only":
        sensors = {"v": sensors["v"], "chk": sensors["chk"]}
    return AsyncFuser(sensors, np.zeros(1), np.eye(1) * 100.0, np.eye(1) * 1e-4,
                      use_common_cause=(fusion != "nocc"))


@dataclass
class Agent:
    n: int
    model: str = "feedback"            # none / feedback / exposure / recip_only / memory
    rel: RelParams = field(default_factory=RelParams)
    P: dict = None
    fusion: str = "full"

    def __post_init__(self):
        self.phi = [PersonModel() for _ in range(self.n)]
        self.M = EpisodicMemory()
        self.B = np.zeros(self.n)
        self.E = AffectiveState(self.n, kappa=self.P["kappa_E"])
        self.fz = make_perception(self.fusion)

    def perceive(self, bursts, j, day):
        """融合観測 → φ_j を事前とした need の事後確率と深刻さの推定。"""
        bs = [b for b in bursts if b.obs_id.startswith(f"{day}-{j}-")]
        if self.fusion == "vision_only":
            bs = [b for b in bs if b.modality != "a"]
        st, _ = self.fz.estimate(bs, t=1.0)
        y, V = st.mean[0], st.cov[0, 0]
        ph = self.phi[j]
        p = ph.p_need()
        if V > 50:   # 観測なし: 人物知識だけに頼る
            return p, ph.mu, V
        s2 = 1.0 / ph.prec + 0.3 ** 2
        l1 = np.log(p + 1e-9) + gauss_logpdf(np.array([y]), np.array([ph.mu]), np.array([[V + s2]]))
        l0 = np.log(1 - p + 1e-9) + gauss_logpdf(np.array([y]), np.array([0.0]), np.array([[V]]))
        post = 1 / (1 + np.exp(np.clip(l0 - l1, -50, 50)))
        sev = (ph.mu / s2 + y / V) / (1 / s2 + 1 / V)
        return float(post), float(sev), V


def env_bursts(rng, env, day, need, sev):
    out = []
    n = len(need)
    for j in range(n):
        for m in ("v", "a"):
            if rng.random() < env.p_missing:
                continue
            src = j
            if rng.random() < env.p_conflict:          # 別人物の様子を誤って捉えた観測（共通原因でない）
                src = (j + 1 + rng.integers(0, n - 1)) % n
            val = (sev[src] if need[src] else 0.0) + rng.normal(0, env.obs_sd)
            out.append(Burst(f"{day}-{j}-{m}", m, 0.1, 0.2, np.array([[val]])))
    return out


def run_sim(seed, model, days, P, rel=None, env=None, fusion="full", mission_until=None,
            erase_at=None, erase_phi=False, shock_at=None, shock=3.0, probe_every=10, log_B=True):
    rng = np.random.default_rng(seed)
    env = env or EnvCfg()
    rel = rel or RelParams()
    n = len(env.p_need)
    ag = Agent(n, model, rel, P, fusion)
    hist = {"B": [], "probe": [], "help": [], "welfare": [], "brier": [[] for _ in range(n)], "wrong_help": 0, "obs": 0}
    for day in range(days):
        m = np.zeros(n)
        if mission_until is None or day < mission_until:
            m[0] = 1.0
        if erase_at is not None and day == erase_at:
            ag.M.erase(0)
            if model == "memory":
                ag.B[0] = 0.0
            if erase_phi:
                ag.phi[0].reset()
        if shock_at is not None and day == shock_at:
            ag.E.shock(0, worry=shock)
        need = rng.random(n) < np.array(env.p_need)
        sev = np.clip(rng.normal(env.sev_mean, env.sev_sd, n), 0.2, None)
        dF = np.array([1.0 if (j == 0 and need[j] and rng.random() < env.p_mission_event) else 0.0 for j in range(n)])
        bursts = env_bursts(rng, env, day, need, sev)
        pn, sv = np.zeros(n), np.zeros(n)
        for j in range(n):
            hist["brier"][j].append((ag.phi[j].p_need() - need[j]) ** 2)
            pn[j], sv[j], _ = ag.perceive(bursts, j, day)
        # 情報収集段階
        jo, _ = choose_info(pn, sv, ag.B, m, dF, ag.E.worry, ag.E.frust, P)
        if jo is not None:
            hist["obs"] += 1
            chk = Burst(f"{day}-{jo}-chk", "chk", 0.5, 0.6,
                        np.array([[(sev[jo] if need[jo] else 0.0) + rng.normal(0, 0.15)]]))
            bursts.append(chk)
            pn[jo], sv[jo], _ = ag.perceive(bursts, jo, day)
        # 目的実行段階
        risk = np.zeros(n)
        jh, _ = choose_goal(pn, sv, ag.B, m, dF, ag.E.worry, ag.E.frust, risk, P, rng)
        dW = np.where(need, -sev, 0.0)
        care = np.zeros(n); recip = np.zeros(n)
        if jh is not None:
            if need[jh]:
                dW[jh] += sev[jh] * P["eff"]
                care[jh] = sev[jh]
                recip[jh] = float(rng.random() < env.recip[jh])
            else:
                hist["wrong_help"] += 1
        hist["welfare"].append(dW.sum())
        hist["help"].append(-1 if jh is None else jh)
        # 更新（φ, M, B, E）
        frust_in = np.zeros(n)
        for j in range(n):
            pe = abs(float(need[j]) - ag.phi[j].p_need())
            ag.phi[j].update(pn[j], sv[j])
            interacted = (jh == j) or (jo == j)
            q = pe if interacted else 0.0
            if interacted:
                ag.M.store(Episode(day, j, pn[j], jh == j, dW[j], recip[j], q + recip[j], care[j]))
            exposure = P["kappa"] * m[j]
            if model == "memory":
                ag.B[j] = B_from_memory(ag.M, j, day, rel)
            elif model != "none":
                ag.B[j] = step_B(ag.B[j], input_g(model, care[j], exposure, recip[j], q, rel), rel)
            if jh == j and not need[j]:
                frust_in[j] = 0.5
        ag.E.update((P["c0"] + ag.B) * pn * sv, frust_in)
        if log_B:
            hist["B"].append(ag.B.copy())
        if day % probe_every == 0:
            hist["probe"].append((day, preference_probe(ag.B, P, 0, 2)))
    hist["B"] = np.array(hist["B"])
    return hist, ag
