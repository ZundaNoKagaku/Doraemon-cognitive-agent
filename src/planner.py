"""第7層：意思決定。

1日を「情報収集段階」と「目的実行段階」に分ける（査読 C-4: 情報収集行動と目的行動の区別）。

目的実行段階（help_j か rest）:
  Q(help_j) = κ m_j ΔF_j                     任務項（任務に関係する出来事のときだけ ΔF_j = 1）
            + (c0 + B_j) · E[ΔW_j]           関係による価値。E[ΔW_j] = P(need_j)·sev_j·eff
            + β_w worry_j − β_f frust_j      短期の感情による偏り
            − cost_help
  Q(rest)   = 0
  π(a) = softmax(Q/τ)。安全規則で禁止された行動（risk_j > risk_max）は候補から外す（価値と交換しない）。

情報収集段階（observe_j か skip）:
  V_info(j) = E_{need_j}[max_a Q(a | need_j 確定)] − max_a Q(a | 現在の信念)   … 意思決定にとっての情報価値
  observe_j を選ぶのは V_info(j) − cost_obs > 0 の最大の j（観測で need_j が確定すると近似＝仮定）
"""
import numpy as np


def softmax(q, tau):
    z = (q - q.max()) / tau
    e = np.exp(z)
    return e / e.sum()


def help_values(p_need, sev, B, m, dF, worry, frust, P):
    EdW = p_need * sev * P["eff"]
    return P["kappa"] * m * dF + (P["c0"] + B) * EdW + P["beta_w"] * worry - P["beta_f"] * frust - P["cost_help"]


def value_of_info(p_need, sev, B, m, dF, worry, frust, P):
    n = len(p_need)
    best_now = max(0.0, help_values(p_need, sev, B, m, dF, worry, frust, P).max())
    voi = np.zeros(n)
    for j in range(n):
        e = np.arange(n) == j
        v1 = max(0.0, help_values(np.where(e, 1.0, p_need), sev, B, m, dF, worry, frust, P).max())
        v0 = max(0.0, help_values(np.where(e, 0.0, p_need), sev, B, m, dF, worry, frust, P).max())
        voi[j] = p_need[j] * v1 + (1 - p_need[j]) * v0 - best_now
    return voi


def choose_info(p_need, sev, B, m, dF, worry, frust, P):
    voi = value_of_info(p_need, sev, B, m, dF, worry, frust, P) - P["cost_obs"]
    j = int(np.argmax(voi))
    return (j if voi[j] > 0 else None), voi


def choose_goal(p_need, sev, B, m, dF, worry, frust, risk, P, rng):
    qh = help_values(p_need, sev, B, m, dF, worry, frust, P)
    allowed = [j for j in range(len(qh)) if risk[j] <= P["risk_max"]]
    q = np.array([qh[j] for j in allowed] + [0.0])
    pi = softmax(q, P["tau"])
    k = rng.choice(len(q), p=pi)
    return (allowed[k] if k < len(allowed) else None), pi


def preference_probe(B, P, i=0, k=2):
    """長期の選好の測定: 人物 i と k が同じ深刻さで困っており、任務に関係しない場面（ΔF=0）、
    感情項を 0 にしたときに i を選ぶ確率。人物知識の精度とは独立に測る。"""
    n = len(B)
    pn, sev = np.zeros(n), np.zeros(n)
    pn[[i, k]] = 1.0; sev[[i, k]] = 1.0
    qh = help_values(pn, sev, B, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), P)
    pi = softmax(np.array([qh[i], qh[k]]), P["tau"])
    return float(pi[0])
