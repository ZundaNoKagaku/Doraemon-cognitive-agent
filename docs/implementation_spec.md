# 実装仕様書（成果物C）

対象: 研究計画書改訂版の付属コード `doraemon_cognitive_agent/`。本書だけで実装を再現できるように書く。
記号は研究計画書 §4・§5 と同じ。**合成環境の実装であり、実データの処理系ではない。**

## 0. 全体の推論の流れ

```
[方針A：知覚, 1物体ごと]
 センサー m ∈ {v, h, a} がバースト b を出す（取得時刻 t_c, 到着時刻 t_a, トークン n×p, 品質フラグ）
   → 判断時刻 t で、t_a ≤ t のバーストだけを選ぶ                         … 未来情報の遮断
   → 十分統計量 (ȳ_b, H_m, S_b) に要約                                      … 信頼度
   → 共通原因確率 r_b を z の全組合せの列挙で計算                            … 信頼度とは別の量
   → 事前 (m0, P0) から取得時刻順に predict → update(r_b) を一度ずつ           … 遅着も取得時刻の位置で1回
   → t まで predict → 事後 (m, P)。sd > abstain_sd なら判断保留
[方針B：エージェント, 1日ごと]
 環境: 困りごと need_j, 深刻さ sev_j, 任務事象 ΔF_j
   → 知覚: 人物ごとに上の融合（1次元）→ φ_j を事前とした P(need_j), 深刻さ推定
   → 情報収集段階: V_info(j) − cost_obs > 0 の最大の j を observe（高精度の追加観測）
   → 目的実行段階: softmax(Q/τ) で help_j / rest。risk_j > risk_max の行動は候補から除外
   → 結果: ΔW, 世話 C_j, 相互性 R_j
   → 更新: φ_j（全員）, M（関わった相手）, B_j（モデル別）, E（全員）
```

## 1. モジュールの入出力・状態変数

| モジュール | 入力 | 出力 | 状態 |
| --- | --- | --- | --- |
| `world_model.GaussianState` | 観測 y, H, S, r, obs_id | 更新後の (mean, cov) | mean, cov, t, log（取り込んだ obs_id） |
| `async_fusion.AsyncFuser.estimate` | バースト列, 判断時刻 t | GaussianState, {obs_id: r} | なし（毎回事前から再フィルタ＝純関数） |
| `async_fusion.information_gain` | 状態, H, S | 0.5·log(P_tt / P'_tt) | なし |
| `person_model.PersonModel` | P(need), 深刻さ | p̂_need, μ | α, β, μ, prec |
| `episodic_memory.EpisodicMemory` | Episode | 想起 [(ω, e)] | episodes |
| `relationship_model` | B, g, RelParams | 次の B | なし（B は Agent が持つ） |
| `affective_state.AffectiveState` | 予期損失, 苛立ち入力 | worry, frust | worry_j, frust_j |
| `planner` | P(need), sev, B, m, ΔF, E, risk | 観測先 j / 行動 j | なし |
| `agent.run_sim` | seed, モデル名, 介入 | 履歴（B, probe, help, welfare, brier, …） | Agent（φ, M, B, E, 融合器） |

## 2. 更新則

### 2.1 状態遷移と観測更新（`world_model.py`）

- 予測: P ← P + Q·Δt（F = I、物体特性はほぼ一定）。Δt < 0 は例外（時間は逆行しない）。
- 更新: K = P Hᵀ (H P Hᵀ + S)⁻¹、m₁ = m + K(y − Hm)、P₁ = (I − KH)P。
- 共通原因つき: 事後は r·N(m₁, P₁) + (1−r)·N(m, P) の混合 → モーメント一致で1つのガウスに近似。
- 同じ obs_id の2回目の update は例外。

### 2.2 バーストの要約（`AsyncFuser.summarize`）

ȳ = トークン平均、S = R_m / n + Σ_b,m（同一バースト内で共通の偏り Σ_b があるので、n → ∞ でも S ≥ Σ_b）。
品質フラグが立っていれば S ← S·c_m（c_m は訓練データの劣化時/通常時のトークン内分散比で較正）。

### 2.3 共通原因確率（`AsyncFuser.responsibilities`）

z_i ∈ {0,1}。p(z) = Π π_i^{z_i}(1−π_i)^{1−z_i}。
- z_i = 1 の集合 Sₒ: 同じ x に由来。同時周辺尤度 N(y_S; H_S m0, H_S P0 H_Sᵀ + blkdiag(S_i))。
- z_i = 0: 別の物体に由来。L0_i = N(y_i; H_i m0, H_i P0 H_iᵀ + S_i)（母集団の事前予測分布）。
- r_i = Σ_z z_i p(z | y)。n ≤ 10 は 2ⁿ 通りを厳密列挙。n > 10 は減衰つき逐次更新（近似）。
- 性質: 観測が1つなら r = π（矛盾は検出できないので、珍しいが正しい観測を捨てない）。
- 近似: フィルタでは r_i を各観測の重みとして独立に使う（組合せ間の相関を無視）。

### 2.4 人物知識 φ（`person_model.py`）

α ← α + P̂(need)、β ← β + 1 − P̂(need)（期待値更新）。P̂ > 0.5 のとき深刻さを精度加重で更新。
助けたかどうかに依存しない（「知ること」は「大切にすること」と独立に更新される）。

### 2.5 エピソード記憶 M（`episodic_memory.py`）

保存: (日, 人物, P̂, 助けたか, ΔW, 相互性 R, 顕著性 q = 予測誤差 + R, 世話 C)。関わった相手だけ。
想起の重み ω = q·(1 + Δt/λ)^(−d)、λ = 30, d = 0.6。erase(j) で人物 j を全消去。

### 2.6 関係性重み B（`relationship_model.py`）

B ← clip((1−ρ)B + η(1 − B/B_max)·g, 0, B_max)、η = 0.05, ρ = 0.01, B_max = 5。

| モデル | 入力 g | 意味 |
| --- | --- | --- |
| none | 0 | B = 0。任務と一般的配慮 c0 だけ |
| exposure | a_c·κ·m_j + a_r R + a_s q | 割り当てられた接触量（行動と無関係）＝フィードバックなし |
| recip_only | a_r R + a_s q | 相互性と共有経験だけ |
| feedback | a_c C + a_r R + a_s q | 自分が行った世話 C（方策が B に依存）＝正のフィードバック |
| memory | B = min(B_max, λ_B Σ ω_k g_k) | 毎日 M から再構成。M を消すと B も消える |

定常解（開ループ g = γB + g0）: ρB = η(1 − B/B_max)(γB + g0) の非負根。
f′(B*) = (1−ρ) + η[−(γB*+g0)/B_max + (1 − B*/B_max)γ]、|f′| < 1 で局所安定。
B* は η/ρ だけに依存する（両辺を ρ で割る）→ 定常データだけでは η と ρ を別々に推定できない。

### 2.7 短期感情 E（`affective_state.py`）

worry ← (1−κ_E)worry + κ_E·(c0 + B)·P̂·sev、frust ← (1−κ_E)frust + 入力（困っていない人を助けたら 0.5）。
κ_E = 0.5/日 ≫ ρ = 0.01/日。shock(j) で E だけを外から操作できる。

### 2.8 意思決定（`planner.py`）

Q(help_j) = κ m_j ΔF_j + (c0 + B_j)·P̂_j·sev_j·eff + β_w worry_j − β_f frust_j − cost_help、Q(rest) = 0。
- 目的実行: π = softmax(Q/τ)、τ = 0.15。risk_j > risk_max の help_j は候補から外す（価値と交換しない）。
- 情報収集: V_info(j) = Σ_{need∈{0,1}} P(need) max_a Q(a|need) − max_a Q(a|現在)。観測で need が確定すると近似。
- 選好の測定 preference_probe: A と C が同じ深刻さで困っている、ΔF = 0、E = 0 のときに A を選ぶ確率。

### 2.9 合成環境（`agent.py`）

人物3人。P(need) = (0.5, 0.3, 0.3)、相互性 0.3、任務事象 0.4（A のみ）、深刻さ N(1, 0.3²)。
観測: 視覚・音（雑音 sd 0.5）、欠測 0.1、別人物の様子の誤観測 0.1。追加観測 sd 0.15。

## 3. 目的関数・損失

- **知覚:** 勾配学習はしていない。H, R, Σ_b, 劣化係数を訓練物体から最小二乗とトークン内分散で較正する
  （`experiments/sensory_fusion.calibrate`）。比較手法の concat はリッジ回帰、late は逆分散の固定重み。
  評価: MAE、ガウス NLL、90%区間の被覆率、判断保留率（事後 sd > 0.35）、回答時 MAE。
- **化学センサー:** s·(1 − exp(−(t−d)/τ)) を (d, τ) の格子で最小二乗、s は線形なので解析解。
- **能動的知覚:** 効用 = 情報利得 − 時間コスト。安全は制約（p_fragile > 0.2 なら press を除外）。
- **エージェント:** 学習される損失はない。B の更新則は仮定であり、競合モデルの比較で検証する。
  評価: B_A, B_C, 選好, φ の Brier, 助けた割合, 1日の幸福合計, 誤った手助け, 追加観測の頻度。

## 4. 数式とコードの対応

| 式 | コード |
| --- | --- |
| 予測・更新・モーメント一致 | `world_model.GaussianState.predict / update` |
| S = R/n + Σ_b、品質による膨張 | `async_fusion.AsyncFuser.summarize` |
| r_i = P(z_i = 1 \| y) | `async_fusion.AsyncFuser.responsibilities` |
| 到着済みのみ・取得時刻順の再フィルタ | `async_fusion.AsyncFuser.estimate` |
| 情報利得 | `async_fusion.information_gain` |
| Beta・Gaussian の人物知識 | `person_model.PersonModel.update` |
| ω = q(1 + Δt/λ)^(−d) | `episodic_memory.EpisodicMemory.weight` |
| B の更新・入力 g・再構成 | `relationship_model.step_B / input_g / B_from_memory` |
| 定常解と f′ | `relationship_model.steady_state` |
| E の更新 | `affective_state.AffectiveState.update` |
| Q, softmax, 安全除外, V_info | `planner.help_values / choose_goal / value_of_info` |

## 5. テスト（`tests/`, 23件, すべて合格を確認）

再帰性（事後→予測）、時間の逆行禁止、観測の二重取り込み禁止、繰返し推定で情報が蓄積しない、
順不同到着＝順序どおりと一致、未来情報の遮断、トークン数の情報上限、共通原因が矛盾を下げ単独観測では r = π、
品質フラグで分散増加、全劣化で判断保留、情報利得 ≥ 0、較正が訓練物体だけを使う、
安全規則、定常解＝数値反復、定常解が η/ρ のみに依存、B ∈ [0, B_max]、記憶の人物単位消去、
感情ショックが E だけを変える、選好測定が感情・任務を無視、記憶消去で B が消えるのは memory モデルだけ、
任務除去で exposure の B が減衰、none モデルでも人物知識は学習される、同シードで再現。

## 6. 再現性

Python 3.13.16、numpy 2.5.4、PyYAML 6.0.3、pytest 9.1.1。シード [0..4] を全実験で共通に使う。
知覚は `default_rng(seed)`、化学は `1000+seed`、能動は `2000+seed`、識別可能性は 7。
