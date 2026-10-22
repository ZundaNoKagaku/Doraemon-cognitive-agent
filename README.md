# 非同期・欠測・矛盾に頑健な多感覚物体推定と、人物別優先度の競合モデル比較（合成環境）

研究計画書「『ドラえもん』の認知過程を模した深層学習エージェントの設計と検証」の付属コードです。
**データはすべて合成データです。** 実センサー・実ロボット・作品の場面データは使っていません。

## 実装済みのモジュール

| 層 | ファイル | 内容 |
| --- | --- | --- |
| 1 感覚入力 | `src/synthetic_perception.py` | 合成の視覚・触覚・接触音の観測バースト（トークン列）を生成 |
| 2 非同期融合 | `src/async_fusion.py` | 到着済み観測だけを取得時刻順に再フィルタ。共通原因確率を厳密列挙で計算。品質フラグ。情報利得 |
| 3 世界状態 | `src/world_model.py` | 線形ガウス状態の予測・更新（予測と更新を分離、同じ観測の二重取り込みを禁止） |
| 4 人物知識 φ | `src/person_model.py` | 困りごとの確率（Beta）と深刻さ（Gaussian） |
| 5 エピソード記憶 M | `src/episodic_memory.py` | 保存・べき乗型忘却・人物単位の消去 |
| 6 関係性重み B | `src/relationship_model.py` | 競合モデル none / exposure / recip_only / feedback / memory、定常解と安定性 |
| 6 短期感情 E | `src/affective_state.py` | 心配・苛立ち（B と別の時間尺度） |
| 7 意思決定 | `src/planner.py` | 情報収集段階と目的実行段階の分離、安全規則による候補除外、選好の測定 |
| 8 統合と環境 | `src/agent.py` | 1日ごとの知覚→決定→結果→更新のループ、反実仮想（記憶消去・任務除去・感情ショック） |

**実装していないもの:** 深層学習の符号化器（`encoders.py` はありません。合成特徴で代替）、
学習による融合（融合は較正済みの確率モデルで、勾配学習はしていません）、実センサーの入出力、
言語・会話、身体の制御、作品場面の符号化データ、主観的経験についてのいかなる測定。

## 実行方法

```bash
pip install -r requirements.txt
python experiments/sensory_fusion.py        # 知覚実験（約45秒）→ results/sensory_fusion.{md,json}
python experiments/relationship_ablation.py # 競合モデル・安定性・識別可能性・知覚→決定（約70秒）
python experiments/memory_ablation.py       # 記憶消去・感情ショック（約50秒）
python -m pytest -q tests                   # 23 件
```

設定は `configs/perception.yaml` と `configs/agent.yaml` に、結果を見る前に固定しました。
シードは全実験で [0, 1, 2, 3, 4]。実行ログは `results/log_*.txt`。

## 主な制約

- 選好の測定（`preference_probe`）は温度 τ = 0.15 の softmax で、B の差が小さくても 1.00 に飽和します。
  モデル間の差は B の値そのもので比べてください。
- memory モデルは B が上限 Bmax = 5 に張り付きます。
