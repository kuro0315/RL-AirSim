# RL-AirSim

## 環境構築

### pythonクライアント

## 現在のプログラム構造（`main.py` 起点）

エントリポイントは `main.py` です。実行時は `config.yaml` 内の `agent_mode` によって以下へ分岐します。

```text
main.py
  └─ main()
      ├─ parse_args()
      ├─ agent_mode: baseline
      │    └─ run_baseline_agent()
      └─ agent_mode: ppo
           └─ run_training()
```

### 1. PPO 学習モード（`agent_mode: ppo`）

- `run_training()` が `env_config` と `training_config` を組み立てる
- `adrl_agent.build_ppo_algorithm()` で RLlib PPO を構築
- `adrl_agent.rllib_agent` で環境 `AirSimDroneRacingEnv-v0` を登録
- `adrl_env.env_creator()` から `AirSimDroneRacingEnv` を生成
- 学習ループで `algorithm.train()` を繰り返し、報酬などを表示

関連ファイル:

- `main.py`
- `rl-airsim/adrl_agent/rllib_agent.py`
- `rl-airsim/adrl_env/airsim_env.py`

### 2. ベースライン実行モード（`agent_mode: baseline`）

- `run_baseline_agent()` が AirSim に接続
- レベル読み込み、レース開始（wrapper/raw RPC のフォールバックあり）
- ゲート座標を取得して `moveOnSplineAsync` で全ゲート追従

関連ファイル:

- `main.py`

### 3. 環境クラス（学習時の実処理）

`AirSimDroneRacingEnv` が Gymnasium 環境として以下を担当します。

- `reset()`:
  - 接続
  - レース状態初期化
  - 離陸
  - ゲート位置取得
- `step()`:
  - 行動適用 (`moveByVelocityAsync`)
  - 観測更新
  - 報酬計算（進捗・ゲート通過・衝突・完走）
  - 終了判定

関連ファイル:

- `rl-airsim/adrl_env/airsim_env.py`

### 4. ルールベース補助制御

- `GateNavigatorRuleController` がゲート方向へのヒューリスティック行動を生成
- 環境側 `_mix_with_rule_controller()` で RL 行動と混合可能
- 混合率は `config.yaml` の `rule_assist.rule_mix` で指定

関連ファイル:

- `rl-airsim/adrl_agent/rule_based_controller.py`
- `rl-airsim/adrl_env/airsim_env.py`

### 補足

- `runtime.sim_launch_mode: nodisplay` のときは `reference/AirSim-Drone-Racing-Lab/settings/settings_no_view.json` を参照して `ViewMode=NoDisplay` を適用します。
- `image_benchmark.enable_image_benchmark: true` のとき、`AirSimDroneRacingEnv` 内で画像APIベンチマーク（`simGetImage` / `simGetImages`）が有効化されます。

## 実行方法（YAML）

`main.py` は YAML ファイルを 1 つ受け取って実行します。

```bash
python main.py configs/main.sample.yaml
```

## 実行コマンド例（Tier別）

Tier 1 ルールベース:

```bash
python main.py configs/tier1_baseline.yaml
```

Tier 1 学習:

```bash
python main.py configs/tier1_ppo.yaml
```

Tier 2 ルールベース:

```bash
python main.py configs/tier2_baseline.yaml
```

Tier 2 学習:

```bash
python main.py configs/tier2_ppo.yaml
```
