# TOTNet Architecture Documentation

## プロジェクト構成

### ディレクトリ構造

```
TOTNet/
├── data/                    # データセット
│   ├── badminton_data/     # バドミントンデータセット
│   ├── tennis_data/        # テニスデータセット  
│   └── tta_dataset/        # 卓球データセット
├── docker/                  # Docker環境設定
├── images/                  # プロジェクト画像資料
├── src/                     # ソースコード
│   ├── config/             # 設定ファイル
│   ├── data_process/       # データ処理モジュール
│   ├── losses_metrics/     # 損失関数と評価指標
│   ├── model/              # モデル定義
│   ├── post_process/       # 後処理
│   └── utils/              # ユーティリティ
└── requirements.txt        # 依存関係
```

## 主要コンポーネント

### 1. モデルアーキテクチャ (`src/model/`)

#### TOTNet (Temporal and Spatial Network)
- **ファイル**: `TOTNet.py`
- **概要**: 時空間情報を活用したボールトラッキングモデル
- **主要クラス**:
  - `ConvBlock`: 空間畳み込みブロック
  - `TemporalConvBlock`: 時間畳み込みブロック  
  - `EncoderBlock`: エンコーダー層（空間+時間処理）
  - `DecoderBlock`: デコーダー層（アップサンプリング+スキップ接続）
  - `BottleNeckBlock`: ボトルネック層

#### その他のモデル
- **WASB** (`wasb.py`): ベースライン比較用モデル
- **TrackNetV2** (`tracknet.py`): 従来手法の実装
- **monoTrack** (`monoTrack.py`): 単一フレームトラッキング
- **TTNet** (`TTNet.py`): 卓球専用モデル

### 2. データ処理 (`src/data_process/`)

#### データローダー (`dataloader.py`)
- **train/val分割**: `create_occlusion_train_val_dataloader()`
- **テスト用**: `create_occlusion_test_dataloader()`
- **対応データセット**: 
  - テニス (`Tennis_Dataset`)
  - バドミントン (`Badminton_Dataset`)
  - 卓球 (`TTA_Dataset`, `Occlusion_Dataset`)

#### データ拡張 (`transformation.py`)
- `RandomColorJitter`: 色調変換
- `Random_Ball_Mask`: ボール遮蔽シミュレーション
- `Random_HFlip/VFlip`: 水平/垂直反転
- `Random_Rotate`: 回転
- `Random_Crop`: ランダムクロップ
- `Resize`: リサイズ
- `Normalize`: 正規化

### 3. 学習・評価 (`src/`)

#### メインスクリプト (`main.py`)
- **分散学習対応**: PyTorch DistributedDataParallel
- **マルチGPU**: torchrun使用
- **学習ループ**: エポック単位の学習と検証

#### テストスクリプト (`test.py`)
- **評価指標**: 
  - 可視性レベル別精度（0:フレーム外, 1:完全可視, 2:部分遮蔽, 3:完全遮蔽）
  - RMSE（距離誤差）
  - Precision/Recall/F1スコア

### 4. 損失関数・評価指標 (`src/losses_metrics/`)

#### 損失関数 (`losses.py`)
- `Heatmap_Ball_Detection_Loss`: ヒートマップベース検出損失
- `Heatmap_Ball_Detection_Loss_Weighted`: 重み付き損失（可視性レベル対応）
- `focal_loss`: Focal Loss実装

#### 評価指標 (`metrics.py`)
- `heatmap_calculate_metrics`: ヒートマップ評価
- `precision_recall_f1_tracknet`: PR曲線計算
- `extract_coords`: ヒートマップから座標抽出

## 推論実行の詳細

### 1. デモスクリプト (`src/demo.py`)

#### 実行方法
```bash
python src/demo.py \
    --video_path <入力動画パス> \
    --pretrained_path <学習済みモデルパス> \
    --model_choice TOTNet \
    --dataset_choice <tennis/badminton/tta> \
    --img_size 288 512 \
    --num_frames 5 \
    --gpu_idx 0 \
    --save_demo_output \
    --save_demo_dir <出力ディレクトリ> \
    --output_format video
```

#### 主要パラメータ
- `--video_path`: 入力動画/フォルダパス
- `--pretrained_path`: 学習済みモデル（必須）
- `--model_choice`: 使用モデル選択
- `--dataset_choice`: データセット種別（前処理方法に影響）
- `--img_size`: 入力解像度 [高さ 幅]
- `--num_frames`: 処理フレーム数（デフォルト5）
- `--save_demo_output`: 結果保存フラグ
- `--output_format`: 出力形式（video/frames）
- `--show_image`: リアルタイム表示

#### 処理フロー
1. **データローダー初期化**
   - ビデオ用: `Video_Loader` (tt/badminton/tta)
   - フォルダ用: `Folder_Loader` (tennis)

2. **モデルロード**
   - 指定モデルアーキテクチャの構築
   - 学習済み重みの読み込み
   - GPU転送・評価モードへ

3. **推論ループ**
   - 連続フレーム読み込み（num_frames分）
   - モデル推論（ヒートマップ+イベント予測）
   - 座標抽出（`extract_coords`）
   - 可視化（ボール位置描画）

4. **出力保存**
   - フレーム単位で画像保存
   - FFmpegで動画変換（video形式指定時）

### 2. バッチテスト (`src/test.py`)

#### 実行方法
```bash
python src/test.py \
    --pretrained_path <モデルパス> \
    --dataset_choice <データセット> \
    --num_samples <サンプル数> \
    --batch_size <バッチサイズ>
```

#### 評価出力
- 可視性レベル別の精度
- 全体のRMSE/Accuracy/Precision/Recall/F1
- FPS計測

### 3. 並列テスト (`src/parallel_test.py`)
複数GPUでの並列評価実行用

## 学習済みモデルの使用

### モデルファイル形式
- PyTorch標準形式（`.pth`または`.pt`）
- state_dict形式で保存

### モデルロード例
```python
from model.TOTNet import build_motion_model_light
from model.model_utils import load_pretrained_model

# モデル構築
model = build_motion_model_light(configs)

# 重みロード
model = load_pretrained_model(model, pretrained_path, gpu_idx)

# 推論モード
model.eval()
```

## 主要設定パラメータ

### 学習設定
- `num_epochs`: エポック数（デフォルト30）
- `batch_size`: バッチサイズ（デフォルト24）
- `lr`: 学習率（5e-4）
- `optimizer_type`: オプティマイザ（adamw）
- `loss_function`: 損失関数（WBCE）
- `weighting_list`: 可視性重み [1,2,2,3]

### データ設定
- `num_frames`: 処理フレーム数（5）
- `img_size`: 入力解像度 [288, 512]
- `occluded_prob`: 遮蔽確率（0.1）
- `ball_size`: ボールサイズ（4）

### 分散学習設定
- `nproc_per_node`: GPU数
- `dist_backend`: バックエンド（nccl）
- `multiprocessing_distributed`: 分散学習フラグ

## 特徴と利点

1. **時空間情報の活用**
   - 連続フレームから時間的文脈を学習
   - 3D畳み込みによる時空間特徴抽出

2. **遮蔽対応**
   - 可視性レベル別の重み付き学習
   - 遮蔽シミュレーションによるデータ拡張

3. **マルチスポーツ対応**
   - テニス、バドミントン、卓球に対応
   - データセット固有の前処理

4. **効率的な推論**
   - リアルタイム処理可能
   - バッチ処理対応

5. **拡張性**
   - モジュラー設計
   - 新規モデル追加が容易