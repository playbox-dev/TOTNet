# TOTNet: 時空間情報を活用したボールトラッキングネットワーク

![Net_Architecture-Model_Example](https://github.com/user-attachments/assets/77b3a677-489b-4ee8-b41b-21c46d08c18c)

![Demo](./docs/images/TOTNet_Example.gif)

TOTNetは、時空間情報を活用してボールトラッキングを行う深層学習モデルです。特に遮蔽が発生する困難なシナリオでの追跡精度向上を目的としています。

> **注意**: これは[オリジナルのTOTNet](https://github.com/AugustRushG/TOTNet)をforkして改良したリポジトリです。  
> オリジナルのREADMEは[docs/README_orgin.md](./docs/README_orgin.md)に保存されています。

---

## ディレクトリ構成

```
TOTNet/
├── data/                       # データセット格納ディレクトリ
│   ├── badminton_data/        # バドミントンデータセット
│   ├── tennis_data/           # テニスデータセット
│   └── tta_dataset/           # 卓球データセット
├── docs/                      # ドキュメント
│   ├── README_orgin.md        # オリジナルREADME
│   └── images/                # ドキュメント用画像
├── environments/              # Docker環境設定
│   ├── Dockerfile             # マルチステージビルド対応
│   ├── docker-compose.yml    # Docker Compose設定
│   └── build.sh              # ビルドスクリプト
├── models/                   # モデル置き場
├── outputs/                  # 出力ディレクトリ
├── src/                      # ソースコード
│   ├── config/               # 設定ファイル
│   │   ├── config.py         # 基本設定
│   │   ├── two_stream_network.yaml
│   │   └── wasb.yaml
│   ├── data_process/         # データ処理
│   │   ├── dataloader.py    # データローダー
│   │   ├── dataset.py       # データセット定義
│   │   └── transformation.py # データ拡張
│   ├── losses_metrics/       # 損失関数と評価指標
│   │   ├── losses.py        # 損失関数定義
│   │   └── metrics.py       # 評価指標
│   ├── model/                # モデル定義
│   │   ├── TOTNet.py        # メインモデル
│   │   ├── TOTNet_OF.py     # オプティカルフロー版
│   │   ├── TTNet.py         # 卓球用モデル
│   │   ├── tracknet.py      # TrackNetV2
│   │   ├── wasb.py          # WASBモデル
│   │   └── ops/             # カスタムオペレーション
│   ├── post_process/         # 後処理
│   │   ├── bounce_detection.py
│   │   └── table_detection.py
│   ├── utils/                # ユーティリティ
│   │   ├── logger.py        # ロギング
│   │   ├── train_utils.py   # 学習ユーティリティ
│   │   └── visualization.py # 可視化
│   ├── main.py              # メイン学習スクリプト
│   ├── test.py              # テストスクリプト
│   ├── demo.py              # デモスクリプト
│   ├── playbox_train.sh     # 学習実行スクリプト（新規）
│   ├── train.sh             # 標準学習スクリプト
│   └── test.sh              # テスト実行スクリプト
├── ARCHITECTURE.md          # アーキテクチャ説明（新規）
├── LICENSE                  # ライセンス
├── README.md               # このファイル
└── requirements.txt        # Python依存関係
```

---

## システム要件

### ハードウェア要件

- **GPU**: NVIDIA GPU (CUDA 11.8対応)

### ソフトウェア要件

- **OS**: Ubuntu 20.04/22.04 LTS
- **Docker**: 20.10以降
- **Docker Compose**: 2.0以降
- **NVIDIA Driver**: 520以降
- **NVIDIA Container Toolkit**: インストール済み

### Python環境（Dockerを使用しない場合）

- **Python**: 3.10
- **PyTorch**: 2.3.1
- **CUDA**: 11.8
- **cuDNN**: 8.6以降

---

## 環境構築

### 1. Dockerを使用する場合（推奨）

```bash
# リポジトリのクローン
git clone <repository-url>
cd TOTNet

# Docker環境のビルド
cd environments
./build.sh

# コンテナの起動
docker-compose up -d

# コンテナに入る
docker exec -it totnet /bin/bash
```

### 2. ローカル環境の場合

```bash
# Python 3.10環境の準備
conda create -n totnet python=3.10
conda activate totnet

# 依存関係のインストール
pip install -r requirements.txt

# CUDAの設定確認
python -c "import torch; print(torch.cuda.is_available())"
```

---

## データセットの準備

### データセットのダウンロード

各データセットについては，[WASB-SBDT](https://github.com/nttcom/WASB-SBDT/blob/main/GET_STARTED.md) からダウンロードできる．

1. **テニス・バドミントンデータセット**
   - [TrackNetV2](https://nycu1-my.sharepoint.com/personal/tik_m365_nycu_edu_tw/_layouts/15/onedrive.aspx?id=/personal/tik_m365_nycu_edu_tw/Documents/OpenDataset/TrackNetV2_Badminton/TrackNetV2.zip&parent=/personal/tik_m365_nycu_edu_tw/Documents/OpenDataset/TrackNetV2_Badminton&ga=1)

### データの配置

```bash
# データディレクトリの作成
mkdir -p data/{badminton_data,tennis_data,tta_dataset}
```

---

## 学習の実行

### 基本的な学習コマンド

```bash
# シングルGPU
python src/main.py \
    --num_epochs 30 \
    --model_choice TOTNet \
    --dataset_choice badminton \
    --batch_size 8

# マルチGPU（分散学習）
torchrun --nproc_per_node=2 src/main.py \
    --num_epochs 30 \
    --model_choice TOTNet \
    --dataset_choice tennis \
    --batch_size 24 \
    --distributed
```

### 最適化された学習スクリプトの使用

```bash
# バドミントンデータセット用の最適化設定
cd src
./playbox_train.sh
```

主要パラメータ：

- `--num_frames 5`: 使用するフレーム数
- `--lr 5e-4`: 学習率
- `--batch_size 3`: バッチサイズ（メモリに応じて調整）
- `--occluded_prob 0.1`: 遮蔽シミュレーション確率

---

## テストと評価

```bash
# 学習済みモデルのテスト
python src/test.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --dataset_choice badminton \
    --test_data_path data/badminton_data/test_match1

# デモの実行
python src/demo.py \
    --video_path data/badminton_data/test_video.mp4 \
    --checkpoint outputs/checkpoints/best_model.pth
```

---

## トラブルシューティング

### CUDA Out of Memory

- `batch_size`を小さくする
- `img_size`を小さくする（例: `--img_size 144 256`）
- gradient checkpointingを有効にする

---

## 参考文献

オリジナル論文とリポジトリ：

- [TOTNet GitHub Repository](https://github.com/AugustRushG/TOTNet)
- 関連研究: TrackNet, WASB-SBDT, TTNet

