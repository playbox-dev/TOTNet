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
├── data/                      # データセット格納ディレクトリ
│   ├── badminton/             # バドミントンデータセット
│   │   └── TrackNetV2/        # TrackNetV2データ(TrackNetV2作者が用意)
│   ├── tennis/                # テニスデータセット
│   │   └── TrackNet/          # TrackNetデータ(TrackNet作者が用意)
│   └── table_tennis/          # 卓球データセット
│       └── tta/                # ttaデータセット(TOTNet作者が用意)
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
│   ├── train.sh             # 標準学習スクリプト
│   └── test.sh              # テスト実行スクリプト
├── LICENSE                  # ライセンス
└── README.md               # このファイル
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

### 2. ローカル環境の場合 (非推奨)

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

元実装には，TrackNet(テニス), TrackNetV2(バドミントン), tta(卓球)データセット用のローダーが用意されている．

各データセットは以下でダウンロード可能．

- [TrackNet](https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut)
- [TrackNetV2](https://nycu1-my.sharepoint.com/:u:/r/personal/tik_m365_nycu_edu_tw/Documents/OpenDataset/TrackNetV2_Badminton/TrackNetV2.zip?csf=1&web=1&e=G1dBec)
- tta -> TOTNet作者にメールで問い合わせ．

```sh
├── data/                      # データセット格納ディレクトリ
│   ├── badminton/             # バドミントンデータセット
│   │   └── TrackNetV2/        # TrackNetV2データ(TrackNetV2作者が用意)
│   ├── tennis/                # テニスデータセット
│   │   └── TrackNet/          # TrackNetデータ(TrackNet作者が用意)
│   └── table_tennis/          # 卓球データセット
│       └── tta/               # ttaデータセット(TOTNet作者が用意)
```

ダウンロードしてきたデータを上記の通りに配置すれば，
学習時に以下のパラメータを指定することで，データセットを読み込むことができる．

- `--dataset_choice 'tennis'`: TrackNetV2
- `--dataset_choice 'badminton'`: TrackNet
- `--dataset_choice 'tennis'`: tta

また上のオリジナル実装にあるオプションに加えて，カスタムのデータセットを読み込むオプションも追加.

- `--dataset_choice 'tracknetv2'`
  実装は，`src/config/config.py` を確認して欲しい．

  ```sh
  .
  ├── README.md
  ├── test
  │   ├── match1
  │   │   ├── csv
  │   │   │   ├── 1_05_02_ball.csv
  │   │   │   ├── (...)
  │   │   │   └── 2_03_10_ball.csv
  │   │   ├── frame
  │   │   │   ├── 1_05_02/
  │   │   │   │   ├──── 00000.png
  │   │   │   │   ├──── (...)
  │   │   │   │   └──── XXXXX.png
  │   │   │   ├── (...)
  │   │   │   └── 2_03_10/
  │   │   └── video
  │   │       ├── 1_05_02.mp4
  │   │       ├── (...)
  │   │       └── 2_03_10.mp4
  │   ├── match2
  │   └── match3
  └── train
      ├── match1
      ├── (...)
      └── match26

  32 directories, 1 file
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
