# DL基礎講座2024　最終課題「Visual Question Answering（VQA）」

課題の詳細は以下を参照してください:
https://github.com/ailorg/dl_lecture_competition_pub/tree/VQA-competition


## 実行方法

Google Colab での実行を想定しています。

[notebooks/VQA_on_repo.ipynb](./notebooks/VQA_on_repo.ipynb) を Google Coab 上にアップロードして各種設定値を変更の上、実行してください。
詳細はノートブックに記載のコメントを参照してください。

T4 GPU を使用して 1 エポック 6 分弱、全実行完了に 1 時間くらいかかります。

Google Colab のライブラリーバージョンが変わると実行できなくなる可能性があります。
最後に実行したときのライブラリーの状態を [requirements-colab-2024-07-05.txt](./requirements-colab-2024-07-05.txt) に記録してあります。


## ファイル構成

* main.py: 学習プログラム
    * `python main.py -h` でオプションが表示されます。
* requirements類:
    * requirements.txt: main.py の依存ライブラリー
    * requirements-local.txt: ローカルで開発したときに使用した環境の依存ライブラリー (pip freeze したもの)。IDE の動作に使っただけで、実際のプログラム実行ができるものになっているかは保証されません。
    * requirements-colab-2024-07-05.txt: 最終結果の実行時の Colab 環境上のライブラリー
* data: 訓練データ(兼検証データ)、テストデータのロード場所
* notebooks/
    * VQA_on_repo.ipynb: 本プログラムを Google Colab 上で実行するためのノートブック
    * VQA_review_data.ipynb: 今回の訓練データの内容を確認するためのノートブック
