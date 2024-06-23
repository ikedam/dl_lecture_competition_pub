import argparse
import collections
import gzip
import logging
import math
import re
import random
import time
import os
import os.path
import typing
from statistics import mode

# /!\ IMPORTANT WARNING ABOUT TORCHTEXT STATUS /!\ 
# Torchtext is deprecated and the last released version will be 0.18 (this one).
# # You can silence this warning by calling the following at the beginnign of your scripts:
# `import torchtext; torchtext.disable_torchtext_deprecation_warning()`
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from PIL import Image
import einops
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchtext.data.utils
import torchtext.vocab
from torch.nn import functional
from torchvision import transforms
from einops.layers import torch as einopstorch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_seeds() -> dict[str, typing.Any]:
    """現在の乱数状態を保存します。restore_seedsで復元します"""
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch.cuda": torch.cuda.get_rng_state(),
        "torch.cuda_all": torch.cuda.get_rng_state_all(),
    }


def restore_seeds(saved: dict[str, typing.Any]) -> None:
    """save_seeds で保存した状態で乱数を復元します"""
    random.setstate(saved["random"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])
    torch.cuda.set_rng_state(saved["torch.cuda"])
    torch.cuda.set_rng_state_all(saved["torch.cuda_all"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    """文章の正規化

    https://visualqa.org/evaluation.html
    におおよそ準じた変換
    """
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点(追記: 「以外」が抜けてる)のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    # 追記: 「アポストロフィ」の意味。
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    # 追記: alnum, 空白文字、アポストロフィ、コロン以外の文字をスペースに変換
    # コロンは時刻で使われている場合に重要なので残している。
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    # 「数字に含まれるカンマは空白に変換しない」を何らかの理由でミスって実装したと思われる。
    # 実際には直前の変換処理でカンマは残っていないので、無意味な処理。
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    BOS = "<BOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"
    UNK = "<unk>"

    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        self.limit = None

        # answerの辞書を作成
        self.answer2idx = {}
        self.idx2answer = {}

        # 単語分割の実施 (第6回演習)
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.counter = collections.Counter()

        # 質問文に含まれる単語を辞書に追加
        # 質問文の最大長も確認 (バッチの作成で長さを揃えるため)
        self.max_question = 0
        for question in self.df["question"]:
            tokens = self.tokenizer(question)
            self.counter.update(tokens)
            self.max_question = max(len(tokens), self.max_question)
        self.vocabulary = torchtext.vocab.vocab(
            self.counter,
            # 1 回だけの単語は omit する。
            min_freq=2,
            specials=(self.UNK, self.PAD, self.BOS, self.EOS)
        )
        # <unk>をデフォルトに設定することにより，min_freq回以上出てこない単語は<unk>になる
        self.vocabulary.set_default_index(self.vocabulary[self.UNK])

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.vocabulary = dataset.vocabulary
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        question = [self.vocabulary[token] for token in self.tokenizer(self.df["question"][idx])]
        pad_num = self.max_question - len(question)
        question = [self.vocabulary[self.BOS]] + question + [self.vocabulary[self.EOS]] + [self.vocabulary[self.PAD]] * pad_num

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, torch.Tensor(question).to(torch.int64), torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, torch.Tensor(question).to(torch.int64)

    def __len__(self):
        if self.limit:
            return min(self.limit, len(self.df))
        return len(self.df)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        # 3 x h x w -> 64 x h/2 x w/2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #  -> 64 x h/4 x w/4
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


# 第9回演習
# Multi-Head Attentionの実装
# 質問文に対する、画像の特徴量の重み付けを行う。
# Query: 質問文のベクトル
# Key / Value: 画像の特徴量
class MultiHeadAttention(nn.Module):
    def __init__(self, text_dim, image_dim, heads, dim_head, dropout=0.):
        """
        Arguments
        ---------
        text_dim : int
            質問文の次元数
        image_dim : int
            画像の特徴量の次元数
        heads : int
            ヘッドの数
        dim_head : int
            各ヘッドのデータの次元数
        dropout : float
            Dropoutの確率(default=0.)
        """
        super().__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.dim_head = dim_head

        self.heads = heads
        self.scale = math.sqrt(dim_head)  # ソフトマックス関数を適用する前のスケーリング係数(dim_k)

        self.attend = nn.Softmax(dim=-1)  # アテンションスコアの算出に利用するソフトマックス関数
        self.dropout = nn.Dropout(dropout)

        # Q, K, Vに変換するための全結合層
        self.to_q = nn.Linear(in_features=text_dim, out_features=dim_head * heads)
        self.to_k = nn.Linear(in_features=image_dim, out_features=dim_head * heads)
        self.to_v = nn.Linear(in_features=image_dim, out_features=dim_head * heads)
        self.to_out = nn.Sequential(
            nn.Linear(in_features=dim_head * heads, out_features=image_dim),
            nn.Dropout(dropout),
        )

    def forward(self, text, image):
        # 入力データをQ, K, Vに変換する
        # (B, text_dim) -> (B, inner_dim)
        # (B, image_dim) -> (B, inner_dim)
        q = self.to_q(text)
        k = self.to_k(image)
        v = self.to_v(image)

        # Q, K, Vをヘッドに分割する
        # inner_dim = head * dim
        # B, heads * dim_head -> B, heads, dim_head
        q = einops.rearrange(q, "b (h d) -> b h d", h=self.heads, d=self.dim_head)
        k = einops.rearrange(k, "b (h d) -> b h d", h=self.heads, d=self.dim_head)
        v = einops.rearrange(v, "b (h d) -> b h d", h=self.heads, d=self.dim_head)

        # QK^T / sqrt(d_k)を計算する
        # (B, heads, dim_head) x (B, heads, dim_head) -> (B, heads, heads)
        dots = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # ソフトマックス関数でスコアを算出し，Dropoutをする
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # softmax(QK^T / sqrt(d_k))Vを計算する
        # (B, heads, heads) x (B, heads, dim_head) -> (B, heads, dim_head)
        out = torch.matmul(attn, v)

        # もとの形に戻す
        # (B, heads, dim_head) -> (B, dim)
        out = einops.rearrange(out, "b h d -> b (h d)", h=self.heads, d=self.dim_head)

        # 次元をもとに戻して出力
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, text_dim, image_dim, heads, dim_head, hidden_dim, dropout=0.):
        """
        TransformerのEncoder Blockの実装.

        Arguments
        ---------
        text_dim : int
            質問文の次元数
        image_dim : int
            画像の特徴量の次元数
        heads : int
            Multi-Head Attentionのヘッドの数
        dim_head : int
            Multi-Head Attentionの各ヘッドの次元数
        hidden_dim : int
            Feed-Forward Networkの隠れ層の次元数
        dropout : float
            Droptou層の確率p
        """
        super().__init__()

        # Attention前のLayerNorm (レイヤー正規化)
        self.attn_ln_text = nn.LayerNorm(text_dim)
        self.attn_ln_image = nn.LayerNorm(image_dim)  # Attention前のLayerNorm (レイヤー正規化)
        self.attn = MultiHeadAttention(text_dim, image_dim, heads, dim_head, dropout)
        self.ffn_ln = nn.LayerNorm(image_dim)  # FFN前のLayerNorm
        self.ffn = nn.Sequential(
            nn.Linear(in_features=image_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=image_dim),
            nn.Dropout(dropout),
        )

    def forward(self, text, image):
        y = self.attn(self.attn_ln_text(text), self.attn_ln(image))
        x = y + x # skip connection
        out = self.ffn(self.ffn_ln(x)) + x

        return out


class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        self.resnet = ResNet18()

        text_emb_dim = 512
        self.embedding_matrix = nn.Parameter(
            torch.rand((vocab_size, text_emb_dim), dtype=torch.float),
        )
        lstm_dim = 256
        self.bilstm = nn.LSTM(text_emb_dim, lstm_dim, 1, batch_first=True, bidirectional=True)

        heads = 10
        dim_head = 64
        hidden_dim = 192
        transformers = 4
        image_dim = 512
        self.transformer = torch.nn.Sequential(*[TransformerBlock(
            lstm_dim * 2,
            image_dim,
            heads,
            dim_head,
            hidden_dim
        ) for _ in range(transformers)])

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量

        # question: [バッチサイズ, 系列長, vocab_size]
        # → [バッチサイズ, 系列長, emb_dim]
        # → [バッチサイズ, lstm_dim] x 2
        # → [バッチサイズ, lstm_dim x 2]
        # out: 各単語の隠れ出力
        # hc: (h, c)
        # h: (フォーワードの最終出力, バックワードの最終出力)
        # c: セルのパラメーター
        emb = functional.embedding(question, self.embedding_matrix)
        out, hc = self.bilstm(emb)  # テキストの特徴量
        h, c = hc
        question_feature = torch.cat([h[0], h[1]], dim=1)

        image_feature = self.transformer(question_feature, image_feature)

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--snapshots", type=str, default=None)
    parser.add_argument("-l", "--limit", type=int, default=None)
    parser.add_argument("-e", "--epoch", type=int, default=20)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("running on %s", device)
    logger.info("cpu_count %s", os.cpu_count())

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    logger.info(
        "Train info: datasize=%s, vocab size=%s/%s, max_question_len=%s",
        format(len(train_dataset), ","),
        format(len(train_dataset.vocabulary), ","),
        format(len(train_dataset.counter), ","),
        train_dataset.max_question,
    )
    logger.info(
        "Test info: datasize=%s, vocab size=%s/%s, max_question_len=%s",
        format(len(test_dataset), ","),
        format(len(test_dataset.vocabulary), ","),
        format(len(test_dataset.counter), ","),
        test_dataset.max_question,
    )

    # 訓練データの量を制限
    if args.limit:
        logger.warning("limit dataset to %s", args.limit)
        train_dataset.limit = args.limit
        test_dataset.limit = args.limit

    # https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587#1-dataloader%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6
    # に従った最適化を実施
    # なおColabT4環境で検証した限りでは、
    # num_workers は 2 にすると 20% ほど高速化、それより大きくしても目立った変化なし
    # pin_memory は True にしても目立った変化なし
    # であった。
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    model = VQAModel(vocab_size=len(train_dataset.vocabulary), n_answer=len(train_dataset.answer2idx)).to(device)

    # optimizer / criterion
    startepoch = 0
    num_epoch = args.epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    if args.snapshots and os.path.exists(args.snapshots):
        files = sorted(
            [
                f for f in os.listdir(args.snapshots)
                if (
                    (f.endswith(".pth") or f.endswith(".pth.gz"))
                    and f.removesuffix(".gz").removesuffix(".pth").isdigit()
                )
            ],
            reverse=True,
        )
        if len(files) > 0:
            loadfile = os.path.join(args.snapshots, files[0])
            logger.info("load from %s", loadfile)
            startepoch = int(files[0].removesuffix(".gz").removesuffix(".pth"))
            with gzip.open(loadfile, "rb") if loadfile.endswith(".gz") else open(loadfile, "rb") as f:
                data = torch.load(f)
            restore_seeds(data["random"])
            model.load_state_dict(data["model"])
            optimizer.load_state_dict(data["optimizer"])

    # train model
    logger.info("start training...")
    for epoch in range(startepoch, num_epoch):
        logger.info("Epoch %s/%s", epoch + 1, num_epoch)
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        logger.info(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

        if args.snapshots:
            if not os.path.exists(args.snapshots):
                os.makedirs(args.snapshots, exist_ok=True)

            # gzip 圧縮しても90%強程度にしかならない
            # intermediatefile = os.path.join(args.snapshots, ("%04d.pth.gz" % (epoch + 1, )))
            intermediatefile = os.path.join(args.snapshots, ("%04d.pth" % (epoch + 1, )))
            if not os.path.exists(intermediatefile):
                with gzip.open(intermediatefile, "wb") if intermediatefile.endswith(".gz") else open(intermediatefile, "wb") as f:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "random": save_seeds(),
                        },
                        f,
                    )
                logger.info("save %s", intermediatefile)
            else:
                logger.warning("skip saving %s as already exists", intermediatefile)


    # 提出用ファイルの作成
    logger.info("evaluating...")
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    logger.info("evaluation done.")
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
