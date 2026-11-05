# Pisikopa翻訳機 lmm風学習 — 訓練記録

位置づけ: **ルール翻訳 (`translator.py`) が主、学習モデルは実験**。ルール翻訳は
EN→PI 1114/1114 完全一致する。学習モデルは lmmもどき形式の tiny transformer で
同じ対応を勾配学習で覚えさせたもの。

## 構成

- 学習データ: `corpus.tsv` 1114ペア。各行を `"英語 = Pisikopa"` の1シーケンスにした
  （例: `I eat an apple. = ka pa ra-ta`）。単語レベル（空白区切りトークン）。
- 語彙: 201トークン（単語198 + `<bos>` `<eos>` `<pad>`）。最大長 20トークン。
- モデル: d_model=48, n_layer=2, n_head=3, d_ff=96, max_len=64。
  `~/workspace/lmm-modoki/train.py` の `Net`（手書き逆伝播 + Adam）と
  `fit.py` の `fmt` を流用。パラメータ数 約60k。
- 生成: `generate_pi.py` が `.modoki` を `modoki.py` のパーサで読み、貪欲生成する。
  プロンプトは `"<英語> ="` の形。

## 実行コマンドと出力

### 1回目（800ステップ）

```
$ python3 -B train_pi.py --steps 800
ペア数: 1114, 語彙数: 201, 最大長: 20 トークン
d_model=48, n_layer=2, steps=800, lr=0.005
step 100 loss 1.131
step 200 loss 0.748
step 300 loss 0.726
step 400 loss 0.702
step 500 loss 0.699
step 600 loss 0.679
step 700 loss 0.672
step 800 loss 0.682
初期 loss 5.958 -> 最終 loss 0.682 (step 800)
書き出し: /home/hatch/workspace/pisikopa-translator/pisikopa.modoki
```

所要時間 約29秒。loss は 0.67〜0.70 で頭打ち気味だった。

### 2回目（2500ステップ、採用）

```
$ python3 -B train_pi.py --steps 2500 --out pisikopa-2500.modoki
step 2000 loss 0.632
step 2100 loss 0.679
step 2200 loss 0.677
step 2300 loss 0.692
step 2400 loss 0.691
step 2500 loss 0.662
初期 loss 5.958 -> 最終 loss 0.662 (step 2500)
```

所要時間 約91秒。loss は 800ステップ版（0.682）からほぼ下がらず 0.66 前後で
横ばい。ただし後述の通り生成の正解率は大きく改善したため、この2500ステップ版を
`pisikopa.modoki` として採用した（`mv pisikopa-2500.modoki pisikopa.modoki`）。

## 生成の評価（貪欲生成、ルール翻訳を正解として比較）

- 学習データ1114プロンプトの完全一致率:
  - 800ステップ版: **1009/1114 = 90.6%**
  - 2500ステップ版: **1114/1114 = 100.0%**
- 800ステップ版の失敗例（動詞の取り違えが多い）:
  - `I felt me drinking an apple.` 期待 `ka pa ri-tu` → 出力 `ka pa ro-tu`（drink→see）
  - `Everyone knows I read an apple.` 期待 `ka pa kæro-tɔ` → 出力 `ka pa kiro-tɔ`（read→buy）
  - `I felt me buying an apple.` 期待 `ka pa kiro-tu` → 出力 `ka pa ro-tu`（buy→see）

## 生成例（最終モデル = 2500ステップ版）

```
$ python3 -B generate_pi.py --prompt "I eat an apple. =" --tokens 10
I eat an apple. = ka pa ra-ta
$ python3 -B generate_pi.py --prompt "I might want to eat an apple. =" --tokens 10
I might want to eat an apple. = ka pa məra-te
$ python3 -B generate_pi.py --prompt "you see an apple. =" --tokens 10
you see an apple. = ka pi ro-ta
$ python3 -B generate_pi.py --prompt "he read a book. =" --tokens 10
he read a book. = kæ pu kæro-ta
```

上記4件はすべて正解（ルール翻訳と一致）。

### 失敗例

```
$ python3 -B generate_pi.py --prompt "You see the sea. =" --tokens 10
エラー: 語彙にない単語: ['You', 'sea.']
$ python3 -B generate_pi.py --prompt "He reads a book. =" --tokens 10
エラー: 語彙にない単語: ['He', 'reads']
```

コーパスに `You` / `sea.` / `He` / `reads` という単語自体が存在しないため、
単語レベルモデルではそもそも入力できない。**語彙外（OOV）は生成不能**という
実験上の制限。`he read a book.` のように語彙内の言い換えなら正しく出る。

```
$ python3 -B generate_pi.py --prompt "we drink a banana. =" --tokens 10
we drink a banana. = pe ro-ti
$ python3 -B generate_pi.py --prompt "In a dream, we see a book. =" --tokens 10
In a dream, we see a book. = kæ pe ro-ta
```

- `we drink a banana.` の正解は `kasæ pe ri-ta`。出力 `pe ro-ti` は目的語なし・
  動詞違い（drink→see）・証拠性違い（-ta→-ti）の三重の間違い。
- `In a dream, we see a book.` の正解は `kæ pe ro-tə`。出力は証拠性だけ `-ta` と
  間違えた。

### 未学習の組み合わせの体系的テスト

主語4種 × 動詞5種 × 目的語5種の単純文で、コーパスにない組み合わせを
ルール翻訳と比較した（ランダム20件）:

- **banana を含むもの: 0/20**。すべて `kasæ`（banana）の代わりに `kæ`（book）を
  出した。主語・動詞は合っている。banana はコーパスに4行しかなく（しかも
  `I saw you stepping on an banana...` という複雑文のみ）、`kasæ` が
  目的語として学習されていないのが原因と思われる。
- banana を除く未学習の組み合わせは、単純文テンプレートがほぼ網羅されているため
  該当が2件しかなく、そのうち `In a dream, we see a book.` は上記の通り証拠性を
  間違えた（0/2 に `reads` の OOV 1件を含む）。

## 結論（盛らずに）

- 学習データの丸暗記は 2500ステップで 100% 達成。loss は 5.958 → 0.662。
- ただし汎化は弱い。特に出現頻度の低い単語（banana / `kasæ`）は未学習の
  組み合わせでまったく出ず、証拠性 `-tə` のようなレアな語尾も取り違える。
  コーパスがテンプレート網羅型で組み合わせの多様性が低いため、暗記に寄った
  結果になっている。
- ルール翻訳が主、学習モデルは実験、という位置づけは変わらない。
  改善するなら banana のようなレア語の行を増やすか、文字レベルへの変更、
  もしくは OOV を扱える仕組みが必要。

## 成果物

- `train_pi.py` — 学習スクリプト（`python3 train_pi.py --help` で動作確認済み）
- `generate_pi.py` — 貪欲生成スクリプト
- `pisikopa.modoki` — 学習済みモデル（2500ステップ版、593,641バイト ≒ 0.57MB）
- `TRAINING.md` — このファイル
- GitHub への push はしていない（指示通り）。
- `__pycache__` は作っていない（`-B` + `sys.dont_write_bytecode = True`）。
