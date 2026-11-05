# Pisikopa翻訳機 lmm風学習 — 訓練記録

位置づけ: **ルール翻訳 (`translator.py`) が主、学習モデルは実験**。ルール翻訳は
EN→PI でコーパス全件 6220/6220 完全一致する。学習モデルは lmmもどき形式の
tiny transformer で同じ対応を勾配学習で覚えさせたもの。

## v0.3 要約（2026-11-05〜06）

- コーパス: 1114ペア → **6172ペア（学習）+ 48ペア（ホールドアウト）**。
  全Pisikopa語のうち名詞69語すべて・動詞14種・主語6種・証拠性8種を網羅。
- ホールドアウト（kasæ+ sell の未学習組み合わせ）: **48/48 = 100%**。
  ベースライン（v0.2モデル）は 0/32（sell 自体が語彙外で全件OOV）。
  banana 20件も 1/20（5%）→ **20/20（100%）**。
- 学習データ全件: **6168/6172 = 99.9%**。残り4件は目的語なしの命令文
  （`run !` / `wake !` / `come !` / `sleep !`）のみ。
- 採用モデル: d_model=64, 3層, 5000ステップ, パラメータ155,208,
  loss 6.248 → 0.840, 学習時間 約7.5分。`pisikopa.modoki`（1,507,102バイト）。
- 目標のホールドアウト80%以上は達成。ただし loss の数字だけ見ると
  v0.2（0.662）より悪く見える。これはコーパスが5.5倍・語彙が約2倍に
  なったためで、loss の単純比較はできない（正解率は上がっている）。

## 1. 診断 — v0.2の汎化の弱さの原因（データで）

v0.2コーパス（1114ペア）を数えた結果:

| 観点 | v0.2の実測 | 意味 |
|---|---|---|
| `kasæ`（banana）の出現 | **4回のみ**。しかも埋め込み文（`I saw you stepping on an banana ...`）の目的語としてだけ。**他動詞の直接目的語としては0回**、証拠性のバリエーションもなし | `kasæ` が「目的語」として学習されていない。未学習の組み合わせで `kasæ` の代わりに `kæ`（book）を出すのはこのため |
| 直接目的語の種類 | **4語のみ**（kaりんご / ke水 / kæ本 / ki米）。`kasæ` は上記4回を除き目的語にならない | 目的語スロットが4語で固定され、モデルは「目的語＝この4語のどれか」を覚えた |
| 動詞 | 他動詞5種（eat/drink/see/read/buy）+ 自動詞4種のみ。sell/write/make/hear/speak/come/wake/step/cry は主動詞として0回 | `sell` は語彙にすらなく、ホールドアウト以前に生成不能 |
| 主語 | 4種（I/you/he/we）のみ。she/they なし | |
| lexicon の網羅率 | `lexicon_en.json` の Pisikopa 値143種のうち **122種がコーパスに一度も出ない**（英語側も145語中123語が出ない） | 語彙の85%が未学習 |
| 証拠性 | ta 404回に対し ti/tu/to/tə/tæ/tɔ は各96回 | ta 過多（否定・疑問・時間・願望が ta のみだったため） |
| 入力の脆さ | `You see the sea.` → 語彙外 `['You', 'sea.']` でエラー。`He reads a book.` → `['He', 'reads']` でエラー | 大文字・句読点・三人称単数 -s がそのまま語彙照合されていた |

結論: 弱さの原因はモデル容量ではなく**データの偏り**。`kasæ` は
「見たことはあるが目的語として使われたことがない」単語で、組み合わせ
汎化のテストになっていなかった。

## 2. コーパス v2（`gen_corpus.py` / `corpus.tsv` / `holdout.tsv`）

### 英語側の正規化（v2の前提）

コーパスは正規化済みの英語で保存する。小文字化、句読点（`. , ; : " ' ( )`）
除去、`?` / `!` は別トークンとして残す、三人称単数 → 原形
（`reads` → `read`）。`translator.normalize_en` と `generate_pi.py`
が同じ正規化を使うので、ユーザ入力 `You see the sea.` は
`you see the sea` になって語彙に当たる。

- 旧: `I eat an apple.` / 新: `i eat an apple`
- 旧: `Did I eat an apple?` / 新: `did i eat an apple ?`
- 旧: `"I eat an apple," someone said.` / 新: `i eat an apple someone said`
  （引用符を落としても `someone said` マーカーで tæ になる）

### 規模と内訳（学習 6172ペア）

| カテゴリ | 数 | 内容 |
|---|---|---|
| transitive-core | 3024 | 主語6 × 他動詞8 × 中核目的語8 × 証拠性8（kasæ+sell 48件を除く） |
| transitive-extended | 1464 | 残り61名詞 × 6主語 × 回転する動詞2種 × 回転する証拠性2種 |
| intransitive | 288 | 主語6 × 自動詞6 × 証拠性8 |
| negation | 192 | 主語6 × 動詞4 × 目的語8（ta） |
| question-yesno | 192 | 主語6 × 動詞4 × 目的語8（ta） |
| question-wh | 24 | `what did {主語} {動詞} ?`（6 × 4） |
| imperative | 64 | 他動詞8 × 目的語8 |
| imperative-intrans | 6 | 自動詞6（目的語なし） |
| want | 384 | 主語6 × 動詞4 × 目的語8 × 証拠性 ta/te |
| time | 384 | 主語6 × 動詞2 × 目的語8 × 時間4（today/tomorrow/yesterday/now） |
| embedding-saw-that | 36 | `i saw that ...`（主語2: you/he/they × 動詞3 × 目的語4） |
| embedding-gerund | 18 | `i saw ... stepping on ... (and crying)`（目的語 banana/apple/sea） |
| see-colour-number | 96 | `see + 色8/数8` × 主語6（ta） |
| **合計** | **6172** | + ホールドアウト 48 = 生成総数 6220 |

他動詞8種: eat/drink/see/read/buy/**sell/write/make**（太字が新規）。
自動詞6種: go/come/sleep/wake/run/swim（come/wake が新規）。
主語6種: i/you/he/**she**/we/**they**（太字が新規）。
目的語: lexicon の名詞 **69語すべて**（中核8 + 拡張61）。

### kasæ の扱い（改善の核心）

- 出現回数: 4回 → **494回**。うち直接目的語として **462回**。
- kasæ × 他動詞7種（sell 以外）× 主語6 × 証拠性8 = 336ペアを学習。
- kasæ × **sell** は学習から除外し、ホールドアウトへ（下記 §3）。
- sell 自体は他目的語で 516行学習している（kasæ 以外の目的語 × sell）。

### 網羅率の変化

- lexicon の Pisikopa 値143種のうちコーパスに一度も出ないもの:
  122種 → **33種**。
- 残る33種は、形容詞（`lala` 新しい、`mako` 暖かい、など22種）と
  指示詞（`pə` これ / `pæ` それ / `lə` 場所）など、**SPEC に
  文テンプレートがない品詞**（形容詞の修飾構文が未定義）か、
  証拠性つきでしか出ない語根の裸形（`moru` 踏むは `moru-ta` として出る、
  `məra` 食べたいは `məra-ta` として出る）。無理に混ぜず未対応として残す。
- 名詞69語・動詞14種・代名詞6種・時間4種・証拠性8種・機能語
  （mæ/lo/rɔ/mə-/læ）はすべてコーパスに出る。

### 重複・衝突チェック

`gen_corpus.py` が重複除去と衝突チェックを行い、**collisions 0** で
`corpus.tsv` + `holdout.tsv` を書き出す（衝突があれば終了コード1）。
ただし2点、意図的な例外がある:

- he/she は Pisikopa では同じ `pu` になるため、同じ PI 文になるのは
  正しい。衝突チェックは he/she（him/her）を同一視して判定する。
- `Who ate an apple?`（`ka pu ra-ta lo`）は生成しない。yes/no 疑問文
  `Did he eat an apple?` と同じ PI 文になり、言語仕様上の本物の
  曖昧さだから。コーパスのバグではないので、wh疑問文は `what` のみ。

### ルール翻訳との照合

`corpus.tsv` + `holdout.tsv` 全6220ペアに対し `en_to_pi` を実行:
**6220/6220 完全一致**。`test_translator.py` は **37 passed, 0 failed**
（正規化の7ケースを追加: `You see the sea.` / `He reads a book.` /
`She sells a banana.` / `They sell a banana.` / `I see red.` /
`She doesn't eat a banana.` / 引用文）。

照合で見つかった translator.py のバグ2件も直した:
`waking` が GERUND になく `paro`（王）と誤訳されていた問題と、
`wake !` が命令文と判定されなかった問題。

## 3. ホールドアウト評価の設計とベースライン

- ホールドアウト: **kasæ（banana）を目的語とする sell（kuro）の
  組み合わせ全48ペア**（主語6 × 証拠性8）を `holdout.tsv` に分離。
  学習データには kasæ+sell が **0件**。
  モデルは kasæ を他動詞7種で、sell を他目的語で見ているが、
  この組み合わせだけ見ていない。合成的汎化のテスト。
- 評価スクリプト: `eval_model.py`
  （`python3 eval_model.py --model <file> [--train-sample N]`）。
  ホールドアウト + banana20 + 学習データのランダムサンプルを測る。
- banana20: v0.2 と同じ20の（動詞, 証拠性）組み合わせを v2 の
  正規化形式で再構成したもの（`eval_model.py` 内で生成）。

ベースライン（v0.2モデル `pisikopa.modoki` 2500ステップ版、旧形式で測定）:

| 評価 | v0.2モデル | 備考 |
|---|---|---|
| banana20 | **1/20 = 5.0%** | 17件は `kasæ` → `kæ` の取り違え、2件は `banana,"` でOOV。1件だけ正解（`In a dream, we drink an banana.` → `kasæ pe ri-tə`）。TRAINING.md v0.2 の 0/20 は別ランダム20件での数字で、組み合わせ集合が違うため単純比較できない |
| kasæ+sell | **0/32 = 0%** | 4主語 × 8証拠性で測定。全件OOV（`sell` / `selling` が語彙にない）。48件版は she/they を含むため旧モデルでは測定不能 |
| OOV | `You see the sea.` → `['You', 'sea.']`、`He reads a book.` → `['He', 'reads']` | いずれもエラー終了 |

## 4. 入力正規化と未知語の扱い

`translator.normalize_en`（`translator.py` / `generate_pi.py` 共用）:

- 小文字化、句読点除去（`?` / `!` は別トークンとして残す）、
  短縮形の展開（`doesn't` → `does not`）、三人称単数 → 原形
  （`reads` → `read`, `sells` → `sell`）。
- `You see the sea.` → `you see the sea`、`He reads a book.` →
  `he read a book` になり、OOV にならない。
- **正規化後も語彙にない単語は生成不能**。`generate_pi.py` は
  未知語を一覧にしてエラー終了する（終了コード1）。
  例: `flibberty gibbet =` →
  `エラー: 語彙にない単語: ['flibberty', 'gibbet']（正規化済み。
  未知語は生成できません。translator.py vec <単語> で一番近い
  Pisikopa 語を確認してください）`。
- 未知語のフォールバックはルール翻訳側にしかない
  （`translator.py vec <単語>` の lexicon / cosine-compound /
  string-overlap-fallback）。ニューラルモデルに未知語の推測機能はない。
  これが今のモデルの仕様上の限界。
- 形容詞も未対応。`I see a red car.` → `ruko pa ro-ta`
  （`red` を落として「車を見る」になる）。ルール翻訳は
  `sæko pa ro-ta`（赤を見る）を返す。どちらも正しい扱いではない。
  SPEC に形容詞の修飾構文がないため。

## 5. モデル実験（v0.3）

共通: `train_pi.py`、単語レベル、語彙392トークン、最大長20、
lr 0.005、バッチ16、seed 0。`--dff` / `--heads` オプションを追加
（d=64 には heads=4、d=96 には heads=6。d を heads で割り切れる必要がある）。

| 実験 | d_model | 層 | heads | d_ff | ステップ | パラメータ | 初期loss → 最終loss | 学習時間 |
|---|---|---|---|---|---|---|---|---|
| A（採用） | 64 | 3 | 4 | 128 | 5000 | 155,208 | 6.248 → **0.840** | 約7.5分 |
| B | 96 | 3 | 6 | 192 | 5000 | 306,344 | 6.567 → 0.857 | 約14分 |
| C | 64 | 3 | 4 | 128 | 8000 | 155,208 | 6.248 → 0.842 | 約12分 |
| （参考 v0.2） | 48 | 2 | 3 | 96 | 2500 | 60,585 | 5.958 → 0.662 | 約91秒 |

実行コマンド（A）:
```
$ python3 -B train_pi.py --steps 5000 --d 64 --layers 3 --heads 4 --dff 128 --out pisikopa-d64L3.modoki
ペア数: 6172, 語彙数: 392, 最大長: 20 トークン
d_model=64, n_layer=3, n_head=4, d_ff=128, steps=5000, lr=0.005
初期 loss 6.248 -> 最終 loss 0.840 (step 5000)
```

### loss 曲線（500ステップごと、実測値）

A（d64, 5000ステップ）:
```
step  100: 1.276    step 4000: 0.794
step  500: 0.980    step 4500: 0.885
step 1500: 0.883    step 5000: 0.840
step 2000: 0.814
```
B（d96, 5000ステップ）:
```
step  500: 0.985    step 3000: 0.845
step 1000: 0.811    step 4000: 0.818
step 1500: 0.890    step 4500: 0.839
step 2000: 0.800    step 5000: 0.857
step 2500: 0.795
```
C（d64, 8000ステップ、A と同一軌道で継続）:
```
step 5500: 0.785    step 7000: 0.794
step 6000: 0.760    step 7500: 0.777
step 6500: 0.767    step 8000: 0.842
```
loss は 500ステップ以降 0.76〜0.98 の範囲で横ばい・上下し、
明確な下降は見られない。v0.2（0.662）より高いのはコーパスが
5.5倍・語彙が約2倍でタスクが難しいためで、loss の値だけでの
新旧比較はできない。

### 評価結果の比較（`eval_model.py`、貪欲生成）

| 実験 | ホールドアウト kasæ+sell | banana20 | 学習データ全件 |
|---|---|---|---|
| A（採用） | **48/48 = 100%** | **20/20 = 100%** | **6168/6172 = 99.9%**（4件失敗） |
| B | 48/48 = 100% | 20/20 = 100% | 6157/6172 = 99.8%（15件失敗） |
| C | 48/48 = 100% | 20/20 = 100% | 6079/6172 = 98.5%（93件失敗） |

- B は A よりパラメータが約2倍だが、学習データの正解率はむしろ
  悪く（wh疑問文で `lɔ pi lɔ lo` のように動詞を `lɔ` と繰り返す
  失敗が目立つ）、ホールドアウトも同じ100%。大きくしても得なし。
- C はステップを8000まで伸ばしたが、学習データの正解率が
  **悪化した**（99.9% → 98.5%、93件失敗。`i felt me seeing a snake`
  → `= po ro-tu` のように `=` や誤った主語を出す）。
  loss は A とほぼ同じ（0.842）なのに貪欲生成の精度が落ちる。
  ステップを伸ばせば良いというものではない、という実測。
- よって **A を採用**。B/C のファイルは削除済み（再現コマンドは上記）。

## 6. 最終モデル（`pisikopa.modoki` = 実験A）の評価

```
$ python3 -B eval_model.py --model pisikopa.modoki --train-sample 6172 --seed 1
holdout (kasæ+sell, unseen combo): 48/48 = 100.0%  (oov 0, wrong 0)
banana20 (same combos as v0.2 baseline): 20/20 = 100.0%  (oov 0, wrong 0)
train sample (seed 1): 6168/6172 = 99.9%  (oov 0, wrong 4)
```

- ホールドアウト 48/48（目標80%を大きく上回る）。
- 学習データ全件 6168/6172 = 99.9%。失敗は4件のみで、すべて
  目的語なしの命令文（下記 §7）。
- 500件サンプル（seed 0）では 498/500 = 99.6%（同じ2件の失敗）。

## 7. 生成例（最終モデル）

成功例（すべてルール翻訳と一致）:
```
$ python3 -B generate_pi.py --prompt "I eat an apple. ="
I eat an apple. = ka pa ra-ta
$ python3 -B generate_pi.py --prompt "You see the sea. ="
You see the sea. = si pi ro-ta
$ python3 -B generate_pi.py --prompt "He reads a book. ="
He reads a book. = kæ pu kæro-ta
$ python3 -B generate_pi.py --prompt "She sells a banana. ="
She sells a banana. = kasæ pu kuro-ta
$ python3 -B generate_pi.py --prompt "I saw that they sell a banana. ="
I saw that they sell a banana. = kasæ po kuro-ta læ pa ro-ta
$ python3 -B generate_pi.py --prompt "In a dream, we drink a banana. ="
In a dream, we drink a banana. = kasæ pe ri-tə
$ python3 -B generate_pi.py --prompt "Did you sell a banana? ="
Did you sell a banana? = kasæ pi kuro-ta lo
$ python3 -B generate_pi.py --prompt "Eat a banana! ="
Eat a banana! = kasæ ra rɔ
```

失敗例（学習データ全件での4件、すべて目的語なし命令文）:
```
$ python3 -B generate_pi.py --prompt "run !"
run ! = siru rɔ          （期待 ruru rɔ、swim と取り違え）
$ python3 -B generate_pi.py --prompt "wake !"
wake ! = siru rɔ         （期待 laro rɔ、swim と取り違え）
$ python3 -B generate_pi.py --prompt "come !"
come ! = pu re-ta        （期待 re rɔ、命令文にならず平叙文になる）
$ python3 -B generate_pi.py --prompt "sleep !"
sleep ! = siru rɔ        （期待 lulo rɔ、swim と取り違え）
```
原因: 目的語なし命令文はコーパスに6行しかなく（自動詞6種 × 1行）、
`swim !` → `siru rɔ` が正解の行があるため、モデルは動詞が不明な
命令文を `siru`（swim）に寄せる。`go !` → `ru rɔ` と
`swim !` → `siru rɔ` は正解するので 6行中2行だけ合っている。
データ量の問題で、容量やステップでは直らなかった（B/C でも同じ傾向）。

失敗例（仕様上の限界）:
```
$ python3 -B generate_pi.py --prompt "I see a red car. ="
I see a red car. = ruko pa ro-ta   （red を落として「車を見る」に）
$ python3 -B generate_pi.py --prompt "flibberty gibbet ="
エラー: 語彙にない単語: ['flibberty', 'gibbet']（正規化済み。未知語は生成できません。
translator.py vec <単語> で一番近い Pisikopa 語を確認してください）
```

## 8. v0.2 の記録（2026-11-05、参考として残す）

- 学習データ: `corpus.tsv` 1114ペア。語彙201トークン。最大長20。
- モデル: d_model=48, n_layer=2, n_head=3, d_ff=96, max_len=64。
  パラメータ数 約60k（実測 60,585）。
- 1回目（800ステップ）: 初期 loss 5.958 -> 最終 loss 0.682。所要時間 約29秒。
  学習データ完全一致率 1009/1114 = 90.6%。
- 2回目（2500ステップ、v0.2として採用）: 最終 loss 0.662。所要時間 約91秒。
  学習データ完全一致率 1114/1114 = 100.0%。
- 失敗例（v0.2）: `we drink a banana.` → `pe ro-ti`（正解 `kasæ pe ri-ta`）、
  `In a dream, we see a book.` → `kæ pe ro-ta`（正解 `kæ pe ro-tə`）。
  未学習の組み合わせで banana を含むものは 0/20（別ランダム20件）。
- v0.2 のモデルファイルは v0.3 で上書きした（`pisikopa.modoki` は現在 v0.3）。
  v0.2 の再現: 旧 `gen_corpus.py` は GitHub 履歴参照。

## 9. 結論（盛らずに）

- ホールドアウト 100%（48/48）で目標の80%は達成。banana の汎化も
  1/20 → 20/20。原因だったデータの偏り（kasæ 4回・目的語4語のみ・
  動詞9種のみ・lexicon の85%未学習）は、コーパス拡充でほぼ解消した。
- ただし「丸暗記に近い」構造は変わらない。学習データ 99.9% に対し、
  ホールドアウトが 100% なのは、ホールドアウトがテンプレート内の
  組み合わせだから。テンプレート外の自由文（関係節・形容詞・未知語
  だらけの文）は相変わらず扱えない。
- モデルを大きくする（d96）・ステップを伸ばす（8000）はいずれも
  改善しなかった。効いたのはデータ拡充と入力正規化。
- ルール翻訳が主、学習モデルは実験、という位置づけは変わらない。
  改善するなら、形容詞の構文を SPEC で決めてコーパスに入れるか、
  文字レベルへの変更、もしくは OOV を扱える仕組みが必要。

## 成果物

- `gen_corpus.py` — コーパス生成 v2（重複・衝突チェック付き、
  ホールドアウト分離つき）
- `corpus.tsv` — 並行コーパス 6172ペア（正規化済み英語）
- `holdout.tsv` — ホールドアウト 48ペア（kasæ+sell）
- `translator.py` — ルール翻訳 + `normalize_en`（小文字化・句読点除去・
  三人称単数→原形）+ GERUND に `waking` 追加 + `wake` の命令文対応
- `test_translator.py` — テスト37ケース（正規化7件を追加）
- `train_pi.py` — 学習スクリプト（`--dff` / `--heads` オプション追加）
- `generate_pi.py` — 貪欲生成（プロンプトを `normalize_en` で正規化、
  OOV は未知語一覧つきエラー）
- `eval_model.py` — 評価スクリプト（ホールドアウト / banana20 /
  学習データサンプル）
- `pisikopa.modoki` — 学習済みモデル（v0.3、実験A、1,507,102バイト、
  md5 3964ff7c282cc406fac4634fb96c975f）
- `TRAINING.md` — このファイル
- GitHub への push はしていない（指示通り）。
- `__pycache__` は置いていない（学習・生成は `-B` + `sys.dont_write_bytecode = True`。
  `python3 test_translator.py` を -B なしで走らせるとできるので、その都度消すこと）。
- 注意: 作業中に `/tmp` が消える出来事が1回あった（VM再起動と思われる）。
  以後、学習ログは `~/workspace/` に置き、評価は workspace 内の
  ファイルだけで行った。`/tmp/pisikopa-old.modoki`（v0.2モデルの
  バックアップ）は消えたが、ベースラインの数字はすべて本記録に残してある。
