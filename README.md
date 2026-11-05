# Pisikopa Translator v0.3

英語 <-> Pisikopa の翻訳機。**ルール翻訳が主、学習モデルは実験。**

Pisikopa: 64音節、OSV、時制なし、証拠性8種必須
（ta目撃 / ti聴覚 / tu体感 / te推測・かも / to伝聞 / tə夢 / tæ引用 / tɔ常識）、
否定は動詞前に機能語 mæ、疑問は証拠性後に lo、命令は主語・証拠性なしで動詞後に rɔ、
願望は mə+動詞、埋め込みは læ（こと）で名詞化。仕様は `~/workspace/conlang/SPEC.md`。

## 仕組み（2段）
1. **ルール翻訳（正確な方）**: SVO -> OSV + 証拠性語尾。Pisikopaは文法が規則的なので、単純文はこれでいける。
2. **ベクトル翻訳（lmm風）**: 各Pisikopa語根に16次元ベクトルを付ける。
   - 前半8次元 = 子音の分野 one-hot、後半8次元 = 母音 one-hot。
   - 英語も `lexicon_en.json` 経由で同じ空間を共有する。
   - 未知語はコサイン類似度で一番近いPisikopa語に落とす。例: banana -> kasæ
     （ka果物 + sæ黄色の平均ベクトルに近い）。ただし `translator.py vec`
     の既知語は lexicon 一致で score 1.0 を返し、未知語は
     cosine-compound / string-overlap-fallback として区別して返す。

## 使い方
```sh
python3 translator.py en "I eat an apple"
python3 translator.py en "You see the sea."
python3 translator.py en "He reads a book."
python3 translator.py en "I might want to eat an apple"
python3 translator.py en "Did I eat an apple?"
python3 translator.py en "Eat an apple!"
python3 translator.py pi "ka pa ra-ta"
python3 translator.py pi "ka pa ra-ta lo"
python3 translator.py pi "ka ra rɔ"
python3 translator.py vec banana
python3 test_translator.py   # 37ケース
python3 gen_corpus.py        # corpus.tsv + holdout.tsv を再生成
python3 eval_model.py --model pisikopa.modoki   # ホールドアウト / banana20 / 学習サンプル
```

## 入力正規化（v0.3）
英語入力は `translator.normalize_en` で正規化される（`translator.py` /
`generate_pi.py` 共用）。小文字化、句読点除去（`?` / `!` は別トークンとして残す）、
短縮形の展開、三人称単数 → 原形。コーパスも同じ正規化済み形式で保存してある。
- `You see the sea.` → `you see the sea` → `si pi ro-ta`
- `He reads a book.` → `he read a book` → `kæ pu kæro-ta`
- `She doesn't eat a banana.` → `she does not eat a banana` → `kasæ pu mæ ra-ta`
- 正規化後も語彙にない単語は生成不能。`generate_pi.py` は未知語を一覧にして
  エラー終了する（例: `flibberty gibbet` → `語彙にない単語: ['flibberty', 'gibbet']`）。
  未知語のフォールバックはルール翻訳側の `translator.py vec` のみ。
  形容詞も未対応（`I see a red car.` → `ruko pa ro-ta` と `red` を落とす）。

## テスト結果（v0.3、2026-11-05〜06）
`python3 test_translator.py` の結果: **37 passed, 0 failed**。
コーパス照合: `corpus.tsv` 6172ペア + `holdout.tsv` 48ペアに対し
`en_to_pi` が **6220/6220 完全一致**。
`gen_corpus.py` の重複・衝突チェック: **collisions 0**。

通った例:
- I eat an apple -> `ka pa ra-ta`
- I eat an apple yesterday -> `ka lu pa ra-ta`
- I might want to eat an apple -> `ka pa məra-te`
- I did not eat an apple -> `ka pa mæ ra-ta`
- Did I eat an apple? -> `ka pa ra-ta lo`
- What did I eat? -> `lɔ pa ra-ta lo`
- Who ate an apple? -> `ka pu ra-ta lo`
- Eat an apple! -> `ka ra rɔ`
- I heard that I eat an apple -> `ka pa ra-to`
- In a dream, I eat an apple -> `ka pa ra-tə`
- Everyone knows I eat an apple -> `ka pa ra-tɔ`
- I saw that he eats an apple -> `ka pu ra-ta læ pa ro-ta`
- I saw him stepping on a banana and crying -> `kasæ pu moru-ta pu məso-ta læ pa ro-ta`
- You see the sea. -> `si pi ro-ta`（v0.3、正規化）
- He reads a book. -> `kæ pu kæro-ta`（v0.3、正規化）
- She sells a banana. -> `kasæ pu kuro-ta`（v0.3、ホールドアウトの組み合わせ）
- ka pa ra-ta lo -> `Did I eat apple? (saw)`（PI->EN）
- ka ra rɔ -> `Eat apple!`

v0.3 での修正（v0.2では通らなかった・なかったもの）:
- `waking` が GERUND になく `paro`（王）と誤訳されていた問題を修正。
- `wake !` が命令文と判定されなかった問題を修正。
- `Who ate an apple?` はコーパスには入れていない。yes/no 疑問文
  `Did he eat an apple?` と同じ PI 文（`ka pu ra-ta lo`）になる
  言語仕様上の本物の曖昧さのため。テストのみに残してある。

通らなかった例（正直な記録）:
- なし（テスト37ケース、コーパス6220ペアとも全通）。ただし対象はテンプレート文のみ。
  複雑な自由文（関係節・形容詞の修飾・未知語だらけの文）は対象外で、通らないことがある。
  例: "I saw the man who ate an apple yesterday" のような関係節は未対応。
  PI->EN の "What did I eat ? (saw)" は `?` の前に半角スペースが入る軽微な表記ゆれがある。
  コーパスの ti/tu 英語（"i heard the sound of me eating an apple" / "i felt me eating an apple"）
  は、テンプレートとして規則的だが自然な英語ではない。

## コーパス（corpus.tsv 6172ペア + holdout.tsv 48ペア、v0.3）
`gen_corpus.py` でテンプレート生成。内訳:
- transitive-core 3024、transitive-extended 1464、intransitive 288、
  negation 192、question-yesno 192、question-wh 24、imperative 64、
  imperative-intrans 6、want 384、time 384、embedding-saw-that 36、
  embedding-gerund 18、see-colour-number 96。
- 他動詞8種（eat/drink/see/read/buy/sell/write/make）、自動詞6種
  （go/come/sleep/wake/run/swim）、主語6種（i/you/he/she/we/they）、
  目的語は lexicon の名詞69語すべて。証拠性8種すべて。
- ホールドアウト: kasæ（banana）+ sell の組み合わせ48ペア
  （主語6 × 証拠性8）を学習から除外し `holdout.tsv` に分離。
  学習データの kasæ は494回（うち直接目的語462回）。
  v0.2 では kasæ は4回のみ・直接目的語0回だった。
- 証拠性マーカー（英語側で区別）:
  ta=plain、ti="i heard the sound of ... -ing"、tu="i felt ... -ing"、
  te="maybe / might"、to="i heard that ..."、tə="in a dream ..."、
  tæ="... someone said"、tɔ="everyone knows ..."
- 一部の組み合わせは意味的におかしい（"I eat a book." など）が、文法の規則性を
  学ばせるためのテンプレートで、意図的なもの。

## 学習モデル（実験、TRAINING.md 参照）
- `train_pi.py` / `generate_pi.py` / `pisikopa.modoki` は lmmもどき型の
  tiny transformer（numpyのみ、単語レベル、d_model=64・3層・155,208パラメータ、
  語彙392トークン）を `corpus.tsv` 6172ペアで学習させたもの。
- loss: 初期 6.248 -> 5000ステップで **0.840**（約7.5分）。
  v0.2（0.662）より高いのはコーパス5.5倍・語彙約2倍でタスクが難しいためで、
  loss の単純比較はできない。
- 生成（貪欲生成）: ホールドアウト kasæ+sell **48/48 = 100%**
  （v0.2モデルは 0/32、全件OOV）、banana20 **20/20 = 100%**
  （v0.2モデルは 1/20）、学習データ全件 **6168/6172 = 99.9%**。
  d=96・3層・5000ステップ版と d=64・3層・8000ステップ版も試したが、
  ホールドアウトは同じ100%で学習データの正解率はむしろ悪く、
  8000ステップ版は 98.5% まで落ちた。採用は d=64・5000ステップ版。
- 成功例: `You see the sea. =` -> `si pi ro-ta`、
  `She sells a banana. =` -> `kasæ pu kuro-ta`、
  `I saw that they sell a banana. =` -> `kasæ po kuro-ta læ pa ro-ta`。
- 失敗例（正直に）: 目的語なし命令文 4件のみ。
  `run !` -> `siru rɔ`（正解 `ruru rɔ`）、`wake !` -> `siru rɔ`、
  `come !` -> `pu re-ta`（正解 `re rɔ`）、`sleep !` -> `siru rɔ`。
  このカテゴリはコーパスに6行しかない。`I see a red car.` は
  `ruko pa ro-ta` と形容詞を落とす。未知語はエラー終了。
  **テンプレート内の汎化は100%だが、テンプレート外の自由文は
  相変わらず扱えない。ルール翻訳が主、学習モデルは実験。**
- モデルファイル: `pisikopa.modoki` 1,507,102バイト（約1.44MB）。

## ファイル
- `translator.py` — ルール翻訳 EN<->PI + 16次元ベクトル最近傍 + `normalize_en`
- `lexicon_en.json` — 英語145語 -> Pisikopa
- `corpus.tsv` — 並行コーパス 6172ペア（正規化済み英語）
- `holdout.tsv` — ホールドアウト 48ペア（kasæ+sell）
- `gen_corpus.py` — コーパス生成（重複・衝突チェック付き、ホールドアウト分離つき）
- `test_translator.py` — テスト37ケース
- `train_pi.py` / `generate_pi.py` / `eval_model.py` / `pisikopa.modoki` / `TRAINING.md` — 学習実験
