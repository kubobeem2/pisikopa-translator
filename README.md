# Pisikopa Translator v0.2

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
python3 translator.py en "I might want to eat an apple"
python3 translator.py en "Did I eat an apple?"
python3 translator.py en "Eat an apple!"
python3 translator.py pi "ka pa ra-ta"
python3 translator.py pi "ka pa ra-ta lo"
python3 translator.py pi "ka ra rɔ"
python3 translator.py vec banana
python3 test_translator.py   # 30ケース
python3 gen_corpus.py        # corpus.tsv を再生成
```

## テスト結果（v0.2、2026-11-05）
`python3 test_translator.py` の結果: **30 passed, 0 failed**。
コーパス照合: `corpus.tsv` 1114ペアに対し `en_to_pi` が **1114/1114 完全一致**。
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
- ka pa ra-ta lo -> `Did I eat apple? (saw)`（PI->EN）
- ka ra rɔ -> `Eat apple!`

v0.1 からの修正（v0.1では通らなかったもの）:
- PI疑問文: v0.1 は `Did i eat apple (saw)?` だった。v0.2 は `Did I eat apple? (saw)`。
- PI命令: v0.1 は `eat make apple (saw)` だった。v0.2 は `Eat apple!`。
- EN命令: v0.1 は `ra ka-ta` だった。v0.2 は `ka ra rɔ`。
- EN願望: v0.1 は目的語が消えて `pa məra-ta` になった。v0.2 は `ka pa məra-ta`。
- EN伝聞 "I heard that I eat an apple": v0.1 は `ra pa pa-to` だった。v0.2 は `ka pa ra-to`。
- EN埋め込み "I saw that he eats an apple": v0.1 は英語がそのまま混ざった。v0.2 は `ka pu ra-ta læ pa ro-ta`。
- コーパスの ti/tu/tæ テンプレート（"I heard the sound of ...", "I felt ...", 引用文）は
  v0.2 でパーサーに追加し、コーパス照合は 846/1114 から 1114/1114 になった。

通らなかった例（正直な記録）:
- なし（テスト30ケース、コーパス1114ペアとも全通）。ただし対象はテンプレート文のみ。
  複雑な自由文（関係節・複文・未知語だらけの文）は対象外で、通らないことがある。
  例: "I saw the man who ate an apple yesterday" のような関係節は未対応。
  PI->EN の "What did I eat ? (saw)" は `?` の前に半角スペースが入る軽微な表記ゆれがある。
  コーパスの ti/tu 英語（"I heard the sound of me eating an apple." / "I felt me eating an apple."）
  は、テンプレートとして規則的だが自然な英語ではない。

## コーパス（corpus.tsv、1114ペア）
`gen_corpus.py` でテンプレート生成。内訳:
- transitive 640、intransitive 128、negation 48、question-yesno 48、question-wh 12、
  imperative 20、want 96、time 96、embedding-saw-that 18、embedding-gerund 8。
- 証拠性マーカー（英語側で区別）:
  ta=plain、ti="I heard the sound of ... -ing"、tu="I felt ... -ing"、
  te="Maybe / might"、to="I heard that ..."、tə="In a dream, ..."、
  tæ='"...," someone said.'、tɔ="Everyone knows ..."
- 一部の組み合わせは意味的におかしい（"I eat a book." など）が、文法の規則性を
  学ばせるためのテンプレートで、意図的なもの。

## 学習モデル（実験、TRAINING.md 参照）
- `train_pi.py` / `generate_pi.py` / `pisikopa.modoki` は lmmもどき型の
  tiny transformer（numpyのみ、単語レベル、d_model=48・2層・約60kパラメータ、
  語彙201トークン）を `corpus.tsv` 1114ペアで学習させたもの。
- loss: 初期 5.958 -> 800ステップで 0.682、2500ステップで **0.662**（約91秒）。
  2500ステップ版を採用。
- 生成（貪欲生成、2500ステップ版）: 学習データ1114プロンプトの完全一致率は
  800ステップ版 1009/1114 (90.6%) -> **2500ステップ版 1114/1114 (100%)**。
  成功例: `I eat an apple. =` -> `ka pa ra-ta`、
  `I might want to eat an apple. =` -> `ka pa məra-te`。
- 失敗例（正直に）: `we drink a banana. =` -> `pe ro-ti`（正解 `kasæ pe ri-ta`）、
  `In a dream, we see a book. =` -> `kæ pe ro-ta`（正解 `kæ pe ro-tə`、証拠性違い）、
  `You see the sea. =` / `He reads a book.` は語彙外（OOV）でエラー。
  未学習の組み合わせで banana を含むものは 0/20（`kasæ` の代わりに `kæ` を出す）。
  **丸暗記は100%だが汎化は弱い。ルール翻訳が主、学習モデルは実験。**
- モデルファイル: `pisikopa.modoki` 593,641バイト（約0.57MB）。

## ファイル
- `translator.py` — ルール翻訳 EN<->PI + 16次元ベクトル最近傍
- `lexicon_en.json` — 英語145語 -> Pisikopa
- `corpus.tsv` — 並行コーパス 1114ペア
- `gen_corpus.py` — コーパス生成（重複・衝突チェック付き）
- `test_translator.py` — テスト30ケース
- `train_pi.py` / `generate_pi.py` / `pisikopa.modoki` / `TRAINING.md` — 学習実験
