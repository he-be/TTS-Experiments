# Gradioで「バッチ並列推論」が動くようになった理由（事実のみ）

## 1. 入口：Gradioが「1回の呼び出し」でバッチを要求するようになった

- `inference_gradio.py` の `run_inference()` は、`batch_count > 1` かつ `model.inference_tts_batch` が存在する場合に `sample_batch_size = batch_count` を `decode_config` に入れます（`inference_gradio.py:206-262`）。
- この条件を満たすと、`inference_one_sample()` を **1回だけ** 呼び、その返り値（複数サンプル）を Gradio の複数出力に展開します（`inference_gradio.py:265-302`）。
- `model.inference_tts_batch` が無い場合は、従来通り seed を変えながら `inference_one_sample()` を `batch_count` 回ループします（`inference_gradio.py:302-344`）。

## 2. モデル側：`inference_tts_batch` が無いチェックポイントでも使えるようにした

- HFチェックポイントを `trust_remote_code=True` でロードしたあと、モデルに `inference_tts_batch` が無ければ、このリポジトリの実装（`hf_export/modeling_t5gemma_voice.py`）を **メソッドとして後付け**します（`inference_gradio.py:101-122`）。
  - `torch.compile` 済みの場合は、`model._orig_mod` に付与し、外側の `model` にも必要なら付与します（同箇所）。

## 3. コア：`inference_one_sample()` がバッチAPIを呼び、バッチ出力を処理するようになった

- `inference_one_sample()` は `decode_config["sample_batch_size"] > 1` かつ `model.inference_tts_batch` がある場合に `model.inference_tts_batch(...)` を呼びます（`inference_tts_utils.py:412-451`）。
- `inference_tts_batch` は **テンソルのバッチ次元 `num_samples`** を使って計算します：
  - エンコーダ出力 `memory_single` を `memory_single.expand(num_samples, -1, -1)` でバッチ化（`hf_export/modeling_t5gemma_voice.py:911-917`）。
  - プロンプト `y` を `expand(num_samples, ...)` でバッチ化（`hf_export/modeling_t5gemma_voice.py:933-939`）。
  - 以後のデコーダは `last_hidden` が `[B, 1, hidden]`、logits が `[B, vocab]` で進み、`torch.multinomial(probs, ...)` が **バッチ分の次トークン**を同時にサンプルします（`hf_export/modeling_t5gemma_voice.py:1001-1021`）。
  - 生成トークンは `generated_buffer[active_mask, cur_num_gen] = tokens[active_mask]` のようにバッチで格納します（`hf_export/modeling_t5gemma_voice.py:1041-1043`）。
  - 最後にサンプルごとの可変長を `gen_lengths` で切り出し、`List[Tensor]` として返します（`hf_export/modeling_t5gemma_voice.py:1084-1105`）。
- `inference_one_sample()` 側は、`concat_frames` / `gen_frames` が **list/tuple で返るケース**に対応し、各サンプルを個別に `audio_tokenizer.decode()` して `concat_samples` / `gen_samples` のリストを返します（`inference_tts_utils.py:523-561`）。

## 4. 「並列」の意味（この実装が行っていること）

- 同一プロセス・単一GPU上で、デコーダの各ステップ計算を **バッチ次元（B=num_samples）** で実行し、1回の生成ループで複数サンプルの系列を同時に進めます（`hf_export/modeling_t5gemma_voice.py:1001-1083`）。
- 生成ステップ自体（時間方向のループ）は `while active_mask.any():` で継続します（`hf_export/modeling_t5gemma_voice.py:1001`）。これは「複数系列を同じループで進める」実装であり、プロセス分割や複数GPU分散は行っていません。

## 5. なぜ高速化できたのか（コード上の事実）

### 5.1 呼び出し回数が変わった

- **従来（シーケンシャル）**: `batch_count=n` のとき、`run_inference()` が `inference_one_sample()` を `n` 回呼びます（`inference_gradio.py:291-331`）。各回で `model.inference_tts(...)` が1回呼ばれ、エンコーダ＋デコーダの推論が `n` 回行われます（`inference_tts_utils.py:325-343`）。
- **現在（バッチ）**: `batch_count=n` かつ `model.inference_tts_batch` があるとき、`run_inference()` は `inference_one_sample()` を **1回だけ**呼びます（`inference_gradio.py:253-275`）。この1回の中で `model.inference_tts_batch(..., num_samples=n)` が1回呼ばれます（`inference_tts_utils.py:307-325`）。

### 5.2 エンコーダ計算が「1回にまとまった」

- `inference_tts_batch()` はエンコーダを **1回だけ**実行し（`hf_export/modeling_t5gemma_voice.py:890-912`）、得られた `memory_single` を `expand(num_samples, ...)` でバッチに拡張します（`hf_export/modeling_t5gemma_voice.py:913-917`）。
- シーケンシャル経路では、`model.inference_tts(...)` が呼ばれるたびにエンコーダが実行されます（`hf_export/modeling_t5gemma_voice.py` の `inference_tts()` はバッチサイズ1前提で動作）。

### 5.3 Python側の「繰り返し処理」が減った

- シーケンシャル経路では、Pythonが `for i in range(batch_count)` の外側ループを持ち、推論呼び出し・後処理を `n` 回繰り返します（`inference_gradio.py:291-331`）。
- バッチ経路では外側ループがなく、推論は1回で、返ってきた `n` 個の音声を出力に展開する処理だけがループします（`inference_gradio.py:277-290`）。
- `inference_tts_batch()` 内でも、生成トークンの格納が `generated_buffer[active_mask, cur_num_gen] = ...` のようにバッチ更新になっており、サンプル数ぶんのPythonループを回さない実装になっています（`hf_export/modeling_t5gemma_voice.py:1041-1043`）。

### 5.4 実行結果（観測ログ）

- あなたのコンソールログでは、バッチ前は `batch_count=4` で各サンプルが約7.5〜7.9秒かかり（合計約30秒）、バッチ後は `Generated 1204 tokens across 4 samples in 8.03s` と表示され、4サンプルが約8秒で完了しています。

## 6. 「およそ1/nになる」ことについて（この実装から言える範囲）

- この実装は「`n` 回の推論呼び出し」を「1回の推論呼び出し」にまとめ、エンコーダ計算とPython側の繰り返しを削減します（上記 5.1〜5.3）。
- 一方でデコーダの生成ループ自体は1回の `while ...` の中で進み（`hf_export/modeling_t5gemma_voice.py:1001`）、その各ステップではバッチサイズ `n` のテンソルを扱います（例: logits は `[B, vocab]`、`B=num_samples`）（`hf_export/modeling_t5gemma_voice.py:1002-1021`）。
- したがって、理論上の所要時間が常に厳密に `1/n` になる、とこのリポジトリのコードだけから一般化はできません。ただし、あなたの実行ログでは「推論呼び出し回数削減＋エンコーダ1回化＋Python反復削減」によって大きな短縮が観測されています。

## 7. 付録：文分割（セグメント並列）について（事実のみ）

- `inference_gradio.py` の `run_inference_segmented()` は、`segment_text_by_sentences()` で文分割し（`inference_gradio.py:381`、実装は `inference_tts_utils.py`）、セグメントごとに推論します。
- モデルに `inference_tts_batch_multi_text` がある場合は、**セグメントごとに異なるテキストを1つのバッチ入力**として `inference_tts_batch_multi_text(...)` を呼びます（`inference_gradio.py:516-536`）。
- `inference_tts_batch_multi_text` が無い場合は、セグメントごとに `inference_one_sample(...)` を呼ぶ直列処理にフォールバックします（`inference_gradio.py:536-576`）。
- `inference_tts_batch_multi_text` は、このリポジトリのHF wrapper に実装されており（`hf_export/modeling_t5gemma_voice.py`）、必要に応じて `inference_tts_batch` と同様にモデルへ自動パッチされます（`inference_gradio.py:116-122`）。
- セグメント結合時は無音を挿入し、`--inter_segment_silence`（デフォルト `0.05` 秒）で長さを調整できます。結合前に各セグメントの前後無音（および結合後の全体の前後無音）をトリムします（`inference_tts_utils.py`）。
