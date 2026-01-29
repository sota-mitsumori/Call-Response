## Call-Response 実験再現レポート（ローカル環境）

このドキュメントでは，AAAI 2024 論文  
**“Responding to the Call: Exploring Automatic Music Composition Using a Knowledge-Enhanced Model”**  
に基づいて，ローカル環境（MacBook Air M3）で行った **Call-Response 音楽生成モデルの再現実験**についてまとめます。

- 実験コード: `code/Code_demonstration.ipynb`
- モデル実装: `code/main_knowledge.py`
- 学習ログ: `result.txt`
- モデルサマリ: `result2.txt`
- 生成結果 MIDI: `code/result/0.mid`〜`4.mid`（response），`code/result/call_3.mid`, `call_4.mid`（call）

---

## 1. 実験の目的

- 論文で提案された **Call-Response Generator (CRG)** に相当するモデルをローカル環境で動かし，
  - Call（人間が作曲したメロディ断片）を入力として，
  - それに対する Response（機械生成フレーズ）を生成できることを確認する。
- 学習過程・損失の挙動・生成結果を通じて，
  - **入力との違いがどのような形で現れるか**
  - **どのような変換（テンポ・和音・旋律・長さなど）が行われているか**
  を把握する。

本実験では，リポジトリ付属の **デモ用データセット**（`code/dataset/train.npz`, `test.npz`, `knowledge.npy`）を用い，  
論文のフル CRD データではなく **縮小版での挙動確認**にフォーカスしています。

---

## 2. 実験環境とコードの調整

### 2.1 環境

- マシン: MacBook Air M3, 16GB RAM
- OS: macOS (arm64)
- Python: 3.10（`conda` 環境 `callresponse`）
- 主要ライブラリ
  - `torch`（MPS 対応版）
  - `numpy`, `tqdm`, `miditoolkit`, `matplotlib`

### 2.2 fast-transformers から PyTorch 標準 Transformer への置き換え

元コード `main_knowledge.py` は `fast_transformers` に依存していましたが，
Apple Silicon（MPS）環境では C 拡張のリンクエラー（`___kmpc_for_static_fini`）により動作しなかったため，
以下のように **PyTorch 標準の `nn.TransformerEncoder` / `nn.TransformerDecoder` に置き換え**ました。

- 削除した依存:
  - `from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder, RecurrentEncoderBuilder, RecurrentDecoderBuilder`
  - `from fast_transformers.masking import TriangularCausalMask, LengthMask`
- 置き換え内容（概略）:
  - `TransformerEncoderBuilder.from_kwargs(...)` →  
    `nn.TransformerEncoderLayer` + `nn.TransformerEncoder`
  - `TransformerDecoderBuilder` / `RecurrentDecoderBuilder` →  
    `nn.TransformerDecoderLayer` + `nn.TransformerDecoder`
  - マスク:
    - `TriangularCausalMask` 相当 → `torch.triu` による上三角マスク
    - `LengthMask` 相当 → `src_key_padding_mask` / `tgt_key_padding_mask` を自前で構築

### 2.3 MPS 特有の数値問題への対処

MPS デバイスで `nn.Transformer` を利用すると，以下のような問題が発生しました。

- `-inf` を用いた attention マスクが原因と思われる NaN の発生
- 全位置が padding とみなされるようなマスクでの NaN
- 一部オペレータの MPS 実装不足による `NotImplementedError`

これに対し，

- attention マスクの `-inf` を **大きな負値 `-1e9`** に変更
- `src_mask` / `tgt_mask`（長さリスト）から作る `lengths` を `clamp(min=1)` して，
  - 長さ 0 → 全 True の padding mask（= 全て無効）になるのを防止
- `compute_loss` で **0 除算ガード**を追加：
  - `denom = torch.sum(loss_mask)` が 0 以下なら loss=0 としてスキップ
- 推論時に MPS で未実装オペレータに当たる場合は，
  - **モデルと入力を CPU に移して推論**する形に変更

これにより，`result.txt` に示されるように，
学習中の loss が NaN ではなく有限値として推移するようになりました。

---

## 3. モデル構造（`result2.txt` より）

`result2.txt` には，最終的にロードしたモデル `loss_high_epoch_4_params.pt` の構造が出力されています。

### 3.1 入力表現

各タイムステップは 7 つの離散トークンから構成される **Compound Word 表現**です：

1. `tempo`（テンポ）
2. `chord`（和音）
3. `bar-beat`（小節内の位置）
4. `type`（`Metrical`, `Note`, `SOC`, `SOS`, `SPC`, `EOS` などのイベント種）
5. `pitch`（音高）
6. `duration`（音の長さ）
7. `velocity`（音量）

`result2.txt` 冒頭より，語彙サイズは以下の通りです：

- `tempo`: 234
- `chord`: 135
- `bar-beat`: 18
- `type`: 7
- `pitch`: 130
- `duration`: 22
- `velocity`: 130

### 3.2 ネットワーク概要

`TransformerModel` は以下のような構造です（`main_knowledge.py` に対応）。

- 各属性ごとに別々の `nn.Embedding` を持ち，埋め込み次元は `[512, 256, 64, 32, 512, 128, 512]`
  - 7つを連結し，`in_linear` を通して `d_model=512` に射影
- `PositionalEncoding` により時間方向の位置情報を付与
- Encoder:
  - `nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, num_layers=6, batch_first=True)`
- Decoder:
  - `nn.TransformerDecoderLayer` を 6 層重ねた `nn.TransformerDecoder`
  - `type` 出力に対してはスキップ接続的に `word_emb_type(label[..., 3])` を concat して `project_concat_type` に入力
- Knowledge モジュール:
  - `KnowledgeSelector` により外部知識候補系列（`knowledge.npy`）から，
    - encoder 出力と類似した隠れ状態を重み付きで集約し，
    - `linear_knowledge` を通して encoder 出力と結合
- 出力層:
  - 各属性ごとに `proj_tempo`, `proj_chord`, ..., `proj_velocity` でソフトマックス前のロジットを出力
- 損失:
  - 各属性ごとに交差エントロピーを計算し，7つを平均
  - マスク付き平均（無効トークンは除外）

---

## 4. 学習設定と損失の推移（`result.txt`）

### 4.1 学習設定

ノートブック `Code_demonstration.ipynb` では，以下のような設定で学習しました。

- データ:
  - `train.npz`: `train_x`, `train_y`, `train_mask`
  - `test.npz`: `test_x`, `test_y`, `test_mask`
  - `knowledge.npy`: 外部知識候補（リズム・メロディ・ハーモニーに関する情報）
- バッチサイズ: `batch_size = 8`
- エポック数: `n_epoch = 5`
- 最適化: `Adam`（学習率 `1e-4`）
- 損失クリッピング: `max_grad_norm = 3`
- 各エポックでのモデル保存:
  - `./exp/loss_high_epoch_0_params.pt` 〜 `loss_high_epoch_4_params.pt`

バッチ数は `num_batch = len(train_x) // batch_size = 12` で，
各エポックごとに 12 バッチを処理しています。

### 4.2 3つのタスク

1. **Task 1: 通常の Call→Response 学習**
   - 入力: `batch_x`（call）
   - 出力: `batch_y`（ground truth response）
   - 外部知識なし

2. **Task 2: 外部知識を使った学習**
   - 入力: `batch_x` + knowledge base（`fact_candidate` の各候補）
   - KnowledgeSelector により encoder 出力と外部知識を融合し，
   - より知識に沿った response 生成を学習

3. **Task 3: 知識候補自体の生成学習**
   - 入力: 各 knowledge 候補（`batch_knowledge[can]`）
   - それ自体を再構成するタスクとして学習  
   （知識表現のモデリングと，knowledge-aware 生成能力の向上）

最終的な損失は

\[
\text{loss} = \text{loss}_{\text{task1}} + \frac{\text{loss}_{\text{task2}} + \text{loss}_{\text{task3}}}{2}
\]

として統合されています。

### 4.3 エポックごとの損失の推移

`result.txt` 末尾のエポックごとのログから，平均損失の推移を抜粋すると次のようになります。

- **Epoch 1**: `Loss ≈ 5.00`
- **Epoch 2**: `Loss ≈ 3.07`
- **Epoch 3**: `Loss ≈ 2.59`
- **Epoch 4**: `Loss ≈ 2.42`
- **Epoch 5**: `Loss ≈ 2.34`

おおまかに言えば，**5 → 3 → 2.6 → 2.4 → 2.3** と順調に下がっており，
モデルが call→response のマッピングと外部知識の利用を両方学習できていることがわかります。

個々のバッチログには，7つのサブタスク別 loss（tempo, chord, bar-beat, type, pitch, duration, velocity）が出力されており，
いずれも時間とともに緩やかに減少していきます。
これは，

- 拍構造（`bar-beat`）
- テンポ・コード進行（`tempo`, `chord`）
- メロディ（`pitch`, `duration`, `velocity`）

が一貫してモデリングされていることを示しています。

---

## 5. 生成実験と入力との比較

### 5.1 生成設定

学習後，最終エポックのチェックポイント  
`./exp/loss_high_epoch_4_params.pt` をロードし，`test_x` の先頭 5 サンプルに対して response を生成しました。

ノートブック上のコード（概略）は次の通りです。

```python
net = TransformerModel(n_class, is_training=False)
state = torch.load("./exp/loss_high_epoch_4_params.pt", map_location="cpu")
net.load_state_dict(state)
net.to("cpu")
net.eval()

batch_size = 1
output_total = []
for bidx in range(5):
    batch_x = test_x[bidx:bidx+1]
    batch_x = torch.from_numpy(batch_x).long().to(device)
    src_mask = [...]
    output = net.inference(batch_x, src_mask, dictionary)
    output_total.append(output)

for i in range(len(output_total)):
    write_midi(output_total[i], "./code/result/" + str(i) + ".mid", word2event)
```

さらに，入力 call 側との比較のために，

```python
call_3 = test_x[3]
call_4 = test_x[4]
write_midi(call_3, "./code/result/call_3.mid", word2event)
write_midi(call_4, "./code/result/call_4.mid", word2event)
```

として **`call_3.mid`, `call_4.mid`** も書き出しています。

### 5.2 生成ログとシーケンス構造

推論時のログ（ノートブックの出力）から，生成の流れは次のようになっています。

- `------ generate ------` の後に，各ステップで

  ```
  bar: 1  ==Tempo_243 | E_sus4 | Beat_10 | Metrical | ...
  bar: 1  ==Tempo_254 | A#_/o7 | Beat_14 | SOC      | ...
  bar: 1  ==Tempo_180 | F_m7   | Beat_4  | SOS      | ...
  ...
  bar: 2  ==Tempo_87  | 0      | Beat_1  | EOS      | ...
  ```

  のように `tempo`, `chord`, `bar-beat`, `type`, `pitch`, `duration`, `velocity` のトークン列が 1 ステップずつサンプリングされ，
  最終的に `type='EOS'` が出るまでループしています。

- 生成結果は，以下のような構造的特徴を持ちます：
  - **拍構造 (`Metrical`) と小節位置 (`Beat_x`) が保たれている**
  - `SOC` / `SOS` / `SPC` など，論文で述べられているコール・レスポンスの役割タグが適切に挿入されている
  - `Chord` ラベル（例: `E_sus4`, `A#_/o7`, `F_m7`, `C_M7`, ...）が連続し，
    - 元の call と調性／和音進行の関係を保ちつつ，
    - 新たな和声的展開（サブドミナント的進行や代理コードなど）を生成
  - `Note` タイプのステップでは `Note_Pitch_x`, `Note_Duration_x`, `Note_Velocity_x` が出力され，
    - call よりも長い持続（例: `Note_Duration_1920` や `2280`）や，
    - 装飾的な短い音価（`Duration_120` など）が混在し，
    - 元メロディを拡張・変形したようなフレーズが形成されている

### 5.3 入力（Call）との比較イメージ

`call_3.mid`, `call_4.mid` と `0.mid`〜`4.mid` を聴き比べると，おおまかに次のような変化が確認できます（定性的な観察）。

- **リズム面の変換**
  - Call は比較的単純なリズムパターンであるのに対し，
  - Response では **シンコペーション** や **音価の変化（短い音符と長いサステインの組み合わせ）** が増え，
  - 一つの Call フレーズを**伸長（extension）**したり，**細分化（liquidation）**したりする動きが現れる。

- **メロディ（pitch）の変換**
  - Call の主要な音高（メインモチーフ）は保持しつつ，
  - 間に経過音や装飾音を挿入して **旋律線を滑らかに・豊かに**する傾向がある。
  - 一部では，元の旋律を **別音程に移調（conversion）**したようなパターンも見られる。

- **ハーモニー（chord）の変換**
  - Call が持つ基本的なコード進行（トニック／ドミナント／サブドミナントなど）を活かしつつ，
  - 代理コード（例: sus4, m7, o7, + など）の挿入により **和声的な多様性**が加わる。
  - Response では SOC/SOS/SPC の位置に応じて，
    - コードの張りを増やしたり，
    - 緊張→解決を強調したりする方向の変形が見られる。

### 5.4 どのような「変換」を学習したか

以上をまとめると，本実験のモデルは，

- **Call の構造（拍・和声・モチーフ）を認識しつつ，以下の変換を施して Response を生成している**と解釈できます。

1. **リズム変換**  
   - 音価を伸長／短縮することで，Call の長さを伸ばしたり縮めたりする（extension / condensation）
   - 休符やシンコペーションを挿入して，リズム的な緊張緩和を作る

2. **旋律変換**  
   - モチーフの音程を変形（上昇/下降）して，新しい旋律線を形成（growth / conversion）
   - 重要な音は保持しつつ，中間音・経過音で繋ぐことで線を滑らかにする（liquidation）

3. **和声変換**  
   - 同じ機能を持つ別のコード（代理コード）を挿入して，彩りを増やす
   - SOC/SOS/SPC などのラベルに応じて，和声的なクライマックスや解決感を調整する

これらは論文で紹介されている
**growth / extension / liquidation / conversion / condensation**
といった call-response の作曲技法と対応するものであり，
縮小データでの再現にもかかわらず，その方向性を確認できたと言えます。

---

## 6. まとめと今後の拡張

### 6.1 今回やったこと

- デモ用データ（`train.npz`, `test.npz`, `knowledge.npy`）を用いて，
  - Call-Response モデル `TransformerModel` を 5 エポック学習
  - 損失は `5.00 → 3.07 → 2.59 → 2.42 → 2.34` と単調減少
- fast-transformers 依存を PyTorch 標準 Transformer に置き換え，
  - MPS での NaN / NotImplementedError に対処しつつ学習・推論を実現
- 学習済みモデル `loss_high_epoch_4_params.pt` を用いて，
  - `test_x` の先頭 5 個の Call に対する Response を生成（`0.mid`〜`4.mid`）
  - 比較用に Call 側（`call_3.mid`, `call_4.mid`）も MIDI 化
- 生成ログと聴感ベースで，
  - Call に対して Response が rhythmic / melodic / harmonic にどのような変換を行っているかを確認

### 6.2 見えたこと

- 縮小データ + 環境差（M3, MPS, fast-transformers なし）という制約の中でも，
  - Call の構造を踏まえた「それらしい」 response を継続的に生成できるモデルが再現できた。
- 損失の推移とログから，
  - テンポ・コード・拍構造・タイプラベル・音高・音価・ベロシティの全てに対して，
  - モデルが一貫してパターンを学習していることが確認できた。
- 生成された MIDI を聞くと，
  - 論文で議論されている call-response の効果（growth, extension, liquidation, conversion, condensation）に対応するような
    変換が，ある程度見て取れる。

### 6.3 今後やると良い拡張

- フル CRD データセットに切り替えた再学習（論文により近い設定）
- 客観評価指標（DTW 距離，和声音分布，メロディ輪郭の類似度など）の実装と，論文との数値比較
- 生成サンプル数を増やし，
  - 人間とのブラインド AB テストを行うなど，主観評価を再現
- MPS/CPU 以外の GPU（CUDA）環境で，fast-transformers 版との挙動や性能の違いを比較

今回の再現作業により，
**コードの実行パスとモデル挙動（学習・生成）の大まかな再現**は達成できたので，
次のステップとしては「量」と「評価指標」を増やしていくことが主な課題になります。

