# PDFドキュメントから質問応答するRAGシステム

本プロジェクトは、LangChainとOpenAI APIを用いて、PDFドキュメントを対象とした**RAG（Retrieval-Augmented Generation）システム**を構築したものです。  
Python初心者の学習目的として開発され、検索精度向上のために**ベクトル検索 + 全文検索のハイブリッド検索**および**Reciprocal Rank Fusion（RRF）**を活用しています。

---

## 本プロジェクトの構成

```
.
├── Raggle2.py              # メイン実装ファイル
├── requirements.txt        # 必要なライブラリ
├── .env                    # OpenAI APIキー設定用（作成が必要）
├── pdf_1.pdf ~ pdf_5.pdf   # コンペ対象PDF（URLからのダウンロードを省略し、ローカルパスに変更済み）
```

---

## 注意事項

- 元々はPDFのURLを指定してダウンロード処理をしていましたが、**提出環境では外部アクセスができない想定**のため、`pdf_file_urls` にローカルPDFファイル名（`pdf_1.pdf` など）を直接指定するよう修正しています。
- `.env`ファイルによるAPIキーの読み込み、および依存ライブラリの事前インストールが必要です。

---

## 環境構築手順

### 1. 依存ライブラリのインストール

以下のように、プロジェクトのルートディレクトリでターミナルを開き、`requirements.txt` に記載されたライブラリを一括でインストールします：

```bash
pip install -r requirements.txt
```

### 2. OpenAI APIキーの設定

`.env` ファイルを作成し、以下のように記述してOpenAIのAPIキーを設定してください（`sk-proj-xxxxx` はあなたのAPIキーに置き換えてください）：

```
OPENAI_API_KEY=sk-proj-xxxxx
```

この設定により、`python-dotenv` 経由でAPIキーが自動的に読み込まれます。

---

## 実行方法

ターミナルで以下のようにコマンドを実行することで、任意の質問に対する回答が得られます。

```bash
python Raggle2.py "肌ラボ 極潤ヒアルロン液の詰め替え用には、何mLが入っていますか？"
```

実行結果（例）：

```json
{
  "result": "170mLが入っています。",
  "run_id": "cb8e66a7-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
}
```

---

##  特徴と技術スタック

- LangChainのRunnableインターフェースによる**シンプルかつ再利用可能なRAGパイプライン**
- Chromaによる**意味ベースのベクトル検索**
- BM25による**n-gramベースの全文検索**
- RRF（Reciprocal Rank Fusion）による**ハイブリッドランキング**

---

## 補足

- 本コードは、Raggleコンペティション（https://raggle.jp）で提出した内容をもとに再構成したものです。
- 学習目的のため、冗長・未最適な部分がある可能性があります。改善提案は歓迎です。

---

## Author

- Qiita記事：https://qiita.com/longzhengdeng20/items/9b56cf50b373da7fa328

