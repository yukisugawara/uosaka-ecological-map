"""
大阪大学 Ecological Map — Step 2: トピックモデル＋ネットワーク構築
収集データからBERTopicでトピック分類し、共著・共同研究ネットワークを構築
"""

import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_details():
    path = os.path.join(DATA_DIR, "researcher_details.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text(item, field_name):
    """多言語フィールドからテキストを取り出す"""
    val = item.get(field_name)
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("ja", val.get("en", ""))
    return ""


def build_researcher_corpus(details):
    """各研究者のテキストコーパスを構築（トピックモデル用）"""
    corpus = []
    researcher_ids = []

    for d in details:
        permalink = d.get("permalink", "")
        if not permalink:
            continue

        texts = []

        # 研究分野
        osaka = d.get("osaka_info", {})
        rf = osaka.get("research_field", [])
        if isinstance(rf, str):
            rf = [rf]
        texts.extend(rf)

        # 論文タイトル
        for paper in d.get("published_papers", []):
            title = extract_text(paper, "paper_title")
            if title:
                texts.append(title)

        # 著書タイトル
        for book in d.get("books_etc", []):
            title = extract_text(book, "book_title")
            if title:
                texts.append(title)

        # 研究発表タイトル
        for pres in d.get("presentations", []):
            title = extract_text(pres, "presentation_title")
            if title:
                texts.append(title)

        # 研究プロジェクトタイトル
        for proj in d.get("research_projects", []):
            title = extract_text(proj, "research_project_title")
            if title:
                texts.append(title)

        combined = " ".join(texts)
        if combined.strip():
            corpus.append(combined)
            researcher_ids.append(permalink)

    return researcher_ids, corpus


def _normalize_name(name):
    """名前を正規化（マッチング精度向上）"""
    import unicodedata
    s = name.replace("　", " ").replace(",", " ").replace(".", " ")
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.split()).strip()


def _build_name_index(details):
    """多様な名前変形からpermalinkへの逆引き辞書を構築"""
    name_to_pl = {}
    pl_to_names = {}

    for d in details:
        pl = d.get("permalink", "")
        if not pl:
            continue
        osaka = d.get("osaka_info", {})
        name_ja = osaka.get("name_ja", osaka.get("name", ""))
        name_en = osaka.get("name_en", "")

        variants = set()
        for raw in [name_ja, name_en]:
            if not raw:
                continue
            norm = _normalize_name(raw)
            # フルネーム（スペースなし）
            variants.add(norm.replace(" ", ""))
            # フルネーム（スペースあり）
            variants.add(norm)
            # 姓名逆転
            parts = norm.split()
            if len(parts) == 2:
                variants.add(f"{parts[1]}{parts[0]}")
                variants.add(f"{parts[1]} {parts[0]}")

        # researchmapのプロフィールから取得した名前
        profile = d.get("profile", {})
        for key in ["family_name", "given_name"]:
            fn = profile.get(key, {})
            if isinstance(fn, dict):
                for lang_val in fn.values():
                    if isinstance(lang_val, str) and lang_val:
                        variants.add(lang_val.strip())
        # 姓+名の組み合わせ
        fam = profile.get("family_name", {})
        giv = profile.get("given_name", {})
        if isinstance(fam, dict) and isinstance(giv, dict):
            for lang in ["ja", "en"]:
                f = fam.get(lang, "")
                g = giv.get(lang, "")
                if f and g:
                    variants.add(f"{f}{g}")
                    variants.add(f"{f} {g}")
                    variants.add(f"{g}{f}")
                    variants.add(f"{g} {f}")
                    # 姓のみ（同姓がいなければ後で使う）
                    variants.add(f"_surname_{f}")

        pl_to_names[pl] = variants
        for v in variants:
            if v and not v.startswith("_surname_"):
                if v not in name_to_pl:
                    name_to_pl[v] = pl
                # 衝突時は登録しない（曖昧さ回避）

    # 姓のみマッチング（ユニークな姓のみ）
    surname_counts = defaultdict(list)
    for pl, names in pl_to_names.items():
        for n in names:
            if n.startswith("_surname_"):
                surname_counts[n[9:]].append(pl)
    for surname, pls in surname_counts.items():
        if len(pls) == 1 and surname not in name_to_pl:
            name_to_pl[surname] = pls[0]

    return name_to_pl


def build_coauthor_network(details):
    """共著・共同研究者ネットワークを構築（改善版）"""
    G = nx.Graph()

    name_to_pl = _build_name_index(details)

    # ノード追加
    for d in details:
        pl = d.get("permalink", "")
        if not pl:
            continue
        osaka = d.get("osaka_info", {})
        G.add_node(pl,
                    name=osaka.get("name_ja", ""),
                    name_en=osaka.get("name_en", ""),
                    org=osaka.get("org_name", ""),
                    org_lv1=osaka.get("org_name_lv1", ""),
                    job=osaka.get("job_name", ""),
                    research_field=osaka.get("research_field", ""))

    def _match_name(raw_name):
        """名前文字列からpermalinkを検索"""
        norm = _normalize_name(raw_name)
        nospace = norm.replace(" ", "")
        # 完全一致
        if nospace in name_to_pl:
            return name_to_pl[nospace]
        if norm in name_to_pl:
            return name_to_pl[norm]
        # 姓名逆転
        parts = norm.split()
        if len(parts) == 2:
            rev = f"{parts[1]}{parts[0]}"
            if rev in name_to_pl:
                return name_to_pl[rev]
        return None

    # 共著関係からエッジ構築
    for d in details:
        pl = d.get("permalink", "")
        if not pl:
            continue

        coauthors = defaultdict(int)

        # 論文の共著者
        for paper in d.get("published_papers", []):
            authors_data = paper.get("authors", {})
            for lang in ["ja", "en"]:
                for a in authors_data.get(lang, []):
                    if not isinstance(a, dict):
                        continue
                    aname = a.get("name", "")
                    if not aname:
                        continue
                    matched_pl = _match_name(aname)
                    if matched_pl and matched_pl != pl:
                        coauthors[matched_pl] += 1

        # 研究プロジェクトの共同研究者
        for proj in d.get("research_projects", []):
            investigators = proj.get("investigators", {})
            for lang in ["ja", "en"]:
                for inv in investigators.get(lang, []):
                    if not isinstance(inv, dict):
                        continue
                    iname = inv.get("name", "")
                    if not iname:
                        continue
                    matched_pl = _match_name(iname)
                    if matched_pl and matched_pl != pl:
                        coauthors[matched_pl] += 1

        # エッジ追加（重み = 共同出現回数）
        for coauthor_pl, weight in coauthors.items():
            if G.has_edge(pl, coauthor_pl):
                G[pl][coauthor_pl]["weight"] += weight
            else:
                G.add_edge(pl, coauthor_pl, weight=weight)

    return G


def run_topic_model(researcher_ids, corpus, n_topics=30):
    """トピックモデルを実行（BERTopicまたはTF-IDFフォールバック）"""
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        print("BERTopicでトピックモデリング中...")
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=n_topics,
            language="multilingual",
            verbose=True,
        )
        topics, probs = topic_model.fit_transform(corpus)

        # トピック情報の取得
        topic_info = topic_model.get_topic_info()
        topic_labels = {}
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                continue
            topic_labels[tid] = row["Name"]

        # 研究者→トピック割り当て
        researcher_topics = {}
        for rid, topic in zip(researcher_ids, topics):
            researcher_topics[rid] = int(topic)

        return researcher_topics, topic_labels, topic_model

    except ImportError:
        print("BERTopicが利用不可。TF-IDFフォールバックを使用...")
        return _tfidf_fallback(researcher_ids, corpus, n_topics)


def _tfidf_fallback(researcher_ids, corpus, n_clusters=30):
    """BERTopicが使えない場合のフォールバック"""
    from sklearn.cluster import KMeans

    vectorizer = TfidfVectorizer(max_features=5000, max_df=0.8, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    km = KMeans(n_clusters=min(n_clusters, len(corpus)), random_state=42, n_init=10)
    labels = km.fit_predict(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topic_labels = {}
    for i in range(km.n_clusters):
        center = km.cluster_centers_[i]
        top_indices = center.argsort()[-5:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        topic_labels[i] = f"Topic_{i}: {', '.join(top_words)}"

    researcher_topics = {rid: int(lbl) for rid, lbl in zip(researcher_ids, labels)}
    return researcher_topics, topic_labels, None


def add_topic_similarity_edges(G, researcher_ids, corpus, threshold=0.15):
    """トピック類似度に基づくエッジを追加（TF-IDFコサイン類似度）"""
    print("トピック類似度エッジを計算中...")

    vectorizer = TfidfVectorizer(max_features=5000, max_df=0.8, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # メモリ節約のため、バッチでコサイン類似度を計算
    n = tfidf_matrix.shape[0]
    edges_added = 0
    batch_size = 500

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        sim_batch = cosine_similarity(tfidf_matrix[i:end_i], tfidf_matrix)

        for local_idx in range(end_i - i):
            global_i = i + local_idx
            for j in range(global_i + 1, n):
                if sim_batch[local_idx, j] > threshold:
                    ri = researcher_ids[global_i]
                    rj = researcher_ids[j]
                    if ri in G and rj in G:
                        if G.has_edge(ri, rj):
                            G[ri][rj]["topic_sim"] = float(sim_batch[local_idx, j])
                        else:
                            G.add_edge(ri, rj,
                                       weight=0,
                                       topic_sim=float(sim_batch[local_idx, j]))
                            edges_added += 1

    print(f"  類似度エッジ追加: {edges_added}本")
    return G


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # データ読み込み
    print("データ読み込み中...")
    details = load_details()
    print(f"  研究者: {len(details)}名")

    # コーパス構築
    print("コーパス構築中...")
    researcher_ids, corpus = build_researcher_corpus(details)
    print(f"  テキストあり: {len(corpus)}名")

    # 共著ネットワーク構築
    print("共著ネットワーク構築中...")
    G = build_coauthor_network(details)
    print(f"  ノード: {G.number_of_nodes()}, エッジ: {G.number_of_edges()}")

    # トピックモデル
    researcher_topics, topic_labels, topic_model = run_topic_model(
        researcher_ids, corpus
    )

    # トピック情報をノードに追加
    for rid, topic in researcher_topics.items():
        if rid in G:
            G.nodes[rid]["topic"] = topic
            G.nodes[rid]["topic_label"] = topic_labels.get(topic, f"Topic_{topic}")

    # トピック類似度エッジ追加
    G = add_topic_similarity_edges(G, researcher_ids, corpus, threshold=0.15)

    # 中心性指標の計算
    print("中心性指標を計算中...")
    pagerank = nx.pagerank(G, weight="weight")
    betweenness = nx.betweenness_centrality(G, weight="weight", k=min(500, G.number_of_nodes()))
    for node in G.nodes():
        G.nodes[node]["pagerank"] = pagerank.get(node, 0)
        G.nodes[node]["betweenness"] = betweenness.get(node, 0)
    print(f"  PageRank, 媒介中心性を計算完了")

    # リスト型属性を文字列に変換（GEXF互換性のため）
    for n in G.nodes():
        for key, val in list(G.nodes[n].items()):
            if isinstance(val, list):
                G.nodes[n][key] = ", ".join(str(v) for v in val)

    # 保存
    # NetworkX GEXF形式
    gexf_path = os.path.join(OUTPUT_DIR, "osaka_researcher_network.gexf")
    nx.write_gexf(G, gexf_path)
    print(f"\nGEXF保存: {gexf_path}")

    # トピック情報をJSON保存
    topic_data = {
        "topic_labels": {str(k): v for k, v in topic_labels.items()},
        "researcher_topics": researcher_topics,
    }
    topic_path = os.path.join(OUTPUT_DIR, "topics.json")
    with open(topic_path, "w", encoding="utf-8") as f:
        json.dump(topic_data, f, ensure_ascii=False, indent=2)
    print(f"トピック保存: {topic_path}")

    # ネットワーク統計
    print(f"\n=== ネットワーク統計 ===")
    print(f"  ノード数: {G.number_of_nodes()}")
    print(f"  エッジ数: {G.number_of_edges()}")
    if G.number_of_nodes() > 0:
        degrees = [d for _, d in G.degree()]
        print(f"  平均次数: {np.mean(degrees):.2f}")
        components = list(nx.connected_components(G))
        print(f"  連結成分数: {len(components)}")
        if components:
            print(f"  最大連結成分: {len(max(components, key=len))}ノード")

    # トピック統計
    print(f"\n=== トピック統計 ===")
    print(f"  トピック数: {len(topic_labels)}")
    topic_counts = defaultdict(int)
    for t in researcher_topics.values():
        topic_counts[t] += 1
    for tid, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
        label = topic_labels.get(tid, f"Topic_{tid}")
        print(f"  {label}: {count}名")


if __name__ == "__main__":
    main()
