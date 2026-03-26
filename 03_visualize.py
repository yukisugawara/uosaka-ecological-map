"""
大阪大学 Ecological Map — Step 3: JSON データ生成 + 統計
ネットワークグラフと研究者情報をブラウザ用JSONに変換
"""

import json
import os
import unicodedata
from collections import defaultdict
from datetime import datetime

import networkx as nx

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "docs")  # GitHub Pages用


# トピック別カラーパレット（最大30色）
COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#e6beff", "#1abc9c", "#e74c3c", "#3498db", "#2ecc71",
    "#9b59b6", "#f39c12", "#1abc9c", "#e67e22", "#2c3e50",
]

# 部局カラーパレット
ORG_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9A6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#e6beff", "#1abc9c", "#e74c3c", "#3498db", "#2ecc71",
    "#9b59b6", "#f39c12", "#e67e22", "#2c3e50", "#fabed4",
]


def load_network():
    gexf_path = os.path.join(DATA_DIR, "osaka_researcher_network.gexf")
    return nx.read_gexf(gexf_path)


def load_topics():
    topic_path = os.path.join(DATA_DIR, "topics.json")
    with open(topic_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_details():
    path = os.path.join(DATA_DIR, "researcher_details.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text(item, field_name):
    val = item.get(field_name)
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("ja", val.get("en", ""))
    return ""


def filter_network(G, min_degree=2, max_edges=12000):
    """可視化用にネットワークをフィルタリング"""
    G_filtered = G.copy()

    nodes_to_remove = [n for n, d in G_filtered.degree() if d < min_degree]
    G_filtered.remove_nodes_from(nodes_to_remove)

    if G_filtered.number_of_edges() > max_edges:
        edges_with_weight = []
        for u, v, data in G_filtered.edges(data=True):
            score = data.get("weight", 0) * 5 + data.get("topic_sim", 0)
            edges_with_weight.append((u, v, score))
        edges_with_weight.sort(key=lambda x: x[2], reverse=True)
        keep_edges = set((u, v) for u, v, _ in edges_with_weight[:max_edges])
        remove_edges = [(u, v) for u, v in G_filtered.edges()
                        if (u, v) not in keep_edges and (v, u) not in keep_edges]
        G_filtered.remove_edges_from(remove_edges)
        nodes_to_remove = [n for n, d in G_filtered.degree() if d < 1]
        G_filtered.remove_nodes_from(nodes_to_remove)

    return G_filtered


def _get_author_role(paper, researcher_name, researcher_name_en):
    """論文における著者の役割を判定"""
    authors_data = paper.get("authors", {})
    all_authors = []
    for lang in ["ja", "en"]:
        for a in authors_data.get(lang, []):
            if isinstance(a, dict):
                all_authors.append(a.get("name", ""))

    if not all_authors:
        return ""

    def normalize(s):
        return unicodedata.normalize("NFKC", s.replace("　", " ").replace(",", " ").strip().lower())

    name_variants = set()
    for name in [researcher_name, researcher_name_en]:
        if not name:
            continue
        n = normalize(name)
        name_variants.add(n)
        name_variants.add(n.replace(" ", ""))
        parts = n.split()
        if len(parts) == 2:
            name_variants.add(f"{parts[1]} {parts[0]}")
            name_variants.add(f"{parts[1]}{parts[0]}")

    pos = -1
    for i, a in enumerate(all_authors):
        norm_a = normalize(a)
        for v in name_variants:
            if v and (v in norm_a or norm_a in v):
                pos = i
                break
        if pos >= 0:
            break

    if pos < 0:
        return ""

    roles = []
    if pos == 0:
        roles.append("第一著者")
    if pos == len(all_authors) - 1 and len(all_authors) > 1:
        roles.append("ラストオーサー")
    ca = paper.get("corresponding_author")
    if ca:
        norm_ca = normalize(str(ca))
        for v in name_variants:
            if v and (v in norm_ca or norm_ca in v):
                roles.append("責任著者")
                break

    return "・".join(roles)


def generate_network_json(G_vis, topic_data):
    """network.json を生成"""
    topic_labels = topic_data.get("topic_labels", {})

    # 部局カラー割り当て
    org_set = sorted(set(G_vis.nodes[n].get("org_lv1", "") for n in G_vis.nodes()) - {""})
    org_colors = {org: ORG_COLORS[i % len(ORG_COLORS)] for i, org in enumerate(org_set)}

    # Pre-compute layout positions (ForceAtlas2 if available, else spring_layout)
    print("レイアウト計算中...")
    try:
        from fa2 import ForceAtlas2
        fa2 = ForceAtlas2(
            outboundAttractionDistribution=True,
            edgeWeightInfluence=1.0,
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,
            verbose=False,
        )
        pos = fa2.forceatlas2_networkx_layout(G_vis, pos=None, iterations=200)
        print("  ForceAtlas2 レイアウト使用")
    except ImportError:
        pos = nx.spring_layout(G_vis, k=0.3, iterations=80, seed=42, weight="weight")
        print("  spring_layout フォールバック (pip install fa2 で改善可能)")
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    scale = 4000

    nodes = []
    for node_id in G_vis.nodes():
        attrs = G_vis.nodes[node_id]
        topic = attrs.get("topic", -1)
        try:
            topic_int = int(topic)
        except (ValueError, TypeError):
            topic_int = -1

        color = COLORS[topic_int % len(COLORS)] if topic_int >= 0 else "#888888"
        degree = G_vis.degree(node_id)
        size = max(8, min(40, 5 + degree * 2))
        label = attrs.get("name", node_id) if degree >= 3 else ""

        node_entry = {
            "id": node_id,
            "label": label,
            "name": attrs.get("name", ""),
            "color": color,
            "size": size,
            "group": topic_int if topic_int >= 0 else -1,
            "org_lv1": attrs.get("org_lv1", ""),
        }
        if node_id in pos:
            x, y = pos[node_id]
            node_entry["x"] = round((x - x_min) / (x_max - x_min) * scale - scale / 2, 1)
            node_entry["y"] = round((y - y_min) / (y_max - y_min) * scale - scale / 2, 1)
        nodes.append(node_entry)

    edges = []
    for u, v, data in G_vis.edges(data=True):
        weight = data.get("weight", 0)
        topic_sim = data.get("topic_sim", 0)
        edge_width = max(0.5, min(5, weight * 0.5 + topic_sim * 2))
        edges.append({
            "from": u, "to": v,
            "width": round(edge_width, 2),
            "origWeight": weight,
            "origTopicSim": round(topic_sim, 3),
        })

    # Data freshness: use modification time of researcher_details.json
    details_path = os.path.join(DATA_DIR, "researcher_details.json")
    last_updated = datetime.fromtimestamp(os.path.getmtime(details_path)).strftime("%Y-%m-%d")

    output = {
        "nodes": nodes,
        "edges": edges,
        "org_colors": org_colors,
        "topic_labels": {str(k): v for k, v in topic_labels.items()},
        "last_updated": last_updated,
    }

    path = os.path.join(OUTPUT_DIR, "network.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"保存: {path} ({len(nodes)}ノード, {len(edges)}エッジ)")


def generate_researchers_json(G_vis, details, topic_data):
    """researchers.json を生成"""
    topic_labels = topic_data.get("topic_labels", {})
    researcher_topics = topic_data.get("researcher_topics", {})

    vis_nodes = set(G_vis.nodes())
    detail_map = {}
    for d in details:
        pl = d.get("permalink", "")
        if pl and pl in vis_nodes:
            detail_map[pl] = d

    researchers = {}
    for node_id in G_vis.nodes():
        attrs = G_vis.nodes[node_id]
        d = detail_map.get(node_id, {})
        osaka = d.get("osaka_info", {})

        name = attrs.get("name", osaka.get("name_ja", node_id))
        name_en = attrs.get("name_en", osaka.get("name_en", ""))
        org = attrs.get("org", osaka.get("org_name", ""))
        org_lv1 = attrs.get("org_lv1", osaka.get("org_name_lv1", ""))
        job = attrs.get("job", osaka.get("job_name", ""))
        rf = attrs.get("research_field", osaka.get("research_field", ""))
        if isinstance(rf, list):
            rf = ", ".join(rf)

        topic_id = researcher_topics.get(node_id, -1)
        topic_label = topic_labels.get(str(topic_id), "")

        # Papers (top 8, short keys)
        papers = []
        for p in d.get("published_papers", [])[:8]:
            title = extract_text(p, "paper_title")
            if not title:
                continue
            journal = extract_text(p, "journal")
            date = ""
            pub_date = p.get("publication_date")
            if pub_date:
                date = pub_date if isinstance(pub_date, str) else ""
            role = _get_author_role(p, name, name_en)
            entry = {"t": title}
            if journal:
                entry["j"] = journal
            if date:
                entry["d"] = date
            if role:
                entry["r"] = role
            papers.append(entry)

        # Grants (top 8, short keys)
        grants = []
        for g in d.get("research_projects", [])[:8]:
            title = extract_text(g, "research_project_title")
            if not title:
                continue
            category = extract_text(g, "category")
            role = ""
            investigators = g.get("investigators", {})
            for lang in ["ja", "en"]:
                for inv in investigators.get(lang, []):
                    if isinstance(inv, dict):
                        inv_role = inv.get("role", "")
                        if inv_role and (name in inv.get("name", "") or name_en in inv.get("name", "")):
                            role = inv_role
                            break
                if role:
                    break
            from_year = str(g.get("from_date", ""))[:7]
            to_year = str(g.get("to_date", ""))[:7]
            entry = {"t": title}
            if category:
                entry["c"] = category
            if role:
                entry["r"] = role
            if from_year:
                entry["f"] = from_year
            if to_year:
                entry["to"] = to_year
            grants.append(entry)

        researchers[node_id] = {
            "n": name, "ne": name_en,
            "o": org, "o1": org_lv1,
            "j": job, "f": rf,
            "t": topic_label,
            "np": len(d.get("published_papers", [])),
            "ng": len(d.get("research_projects", [])),
            "pp": papers, "gg": grants,
        }

    path = os.path.join(OUTPUT_DIR, "researchers.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(researchers, f, ensure_ascii=False)
    print(f"保存: {path} ({len(researchers)}名)")


def generate_stats_json(G, topic_data):
    """stats.json を生成"""
    topic_labels = topic_data.get("topic_labels", {})
    researcher_topics = topic_data.get("researcher_topics", {})

    topic_counts = defaultdict(list)
    for rid, tid in researcher_topics.items():
        if rid in G:
            name = G.nodes[rid].get("name", rid)
            topic_counts[tid].append(name)

    stats = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "topics": [],
    }
    for tid, members in sorted(topic_counts.items(), key=lambda x: -len(x[1])):
        label = topic_labels.get(str(tid), f"Topic_{tid}")
        stats["topics"].append({
            "id": tid, "label": label,
            "count": len(members), "members": members[:20],
        })

    path = os.path.join(OUTPUT_DIR, "stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"保存: {path}")


def generate_keywords_json(researchers_path, network_json_path):
    """キーワード（概念）ネットワークを生成"""
    import re
    from collections import Counter
    from sklearn.feature_extraction.text import TfidfVectorizer

    with open(researchers_path, "r", encoding="utf-8") as f:
        researchers = json.load(f)

    with open(network_json_path, "r", encoding="utf-8") as f:
        net = json.load(f)
    node_topics = {n["id"]: n.get("group", -1) for n in net["nodes"]}

    # MeCab tokenization (optional)
    try:
        import MeCab
        mecab_tagger = MeCab.Tagger()
        has_mecab = True
    except ImportError:
        has_mecab = False
        print("  MeCab不可。英語キーワードのみ抽出。")

    STOPS_JA = {'日本','研究','大阪','解析','開発','評価','構造','制御','応用','利用',
                '影響','検討','機構','活性','変化','関係','表現','特性','条件','対象',
                '結果','理論','問題','方法','手法','分析','観察','比較','実験','測定',
                '効果','形成','機能','反応','設計','現象','システム','モデル','処理',
                '情報','社会','教育','技術','調査','提案','生成','因子','過程','推定','運動'}

    STOPS_EN = {'the','and','for','with','from','that','this','are','was','were','been','have',
                'has','had','not','but','its','our','their','can','may','will','all','also',
                'into','between','using','based','after','under','over','about','than',
                'both','each','during','through','study','analysis','effect','effects',
                'method','approach','results','research','role','new','use','case',
                'two','one','first','however','among','associated','related','high',
                'low','level','levels','different','type','types','showed','found',
                'report','novel','group','used','model','system','data','time',
                'total','major','general','large','small','human','patients','cells',
                'cell','protein','gene','genes','expression','molecular',
                'japan','japanese','sub','sup','non','pre','via','induced','development',
                'single','evaluation','control','activity','response','potential',
                'structure','surface','specific','clinical','treatment','patient',
                'disease','function','process','increase','change','detection',
                'enhanced','improved','regulation','receptor','factor','signaling',
                'mechanism','involved','experimental','application','properties',
                'formation','rate','age','various','international','national',
                'multi','early','phase','long','term','cross','common','review',
                'condition','positive','negative','studies','design','test',
                'performance','comparison','target','impact','outcome','outcomes',
                'risk','therapy','diagnosis','identification','population',
                'area','region','regions','state','states','recent','evidence',
                'including','cancer','mice','rat','rats','mouse','vivo','vitro'}

    def tokenize(text):
        tokens = []
        if has_mecab:
            node = mecab_tagger.parseToNode(text)
            while node:
                f = node.feature.split(",")
                if f[0] == "名詞" and f[1] in ("一般", "サ変接続", "固有名詞"):
                    w = node.surface
                    if len(w) >= 2 and w not in STOPS_JA:
                        tokens.append(w)
                node = node.next
        en_words = re.findall(r'[a-zA-Z]{4,}', text.lower())
        tokens.extend(w for w in en_words if w not in STOPS_EN)
        return tokens

    # Build per-researcher token lists
    researcher_keywords = {}
    keyword_researchers = defaultdict(set)
    all_docs, all_rids = [], []

    for rid, info in researchers.items():
        texts = [info.get("f", "")]
        for p in info.get("pp", []):
            if p.get("t"): texts.append(p["t"])
        for g in info.get("gg", []):
            if g.get("t"): texts.append(g["t"])
        combined = " ".join(texts)
        if not combined.strip():
            continue
        tokens = tokenize(combined)
        if tokens:
            researcher_keywords[rid] = tokens
            all_docs.append(" ".join(tokens))
            all_rids.append(rid)

    vectorizer = TfidfVectorizer(max_features=2000, max_df=0.2, min_df=3, token_pattern=r'[^\s]+')
    vectorizer.fit_transform(all_docs)
    top_kw_set = set(vectorizer.get_feature_names_out())

    for rid, tokens in researcher_keywords.items():
        for t in set(tokens):
            if t in top_kw_set:
                keyword_researchers[t].add(rid)
    keyword_researchers = {k: v for k, v in keyword_researchers.items() if len(v) >= 3}
    active_keywords = set(keyword_researchers.keys())

    cooccurrence = Counter()
    for rid, tokens in researcher_keywords.items():
        kws = sorted(set(t for t in tokens if t in active_keywords))
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                cooccurrence[(kws[i], kws[j])] += 1

    edges_raw = [(k1, k2, c) for (k1, k2), c in cooccurrence.items() if c >= 3]
    edges_raw.sort(key=lambda x: -x[2])
    edges_raw = edges_raw[:4000]

    edge_keywords = set()
    for k1, k2, _ in edges_raw:
        edge_keywords.add(k1)
        edge_keywords.add(k2)

    keyword_topic = {}
    for kw in edge_keywords:
        rids = keyword_researchers.get(kw, set())
        tc = Counter(node_topics.get(r, -1) for r in rids if node_topics.get(r, -1) >= 0)
        keyword_topic[kw] = tc.most_common(1)[0][0] if tc else -1

    G = nx.Graph()
    for kw in edge_keywords:
        G.add_node(kw)
    for k1, k2, w in edges_raw:
        G.add_edge(k1, k2, weight=w)

    pos = nx.spring_layout(G, k=0.5, iterations=60, seed=42, weight="weight")
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    scale = 4000

    nodes = []
    for kw in edge_keywords:
        n_res = len(keyword_researchers.get(kw, []))
        size = max(6, min(35, 4 + n_res * 0.8))
        topic = keyword_topic.get(kw, -1)
        color = COLORS[topic % len(COLORS)] if topic >= 0 else "#888"
        x, y = pos[kw]
        nodes.append({
            "id": kw, "label": kw, "size": round(size, 1), "color": color,
            "group": topic, "count": n_res,
            "researchers": sorted(keyword_researchers.get(kw, []))[:30],
            "x": round((x - x_min) / (x_max - x_min) * scale - scale / 2, 1),
            "y": round((y - y_min) / (y_max - y_min) * scale - scale / 2, 1),
        })

    edges = [{"from": k1, "to": k2, "width": round(max(0.5, min(4, w * 0.3)), 1), "weight": w}
             for k1, k2, w in edges_raw]

    path = os.path.join(OUTPUT_DIR, "keywords.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False)
    print(f"保存: {path} ({len(nodes)}キーワード, {len(edges)}エッジ)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== データ生成 ===")
    G = load_network()
    topic_data = load_topics()
    details = load_details()
    print(f"ネットワーク: {G.number_of_nodes()}ノード, {G.number_of_edges()}エッジ")

    G_vis = filter_network(G, min_degree=2, max_edges=12000)
    print(f"フィルタ後: {G_vis.number_of_nodes()}ノード, {G_vis.number_of_edges()}エッジ")

    generate_network_json(G_vis, topic_data)
    generate_researchers_json(G_vis, details, topic_data)
    generate_stats_json(G, topic_data)

    # キーワードネットワーク
    print("キーワードネットワーク生成中...")
    researchers_path = os.path.join(OUTPUT_DIR, "researchers.json")
    network_path = os.path.join(OUTPUT_DIR, "network.json")
    generate_keywords_json(researchers_path, network_path)

    print("\n=== 完了 ===")
    print("docs/ 内に network.json, researchers.json, stats.json, keywords.json を生成しました。")
    print("docs/index.html をブラウザで開いてください。")


if __name__ == "__main__":
    main()
