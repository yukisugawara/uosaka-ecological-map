"""
大阪大学 Ecological Map — Step 3: インタラクティブ可視化
Pyvisでブラウザ閲覧可能なネットワークマップを生成
"""

import json
import os
from collections import defaultdict

import networkx as nx
from pyvis.network import Network

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


def load_network():
    gexf_path = os.path.join(DATA_DIR, "osaka_researcher_network.gexf")
    return nx.read_gexf(gexf_path)


def load_topics():
    topic_path = os.path.join(DATA_DIR, "topics.json")
    with open(topic_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_network(G, min_degree=1, max_edges=15000):
    """可視化用にネットワークをフィルタリング"""
    G_filtered = G.copy()

    # 孤立ノード除去
    nodes_to_remove = [n for n, d in G_filtered.degree() if d < min_degree]
    G_filtered.remove_nodes_from(nodes_to_remove)

    # エッジが多すぎる場合、重みが低いエッジを間引く
    if G_filtered.number_of_edges() > max_edges:
        edges_with_weight = []
        for u, v, data in G_filtered.edges(data=True):
            # 共著の重み + トピック類似度を合算スコアに
            score = data.get("weight", 0) * 5 + data.get("topic_sim", 0)
            edges_with_weight.append((u, v, score))
        edges_with_weight.sort(key=lambda x: x[2], reverse=True)
        # 上位エッジのみ残す
        keep_edges = set((u, v) for u, v, _ in edges_with_weight[:max_edges])
        remove_edges = [(u, v) for u, v in G_filtered.edges() if (u, v) not in keep_edges and (v, u) not in keep_edges]
        G_filtered.remove_edges_from(remove_edges)
        # エッジ除去後の孤立ノードも除去
        nodes_to_remove = [n for n, d in G_filtered.degree() if d < 1]
        G_filtered.remove_nodes_from(nodes_to_remove)

    return G_filtered


def create_visualization(G, topic_data, output_name="index.html"):
    """Pyvisでインタラクティブなネットワーク可視化を生成"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ネットワークをフィルタリング
    G_vis = filter_network(G, min_degree=2, max_edges=12000)
    print(f"可視化対象: {G_vis.number_of_nodes()}ノード, {G_vis.number_of_edges()}エッジ")

    # Pyvisネットワーク作成
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#0a0a0a",
        font_color="white",
        directed=False,
        select_menu=True,
        filter_menu=True,
    )

    # 物理シミュレーション設定
    net.set_options(json.dumps({
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.02,
                "damping": 0.4,
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": True,
                "iterations": 200,
                "updateInterval": 25,
            },
        },
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": {"size": 12, "face": "arial"},
        },
        "edges": {
            "color": {"color": "#444444", "highlight": "#ffffff"},
            "smooth": {"type": "continuous"},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "multiselect": True,
            "navigationButtons": True,
        },
    }))

    topic_labels = topic_data.get("topic_labels", {})

    # ノード追加
    for node_id in G_vis.nodes():
        attrs = G_vis.nodes[node_id]
        name = attrs.get("name", node_id)
        name_en = attrs.get("name_en", "")
        org = attrs.get("org", "")
        org_lv1 = attrs.get("org_lv1", "")
        job = attrs.get("job", "")
        topic = attrs.get("topic", -1)
        topic_label = attrs.get("topic_label", "")

        # トピック別色分け
        try:
            topic_int = int(topic)
        except (ValueError, TypeError):
            topic_int = -1
        color = COLORS[topic_int % len(COLORS)] if topic_int >= 0 else "#888888"

        # ノードサイズ（次数比例）
        degree = G_vis.degree(node_id)
        size = max(8, min(40, 5 + degree * 2))

        # ツールチップ
        rf = attrs.get("research_field", "")
        if isinstance(rf, list):
            rf = ", ".join(rf)
        tooltip = (
            f"<b>{name}</b> ({name_en})<br>"
            f"所属: {org_lv1} {org}<br>"
            f"職位: {job}<br>"
            f"研究分野: {rf}<br>"
            f"トピック: {topic_label}<br>"
            f"共著つながり: {degree}"
        )

        # ラベルは次数が高いノードのみ表示
        label = name if degree >= 3 else ""

        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color=color,
            size=size,
            group=str(topic_int) if topic_int >= 0 else "other",
        )

    # エッジ追加
    for u, v, data in G_vis.edges(data=True):
        weight = data.get("weight", 0)
        topic_sim = data.get("topic_sim", 0)

        # エッジの太さと透明度
        edge_width = max(0.5, min(5, weight * 0.5))
        edge_color = "#666666" if weight > 0 else "#333333"

        net.add_edge(u, v, width=edge_width, color=edge_color)

    # HTML出力
    output_path = os.path.join(OUTPUT_DIR, output_name)
    net.save_graph(output_path)

    # カスタムHTMLヘッダーを追加
    _inject_header(output_path, topic_labels)

    print(f"保存: {output_path}")
    return output_path


def _inject_header(html_path, topic_labels):
    """HTMLにタイトルとトピック凡例を注入"""
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    legend_items = ""
    for tid, label in sorted(topic_labels.items(), key=lambda x: int(x[0]) if x[0].lstrip('-').isdigit() else 999):
        try:
            tid_int = int(tid)
        except ValueError:
            continue
        if tid_int < 0:
            continue
        color = COLORS[tid_int % len(COLORS)]
        short_label = label[:50] + "..." if len(label) > 50 else label
        legend_items += (
            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<span style="width:12px;height:12px;border-radius:50%;'
            f'background:{color};display:inline-block;margin-right:6px;flex-shrink:0;"></span>'
            f'<span style="font-size:11px;">{short_label}</span></div>\n'
        )

    header_html = f"""
    <div id="map-header" style="position:fixed;top:0;left:0;right:0;z-index:1000;
        background:rgba(10,10,10,0.9);padding:10px 20px;border-bottom:1px solid #333;">
        <h1 style="color:white;margin:0;font-size:20px;font-family:sans-serif;">
            大阪大学 研究者エコロジカルマップ
            <span style="font-size:12px;color:#888;margin-left:10px;">
                University of Osaka Researcher Ecological Map
            </span>
        </h1>
        <p style="color:#aaa;margin:2px 0 0;font-size:12px;">
            ノードをクリックで詳細表示 | ホイールでズーム | ドラッグで移動
        </p>
    </div>
    <div id="topic-legend" style="position:fixed;top:70px;right:10px;z-index:1000;
        background:rgba(10,10,10,0.85);padding:10px;border-radius:8px;
        border:1px solid #333;max-height:60vh;overflow-y:auto;width:250px;">
        <div style="color:white;font-size:13px;font-weight:bold;margin-bottom:5px;">
            トピック Topics</div>
        {legend_items}
    </div>
    <style>
        body {{ margin: 0; padding: 0; }}
        #mynetwork {{ margin-top: 60px; }}
    </style>
    """

    html = html.replace("<body>", f"<body>\n{header_html}", 1)

    # metaタグ追加
    html = html.replace(
        "<head>",
        '<head>\n<meta charset="utf-8">\n'
        "<title>大阪大学 研究者エコロジカルマップ</title>\n",
        1,
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def generate_stats_page(G, topic_data):
    """統計情報ページを生成"""
    topic_labels = topic_data.get("topic_labels", {})
    researcher_topics = topic_data.get("researcher_topics", {})

    # トピック別研究者数
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
            "id": tid,
            "label": label,
            "count": len(members),
            "members": members[:20],  # 上位20名
        })

    stats_path = os.path.join(OUTPUT_DIR, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"統計保存: {stats_path}")


def main():
    print("=== 可視化生成 ===")

    G = load_network()
    topic_data = load_topics()
    print(f"ネットワーク: {G.number_of_nodes()}ノード, {G.number_of_edges()}エッジ")

    # メイン可視化
    create_visualization(G, topic_data)

    # 統計ページ
    generate_stats_page(G, topic_data)

    print("\n=== 完了 ===")
    print(f"docs/index.html をブラウザで開いてください。")
    print(f"GitHub Pagesで公開するには docs/ フォルダを設定してください。")


if __name__ == "__main__":
    main()
