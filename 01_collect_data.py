"""
大阪大学 Ecological Map — Step 1: データ収集
阪大研究者総覧API + researchmap API から研究者情報を収集
"""

import json
import time
import sys
import os
import requests
import urllib3

# SSL警告を抑制（阪大サーバーの証明書問題）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── 設定 ──────────────────────────────────────────────
OSAKA_API = "https://rd.iai.osaka-u.ac.jp/api/query"
RESEARCHMAP_API = "https://api.researchmap.jp"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
PAGE_SIZE = 50
RESEARCHMAP_DELAY = 0.5  # researchmap APIへのリクエスト間隔（秒）

# researchmapから取得する業績カテゴリ
ACHIEVEMENT_TYPES = [
    "published_papers",
    "books_etc",
    "presentations",
    "research_projects",
]


def fetch_osaka_researchers():
    """阪大研究者総覧APIから全研究者リストを取得"""
    researchers = []
    page = 1

    print("=== 阪大研究者総覧APIから研究者リスト取得 ===")

    while True:
        url = f"{OSAKA_API}?page={page}&size={PAGE_SIZE}"
        resp = requests.get(url, verify=False, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        total = data["total"]
        hits = data["hits"]
        researchers.extend(hits)

        print(f"  取得: {len(researchers)}/{total}")

        if len(researchers) >= total or not hits:
            break
        page += 1
        time.sleep(0.3)

    print(f"  完了: {len(researchers)}名")
    return researchers


def fetch_researchmap_profile(permalink):
    """researchmap APIから研究者プロフィールを取得"""
    url = f"{RESEARCHMAP_API}/{permalink}?format=json"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def fetch_researchmap_achievements(permalink, achievement_type, limit=100):
    """researchmap APIから業績を取得（全ページ）"""
    items = []
    start = 1

    while True:
        url = (
            f"{RESEARCHMAP_API}/{permalink}/{achievement_type}"
            f"?format=json&limit={limit}&start={start}"
        )
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return items
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("items", [])
        items.extend(batch)

        total = data.get("total_items", 0)
        if len(items) >= total or not batch:
            break
        start += limit
        time.sleep(RESEARCHMAP_DELAY)

    return items


def collect_researcher_detail(researcher):
    """1名分の研究者データをresearchmapから収集"""
    permalink = researcher.get("permalink")
    if not permalink:
        return None

    detail = {"osaka_info": researcher, "permalink": permalink}

    # プロフィール取得
    profile = fetch_researchmap_profile(permalink)
    if profile is None:
        return detail  # researchmapに登録なし
    detail["profile"] = profile

    # 各業績カテゴリを取得
    for atype in ACHIEVEMENT_TYPES:
        time.sleep(RESEARCHMAP_DELAY)
        items = fetch_researchmap_achievements(permalink, atype)
        detail[atype] = items

    return detail


def save_checkpoint(data, filename):
    """進捗を保存（アトミック書き込みで破損防止）"""
    path = os.path.join(OUTPUT_DIR, filename)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)  # アトミック操作


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: 阪大研究者リスト取得
    researchers_list_path = os.path.join(OUTPUT_DIR, "osaka_researchers_list.json")
    if os.path.exists(researchers_list_path):
        print("既存の研究者リストを読み込み中...")
        with open(researchers_list_path, "r", encoding="utf-8") as f:
            researchers = json.load(f)
        print(f"  {len(researchers)}名")
    else:
        researchers = fetch_osaka_researchers()
        save_checkpoint(researchers, "osaka_researchers_list.json")

    # Step 2: researchmapから詳細取得（チェックポイント付き）
    details_path = os.path.join(OUTPUT_DIR, "researcher_details.json")
    if os.path.exists(details_path):
        with open(details_path, "r", encoding="utf-8") as f:
            all_details = json.load(f)
        done_permalinks = {d["permalink"] for d in all_details if d}
        print(f"既存データ: {len(done_permalinks)}名分")
    else:
        all_details = []
        done_permalinks = set()

    remaining = [
        r for r in researchers
        if r.get("permalink") and r["permalink"] not in done_permalinks
    ]
    print(f"\n=== researchmap詳細取得: 残り{len(remaining)}名 ===")

    for i, researcher in enumerate(remaining):
        permalink = researcher["permalink"]
        name = researcher.get("name_ja", researcher.get("name", permalink))

        try:
            detail = collect_researcher_detail(researcher)
            if detail:
                all_details.append(detail)
                n_papers = len(detail.get("published_papers", []))
                n_grants = len(detail.get("research_projects", []))
                print(f"  [{i+1}/{len(remaining)}] {name} — 論文{n_papers}, 助成金{n_grants}")
            else:
                print(f"  [{i+1}/{len(remaining)}] {name} — スキップ（permalink無し）")
        except Exception as e:
            print(f"  [{i+1}/{len(remaining)}] {name} — エラー: {e}")
            all_details.append({
                "osaka_info": researcher,
                "permalink": permalink,
                "error": str(e),
            })

        # 50名ごとにチェックポイント保存
        if (i + 1) % 50 == 0:
            save_checkpoint(all_details, "researcher_details.json")
            print(f"  >>> チェックポイント保存: {len(all_details)}名")

    # 最終保存
    save_checkpoint(all_details, "researcher_details.json")
    print(f"\n=== 完了: {len(all_details)}名のデータを保存 ===")
    print(f"  出力: {details_path}")

    # 統計
    n_with_papers = sum(1 for d in all_details if len(d.get("published_papers", [])) > 0)
    n_with_grants = sum(1 for d in all_details if len(d.get("research_projects", [])) > 0)
    total_papers = sum(len(d.get("published_papers", [])) for d in all_details)
    total_grants = sum(len(d.get("research_projects", [])) for d in all_details)
    print(f"\n  論文あり: {n_with_papers}名 (計{total_papers}件)")
    print(f"  助成金あり: {n_with_grants}名 (計{total_grants}件)")


if __name__ == "__main__":
    main()
