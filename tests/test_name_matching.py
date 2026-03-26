"""
名前マッチング・正規化ロジックのユニットテスト
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from importlib.machinery import SourceFileLoader

# Import modules with numeric prefixes using SourceFileLoader
build_network = SourceFileLoader(
    "build_network",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "02_build_network.py"),
).load_module()

visualize = SourceFileLoader(
    "visualize",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "03_visualize.py"),
).load_module()

_normalize_name = build_network._normalize_name
_build_name_index = build_network._build_name_index
_get_author_role = visualize._get_author_role


class TestNormalizeName:
    def test_full_width_space(self):
        assert _normalize_name("田中　太郎") == "田中 太郎"

    def test_comma_separator(self):
        assert _normalize_name("Tanaka, Taro") == "Tanaka Taro"

    def test_dot_separator(self):
        assert _normalize_name("T. Tanaka") == "T Tanaka"

    def test_multiple_spaces(self):
        assert _normalize_name("田中  太郎") == "田中 太郎"

    def test_nfkc_normalization(self):
        # Full-width alphabet to half-width
        assert _normalize_name("Ｔａｎａｋａ") == "Tanaka"

    def test_empty(self):
        assert _normalize_name("") == ""

    def test_whitespace_only(self):
        assert _normalize_name("   ") == ""

    def test_leading_trailing_spaces(self):
        assert _normalize_name("  田中 太郎  ") == "田中 太郎"


class TestBuildNameIndex:
    def _make_detail(self, permalink, name_ja="", name_en="", family_ja="", given_ja="", family_en="", given_en=""):
        d = {
            "permalink": permalink,
            "osaka_info": {"name_ja": name_ja, "name_en": name_en},
            "profile": {
                "family_name": {"ja": family_ja, "en": family_en},
                "given_name": {"ja": given_ja, "en": given_en},
            },
        }
        return d

    def test_basic_lookup(self):
        details = [self._make_detail("tanaka01", name_ja="田中 太郎", name_en="Tanaka Taro")]
        idx = _build_name_index(details)
        assert idx.get("田中太郎") == "tanaka01"
        assert idx.get("田中 太郎") == "tanaka01"

    def test_reversed_name(self):
        details = [self._make_detail("tanaka01", name_ja="田中 太郎")]
        idx = _build_name_index(details)
        assert idx.get("太郎田中") == "tanaka01"
        assert idx.get("太郎 田中") == "tanaka01"

    def test_english_name(self):
        details = [self._make_detail("tanaka01", name_en="Taro Tanaka")]
        idx = _build_name_index(details)
        assert idx.get("TaroTanaka") == "tanaka01"
        assert idx.get("TanakaTaro") == "tanaka01"

    def test_profile_names(self):
        details = [self._make_detail("tanaka01", family_ja="田中", given_ja="太郎")]
        idx = _build_name_index(details)
        assert idx.get("田中太郎") == "tanaka01"

    def test_no_permalink_skipped(self):
        details = [{"permalink": "", "osaka_info": {"name_ja": "田中 太郎"}}]
        idx = _build_name_index(details)
        assert "田中太郎" not in idx

    def test_collision_not_overwritten(self):
        """同姓同名の場合、最初に登録された方が残る"""
        details = [
            self._make_detail("person_a", name_ja="田中 太郎"),
            self._make_detail("person_b", name_ja="田中 太郎"),
        ]
        idx = _build_name_index(details)
        # First one wins
        assert idx.get("田中太郎") == "person_a"


class TestGetAuthorRole:
    def _make_paper(self, authors_ja=None, corresponding=None):
        paper = {"authors": {"ja": [], "en": []}}
        if authors_ja:
            paper["authors"]["ja"] = [{"name": n} for n in authors_ja]
        if corresponding:
            paper["corresponding_author"] = corresponding
        return paper

    def test_first_author(self):
        paper = self._make_paper(["田中 太郎", "山田 花子", "佐藤 一郎"])
        role = _get_author_role(paper, "田中 太郎", "Taro Tanaka")
        assert "第一著者" in role

    def test_last_author(self):
        paper = self._make_paper(["田中 太郎", "山田 花子", "佐藤 一郎"])
        role = _get_author_role(paper, "佐藤 一郎", "Ichiro Sato")
        assert "ラストオーサー" in role

    def test_corresponding_author(self):
        paper = self._make_paper(["田中 太郎", "佐藤 一郎"], corresponding="田中 太郎")
        role = _get_author_role(paper, "田中 太郎", "Taro Tanaka")
        assert "責任著者" in role

    def test_first_and_last_single_not_last(self):
        """1人の場合はラストオーサーにはならない"""
        paper = self._make_paper(["田中 太郎"])
        role = _get_author_role(paper, "田中 太郎", "Taro Tanaka")
        assert "第一著者" in role
        assert "ラスト" not in role

    def test_middle_author_no_role(self):
        paper = self._make_paper(["田中 太郎", "山田 花子", "佐藤 一郎"])
        role = _get_author_role(paper, "山田 花子", "Hanako Yamada")
        assert role == ""

    def test_not_found(self):
        paper = self._make_paper(["田中 太郎", "山田 花子"])
        role = _get_author_role(paper, "鈴木 次郎", "Jiro Suzuki")
        assert role == ""

    def test_empty_authors(self):
        paper = {"authors": {}}
        role = _get_author_role(paper, "田中 太郎", "Taro Tanaka")
        assert role == ""

    def test_reversed_name_match(self):
        """姓名逆転でもマッチする"""
        paper = self._make_paper(["太郎 田中", "花子 山田"])
        role = _get_author_role(paper, "田中 太郎", "")
        assert "第一著者" in role

    def test_combined_roles(self):
        """第一著者かつ責任著者"""
        paper = self._make_paper(["田中 太郎", "山田 花子"], corresponding="田中太郎")
        role = _get_author_role(paper, "田中 太郎", "Taro Tanaka")
        assert "第一著者" in role
        assert "責任著者" in role
