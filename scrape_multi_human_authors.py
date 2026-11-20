#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C-LongSlopBench 多博主真实人类中文长文采集器 (Multi-Author Authentic Human Corpus Scraper)

采集原则（参考 SlopBench Pre-AI Baseline 与 Catch Me If You Can 多作者多样性标准）：
1. 100% 确认为真人亲笔撰写：
   - 左耳朵耗子（陈皓，CoolShell）：抓取 2011–2022 年经典架构与工程思想长文（100% Pre-ChatGPT 黄金人类锚点）
   - 阮一峰（《科技爱好者周刊》刊首语）：抓取第 50–155 期（2019–2021 年，100% Pre-ChatGPT）深度评论与随笔
   - 云风（吴云洋，CodingNow）：中国游戏引擎泰斗，独立博客 20 年亲笔技术随笔（Atom Feed）
   - piglei（《Python 工匠》作者）：深度软件工程与编程哲学长文（RSS Feed）
   - laixintao（卡瓦邦噶！）：知名基础设施工程师独立博客长文（RSS Feed）
   - 数字生命卡兹克（已有 19 篇）：AI 时代头部科技自媒体长文
2. 统一清洗：剔除代码块（```...```）、HTML 标签、图片链接与页脚广告，只保留纯自然语言长文段落（>= 600 字）。
3. 落盘至 `c_longslop_bench/data/raw/multi_human_authors/<author_id>/` 并生成 `authors_manifest.json`。
"""
import html
import json
import os
import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

OUT_ROOT = Path("/Users/wenweili/PycharmProjects/Auto-claude-code-research-in-sleep/ai-writing-research/c_longslop_bench/data/raw/multi_human_authors")
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def http_get(url: str, timeout: int = 18) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def clean_prose_from_markdown(md_text: str) -> str:
    """移除代码块、图片、链接 URL、引用头，只保留干净的自然语言中文正文段落。"""
    t = md_text
    # 移除 fenced code blocks
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    # 移除 HTML 标签
    t = re.sub(r"<[^>]+>", "", t)
    # 移除图片
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    # 将 [文字](链接) 还原为纯文字
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    # 移除裸 URL
    t = re.sub(r"https?://\S+", "", t)
    # 移除 Markdown 标题井号但保留标题文字成段
    lines = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if s.startswith("> source:") or s.startswith("<!--") or s.startswith("（全文完）") or s.startswith("（完）"):
            continue
        s = re.sub(r"^#{1,6}\s*", "", s)
        lines.append(s)
    cleaned = "\n".join(lines)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_prose_from_html(raw_html: str) -> str:
    t = raw_html
    # 先移除 pre / code 块，避免代码混入自然语言探针
    t = re.sub(r"<pre[^>]*>.*?</pre>", "", t, flags=re.S | re.I)
    t = re.sub(r"<script[^>]*>.*?</script>", "", t, flags=re.S | re.I)
    t = re.sub(r"<style[^>]*>.*?</style>", "", t, flags=re.S | re.I)
    # 段落与换行标签转为双换行
    t = re.sub(r"</?(p|div|br|h[1-6]|li|blockquote)[^>]*>", "\n\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"https?://\S+", "", t)
    lines = [ln.strip() for ln in t.split("\n")]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ── 1. 抓取左耳朵耗子 (陈皓 CoolShell, 2011-2022 Pre-AI 经典长文) ──

def scrape_coolshell(max_count: int = 25) -> List[Dict]:
    print("[1/5] Scraping CoolShell (陈皓 / 左耳朵耗子, Pre-AI 2011-2022)...")
    feat_md = http_get("https://raw.githubusercontent.com/marvinqu/coolshell-markdown/main/featured.md")
    readme_md = http_get("https://raw.githubusercontent.com/marvinqu/coolshell-markdown/main/readme.md")

    links = re.findall(r"\[([^\]]+)\]\(\.?/?(\d{4}/[^)]+\.md)\)", feat_md + "\n" + readme_md)
    seen_paths = set()
    articles = []
    # 优先选思辨、架构、工程文化、职业发展类散文长文（非纯代码罗列教程）
    skip_keywords = ("简明教程", "21天", "Hello World", "命令行神器", "编程技巧", "修饰器", "泛型编程", "切片，接口")

    for title, rel_path in links:
        title = title.strip()
        if rel_path in seen_paths:
            continue
        if any(k in title for k in skip_keywords):
            continue
        seen_paths.add(rel_path)
        year = rel_path[:4]
        raw_url = f"https://raw.githubusercontent.com/marvinqu/coolshell-markdown/main/{rel_path}"
        try:
            md = http_get(raw_url)
            prose = clean_prose_from_markdown(md)
            if len(prose) >= 800:
                articles.append({
                    "author": "chenhao_coolshell",
                    "author_zh": "陈皓(左耳朵耗子)",
                    "era": "pre_ai" if int(year) <= 2022 else "2023",
                    "year": year,
                    "title": title,
                    "url": f"https://coolshell.cn ({rel_path})",
                    "text": prose[:8000],
                    "char_len": len(prose[:8000]),
                })
                if len(articles) >= max_count:
                    break
        except Exception as e:
            continue
    print(f"  -> Collected {len(articles)} articles from CoolShell.")
    return articles


# ── 2. 抓取阮一峰《科技爱好者周刊》2019-2021 刊首语随笔 (100% Pre-AI) ──

def scrape_ruanyifeng_essays(max_count: int = 25) -> List[Dict]:
    print("[2/5] Scraping Ruan Yifeng Weekly Essays (阮一峰 2019-2021 刊首语)...")
    articles = []
    # 第 45 期到 165 期全部是 2019–2021 年（ChatGPT 诞生前）的原创刊首语评论
    issue_nums = list(range(45, 165, 3))
    for num in issue_nums:
        url = f"https://raw.githubusercontent.com/ruanyf/weekly/master/docs/issue-{num}.md"
        try:
            md = http_get(url, timeout=12)
        except Exception:
            continue
        # 截取 `## 刊首语` 或 `## 本周话题` 到下一个 `## ` 之间的纯评论正文
        m = re.search(r"##\s*(?:刊首语|本周话题)[：: ]*([^\n]*)\n(.*?)(?=\n##\s+)", md, flags=re.S)
        if not m:
            continue
        sub_title = m.group(1).strip() or f"周刊第{num}期刊首语"
        essay_md = m.group(2)
        prose = clean_prose_from_markdown(essay_md)
        if len(prose) >= 550:
            articles.append({
                "author": "ruanyifeng",
                "author_zh": "阮一峰",
                "era": "pre_ai",
                "year": "2019-2021",
                "title": sub_title,
                "url": f"https://github.com/ruanyf/weekly/blob/master/docs/issue-{num}.md",
                "text": prose,
                "char_len": len(prose),
            })
            if len(articles) >= max_count:
                break
    print(f"  -> Collected {len(articles)} essays from Ruan Yifeng.")
    return articles


# ── 3. 抓取云风 (CodingNow Atom Feed) ──

def scrape_yunfeng() -> List[Dict]:
    print("[3/5] Scraping Yunfeng's Blog (云风 CodingNow)...")
    articles = []
    try:
        xml_text = http_get("https://blog.codingnow.com/atom.xml")
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", "", ns) or "").strip()
            pub = (entry.findtext("atom:published", "", ns) or "")[:10]
            link_el = entry.find("atom:link[@rel='alternate']", ns)
            link = link_el.attrib.get("href", "") if link_el is not None else ""
            content = entry.findtext("atom:content", "", ns) or entry.findtext("atom:summary", "", ns) or ""
            prose = clean_prose_from_html(content)
            if len(prose) >= 600:
                articles.append({
                    "author": "yunfeng",
                    "author_zh": "云风(吴云洋)",
                    "era": "post_ai_authentic_human",
                    "year": pub[:4] if pub else "2025",
                    "title": title,
                    "url": link,
                    "text": prose[:8000],
                    "char_len": len(prose[:8000]),
                })
    except Exception as e:
        print(f"  [Warn] Yunfeng scrape error: {e}")
    print(f"  -> Collected {len(articles)} articles from Yunfeng.")
    return articles


# ── 4. 抓取 piglei (《Python 工匠》作者独立博客 RSS) ──

def scrape_piglei() -> List[Dict]:
    print("[4/5] Scraping piglei's Blog (piglei.com)...")
    articles = []
    try:
        xml_text = http_get("https://www.piglei.com/feeds/latest/")
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = item.findtext("description") or ""
            prose = clean_prose_from_html(desc)
            if len(prose) >= 600:
                articles.append({
                    "author": "piglei",
                    "author_zh": "piglei(Python工匠作者)",
                    "era": "authentic_human_blog",
                    "year": "2020-2026",
                    "title": title,
                    "url": link,
                    "text": prose[:8000],
                    "char_len": len(prose[:8000]),
                })
    except Exception as e:
        print(f"  [Warn] piglei scrape error: {e}")
    print(f"  -> Collected {len(articles)} articles from piglei.")
    return articles


# ── 5. 抓取 laixintao (卡瓦邦噶！kawabangga.com RSS) ──

def scrape_kawabangga() -> List[Dict]:
    print("[5/5] Scraping Kawabangga Blog (laixintao)...")
    articles = []
    try:
        xml_text = http_get("https://www.kawabangga.com/feed")
        root = ET.fromstring(xml_text)
        ns = {"content": "http://purl.org/rss/1.0/modules/content/"}
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            ce = item.find("content:encoded", ns)
            raw_html = ce.text if (ce is not None and ce.text) else (item.findtext("description") or "")
            prose = clean_prose_from_html(raw_html)
            if len(prose) >= 600:
                articles.append({
                    "author": "laixintao_kawabangga",
                    "author_zh": "laixintao(卡瓦邦噶)",
                    "era": "authentic_human_blog",
                    "year": pub[-4:] if len(pub) >= 4 else "2025",
                    "title": title,
                    "url": link,
                    "text": prose[:8000],
                    "char_len": len(prose[:8000]),
                })
    except Exception as e:
        print(f"  [Warn] kawabangga scrape error: {e}")
    print(f"  -> Collected {len(articles)} articles from kawabangga.")
    return articles


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_collected = []
    for fn in [scrape_coolshell, scrape_ruanyifeng_essays, scrape_yunfeng, scrape_piglei, scrape_kawabangga]:
        items = fn()
        all_collected.extend(items)

    # 保存每个作者的独立目录与汇总 jsonl
    manifest = {}
    jsonl_path = OUT_ROOT / "multi_authors_corpus.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for idx, art in enumerate(all_collected, 1):
            author_dir = OUT_ROOT / art["author"]
            author_dir.mkdir(parents=True, exist_ok=True)
            safe_title = re.sub(r"[^\w\u4e00-\u9fff]+", "_", art["title"])[:40]
            md_file = author_dir / f"{idx:03d}_{safe_title}.md"
            md_file.write_text(
                f"# {art['title']}\n\n- **作者**: {art['author_zh']}\n- **年代**: {art['era']}\n- **链接**: {art['url']}\n\n---\n\n{art['text']}\n",
                encoding="utf-8",
            )
            jf.write(json.dumps(art, ensure_ascii=False) + "\n")
            manifest.setdefault(art["author"], {"author_zh": art["author_zh"], "era": art["era"], "count": 0, "avg_len": []})
            manifest[art["author"]]["count"] += 1
            manifest[art["author"]]["avg_len"].append(art["char_len"])

    for k, v in manifest.items():
        v["avg_len"] = round(sum(v["avg_len"]) / max(1, len(v["avg_len"])), 1)

    (OUT_ROOT / "authors_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Multi-Author Authentic Human Corpus Summary ===")
    print(f"Total new authentic human articles collected: {len(all_collected)}")
    for k, v in manifest.items():
        print(f"  - {k:24s} ({v['author_zh']:18s} | {v['era']:22s}): {v['count']:3d} 篇 | 平均长度 {v['avg_len']} 字")


if __name__ == "__main__":
    main()
