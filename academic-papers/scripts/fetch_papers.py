import os
import json
import time
import re
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PAPERS_JSON = REPO_ROOT / "papers.json"
INDEX_HTML = REPO_ROOT / "index.html"
SENTINEL = REPO_ROOT / ".no_new_papers"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENAI_KEY = os.environ["OPENAI_API_KEY"]  # fails fast if secret is missing

KEYWORDS = [
    "social work disability",
    "cognitive disability intervention",
    "AAC augmentative communication",
    "intellectual disability social support",
]

DAYS_BACK = 7
SLEEP = 0.5  # seconds between API calls (NCBI rate-limit courtesy)
NCBI_EMAIL = "academic-papers-bot@example.com"
NCBI_TOOL = "academic-papers-bot"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SS_BASE = "https://api.semanticscholar.org/graph/v1"

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
client = OpenAI(api_key=OPENAI_KEY)

SYSTEM_PROMPT = (
    "אתה עוזר מחקרי המתמחה ברווחה סוציאלית, לקויות קוגניטיביות ו-AAC. "
    "קבל מאמר מדעי ותחזיר JSON עם שני שדות בלבד:\n"
    '{"title_he": "תרגום כותרת קצר ומדויק לעברית", '
    '"summary_he": "סיכום של 10-15 משפטים בעברית המיועד לאנשי מקצוע בתחום העבודה הסוציאלית"}'
)

SYSTEM_PROMPT_NO_ABSTRACT = (
    "אתה עוזר מחקרי המתמחה ברווחה סוציאלית, לקויות קוגניטיביות ו-AAC. "
    "קיבלת כותרת של מאמר מדעי. תרגם את הכותרת לעברית וכתוב סיכום של 10-15 משפטים בעברית "
    "על תוכן המאמר — מטרתו, שיטות המחקר, ממצאים עיקריים והשלכות לתחום העבודה הסוציאלית. "
    "החזר JSON עם שני שדות בלבד:\n"
    '{"title_he": "תרגום כותרת קצר ומדויק לעברית", '
    '"summary_he": "סיכום של 10-15 משפטים בעברית"}'
)

# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _normalize(title: str) -> str:
    t = unicodedata.normalize("NFKD", title).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def build_dedup_sets(papers: list) -> tuple:
    dois = {p["id"] for p in papers}
    titles = {_normalize(p["title_en"]) for p in papers}
    return dois, titles


def is_duplicate(paper: dict, existing_ids: set, existing_titles: set) -> bool:
    return paper["id"] in existing_ids or _normalize(paper["title_en"]) in existing_titles

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_papers() -> list:
    if PAPERS_JSON.exists():
        return json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
    return []


def save_papers(papers: list) -> None:
    PAPERS_JSON.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ---------------------------------------------------------------------------
# PubMed API
# ---------------------------------------------------------------------------

def _ncbi_params(**extra) -> dict:
    return {"tool": NCBI_TOOL, "email": NCBI_EMAIL, "retmode": "json", **extra}


def pubmed_search(keyword: str) -> list:
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).strftime("%Y/%m/%d")
    today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    params = _ncbi_params(
        db="pubmed",
        term=keyword,
        datetype="pdat",
        mindate=since,
        maxdate=today,
        retmax=50,
    )
    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=15)
    resp.raise_for_status()
    time.sleep(SLEEP)
    return resp.json()["esearchresult"]["idlist"]


def _extract_year(pub_date_elem) -> int | None:
    if pub_date_elem is None:
        return None
    year_el = pub_date_elem.find("Year")
    if year_el is not None and year_el.text:
        return int(year_el.text)
    medline_el = pub_date_elem.find("MedlineDate")
    if medline_el is not None and medline_el.text:
        m = re.search(r"\d{4}", medline_el.text)
        if m:
            return int(m.group())
    return None


def _extract_abstract(article_elem) -> str:
    abstract_elem = article_elem.find(".//Abstract")
    if abstract_elem is None:
        return ""
    parts = []
    for text_el in abstract_elem.findall("AbstractText"):
        label = text_el.get("Label")
        text = (text_el.text or "").strip()
        if text:
            parts.append(f"{label}: {text}" if label else text)
    return "\n\n".join(parts)


def _extract_doi_from_summary(summary_item: dict) -> str | None:
    for id_obj in summary_item.get("articleids", []):
        if id_obj.get("idtype") == "doi":
            return id_obj["value"].strip()
    return None


def _extract_pmc_from_summary(summary_item: dict) -> str | None:
    for id_obj in summary_item.get("articleids", []):
        if id_obj.get("idtype") == "pmc":
            return id_obj["value"].strip().replace("PMC", "")
    return None


def pubmed_fetch(pmids: list, keyword: str) -> list:
    if not pmids:
        return []

    # esummary: metadata
    summary_resp = requests.get(
        f"{NCBI_BASE}/esummary.fcgi",
        params=_ncbi_params(db="pubmed", id=",".join(pmids)),
        timeout=15,
    )
    summary_resp.raise_for_status()
    summary_data = summary_resp.json().get("result", {})
    time.sleep(SLEEP)

    # efetch: abstract XML
    fetch_resp = requests.get(
        f"{NCBI_BASE}/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "tool": NCBI_TOOL,
            "email": NCBI_EMAIL,
        },
        timeout=30,
    )
    fetch_resp.raise_for_status()
    time.sleep(SLEEP)

    root = ET.fromstring(fetch_resp.text)
    papers = []
    for article_set in root.findall("PubmedArticle"):
        medline = article_set.find("MedlineCitation")
        if medline is None:
            continue
        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else None
        if not pmid:
            continue

        article = medline.find("Article")
        if article is None:
            continue

        title_el = article.find("ArticleTitle")
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            continue

        abstract = _extract_abstract(article)

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName") or ""
            fore = author.findtext("ForeName") or ""
            collective = author.findtext("CollectiveName")
            if collective:
                authors.append(collective)
            elif last:
                authors.append(f"{last} {fore}".strip())

        # Year
        pub_date = medline.find(".//PubDate")
        year = _extract_year(pub_date)

        # DOI and PMC ID from summary
        summary_item = summary_data.get(pmid, {})
        doi = _extract_doi_from_summary(summary_item)
        pmc_id = _extract_pmc_from_summary(summary_item)
        paper_id = doi if doi else f"pubmed-{pmid}"

        papers.append({
            "id": paper_id,
            "title_en": title,
            "title_he": "",
            "authors": authors,
            "year": year,
            "source": "PubMed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "keywords": [keyword],
            "summary_he": "",
            "abstract_en": abstract,
            "pmc_id": pmc_id or "",
            "date_added": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        })

    return papers

# ---------------------------------------------------------------------------
# Semantic Scholar API
# ---------------------------------------------------------------------------

def _map_ss(raw: dict, keyword: str) -> dict | None:
    title = (raw.get("title") or "").strip()
    if not title:
        return None

    doi = (raw.get("externalIds") or {}).get("DOI")
    paper_id = doi if doi else f"ss-{raw.get('paperId', '')}"

    authors = [a.get("name", "") for a in (raw.get("authors") or [])]

    year = raw.get("year")
    if not year and raw.get("publicationDate"):
        try:
            year = int(raw["publicationDate"][:4])
        except (ValueError, TypeError):
            year = None

    url = raw.get("url") or f"https://www.semanticscholar.org/paper/{raw.get('paperId','')}"

    return {
        "id": paper_id,
        "title_en": title,
        "title_he": "",
        "authors": authors,
        "year": year,
        "source": "Semantic Scholar",
        "url": url,
        "keywords": [keyword],
        "summary_he": "",
        "abstract_en": (raw.get("abstract") or "").strip(),
        "date_added": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def ss_fetch_abstract(doi: str) -> str:
    """Try to fetch an abstract from Semantic Scholar by DOI."""
    try:
        resp = requests.get(
            f"{SS_BASE}/paper/DOI:{doi}",
            params={"fields": "abstract"},
            timeout=10,
        )
        resp.raise_for_status()
        time.sleep(SLEEP)
        return (resp.json().get("abstract") or "").strip()
    except requests.RequestException:
        return ""


def pmc_fetch_fulltext(pmc_id: str) -> str:
    """Fetch full article text from PubMed Central (open access only)."""
    try:
        resp = requests.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params={
                "db": "pmc",
                "id": pmc_id,
                "rettype": "full",
                "retmode": "xml",
                "tool": NCBI_TOOL,
                "email": NCBI_EMAIL,
            },
            timeout=30,
        )
        resp.raise_for_status()
        time.sleep(SLEEP)
        root = ET.fromstring(resp.text)
        # Extract all paragraph text from body sections
        parts = []
        for elem in root.iter():
            if elem.tag in ("p", "title") and elem.text and elem.text.strip():
                parts.append(elem.text.strip())
        fulltext = " ".join(parts)
        # Limit to ~6000 chars to stay within token budget
        return fulltext[:6000]
    except Exception:
        return ""


def ss_search(keyword: str) -> list:
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    params = {
        "query": keyword,
        "fields": "title,authors,year,externalIds,abstract,publicationDate,url",
        "publicationDateOrYear": f"{since}:",
        "limit": 50,
    }
    try:
        resp = requests.get(f"{SS_BASE}/paper/search", params=params, timeout=15)
        resp.raise_for_status()
        time.sleep(SLEEP)
        raw_list = resp.json().get("data", [])
        return [p for r in raw_list if (p := _map_ss(r, keyword)) is not None]
    except requests.RequestException as exc:
        print(f"  [Semantic Scholar] Warning: {exc} — skipping")
        return []

# ---------------------------------------------------------------------------
# OpenAI summarization
# ---------------------------------------------------------------------------

def summarize(title_en: str, abstract_en: str, doi: str = "", pmc_id: str = "") -> tuple:
    content = abstract_en

    # Step 2: try Semantic Scholar if no abstract and DOI is available
    if not content and doi:
        print(f"    [Fallback] Trying Semantic Scholar for DOI: {doi}")
        content = ss_fetch_abstract(doi)

    # Step 3: try PMC full text if still no content
    if not content and pmc_id:
        print(f"    [Fallback] Fetching full text from PMC: {pmc_id}")
        content = pmc_fetch_fulltext(pmc_id)

    has_content = bool(content)
    system_prompt = SYSTEM_PROMPT if has_content else SYSTEM_PROMPT_NO_ABSTRACT
    user_content = (
        f"כותרת: {title_en}\n\nתוכן המאמר:\n{content}"
        if has_content
        else f"כותרת: {title_en}"
    )
    if not has_content:
        print(f"    [Fallback] No content found — summarizing from title only: {title_en[:70]}")

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1000,
        )
        result = json.loads(resp.choices[0].message.content)
        return result.get("title_he", ""), result.get("summary_he", "")
    except Exception as exc:
        print(f"    [OpenAI] Error for '{title_en[:60]}': {exc}")
        return "", ""

# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>מאמרים אקדמיים | עבודה סוציאלית ולקויות</title>
  <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;800&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --rose:        #E8A598;
      --rose-dark:   #C97B6E;
      --cream:       #FFF5F0;
      --gray:        #5C4A47;
      --gray-light:  #8B7570;
      --white:       #FFFFFF;
      --badge-pm-bg: #e0f2fe;
      --badge-pm-fg: #0369a1;
      --badge-ss-bg: #f0fdf4;
      --badge-ss-fg: #15803d;
    }

    body {
      font-family: 'Heebo', sans-serif;
      background: var(--cream);
      color: var(--gray);
      min-height: 100vh;
    }

    /* ── Header ─────────────────────────────────── */
    header {
      background: linear-gradient(135deg, var(--rose-dark), var(--rose));
      color: var(--white);
      text-align: center;
      padding: 3rem 1.5rem 2.5rem;
    }
    header h1 {
      font-size: clamp(1.8rem, 5vw, 2.8rem);
      font-weight: 800;
      margin-bottom: 0.5rem;
    }
    header p { font-size: 1rem; opacity: 0.9; }
    header .meta {
      margin-top: 0.8rem;
      font-size: 0.85rem;
      opacity: 0.75;
    }

    /* ── Filters ─────────────────────────────────── */
    .filters {
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
      justify-content: center;
      padding: 1.5rem 1rem 0.5rem;
      max-width: 900px;
      margin: 0 auto;
    }
    .filter-btn {
      border: 2px solid var(--rose);
      background: var(--white);
      color: var(--rose-dark);
      font-family: 'Heebo', sans-serif;
      font-size: 0.82rem;
      font-weight: 600;
      padding: 0.45rem 1.1rem;
      border-radius: 999px;
      cursor: pointer;
      transition: background 0.2s, color 0.2s;
    }
    .filter-btn:hover,
    .filter-btn.active {
      background: linear-gradient(135deg, var(--rose), var(--rose-dark));
      color: var(--white);
      border-color: transparent;
    }

    /* ── Grid ────────────────────────────────────── */
    .papers-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 1.5rem;
      max-width: 1200px;
      margin: 1.5rem auto;
      padding: 0 1.2rem 3rem;
    }

    /* ── Card ────────────────────────────────────── */
    .paper-card {
      background: var(--white);
      border-radius: 16px;
      padding: 1.6rem;
      box-shadow: 0 4px 24px rgba(92, 74, 71, 0.10);
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .paper-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 8px 32px rgba(92, 74, 71, 0.16);
    }

    .card-header {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .source-badge {
      font-size: 0.68rem;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 20px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .badge-pubmed { background: var(--badge-pm-bg); color: var(--badge-pm-fg); }
    .badge-ss     { background: var(--badge-ss-bg); color: var(--badge-ss-fg); }
    .year-badge {
      font-size: 0.75rem;
      color: var(--gray-light);
      font-weight: 500;
    }

    .title-he {
      font-size: 1.1rem;
      font-weight: 800;
      color: var(--gray);
      line-height: 1.4;
    }
    .title-en {
      font-size: 0.8rem;
      color: var(--gray-light);
      font-style: italic;
      line-height: 1.4;
    }
    .authors {
      font-size: 0.8rem;
      color: var(--gray-light);
    }

    .keywords-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
    }
    .kw-tag {
      background: #fdf2f0;
      color: var(--rose-dark);
      font-size: 0.68rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 999px;
    }

    .summary {
      font-size: 0.9rem;
      line-height: 1.85;
      color: var(--gray);
      flex: 1;
    }
    .no-summary {
      font-style: italic;
      color: var(--gray-light);
      font-size: 0.85rem;
    }

    .read-link {
      display: inline-block;
      margin-top: 0.4rem;
      color: var(--rose-dark);
      font-weight: 700;
      font-size: 0.88rem;
      text-decoration: none;
      transition: color 0.2s;
    }
    .read-link:hover { color: var(--gray); text-decoration: underline; }

    /* ── Empty state ─────────────────────────────── */
    .empty-state {
      text-align: center;
      padding: 4rem 1rem;
      color: var(--gray-light);
      grid-column: 1 / -1;
    }
    .empty-state p { font-size: 1.1rem; }

    /* ── Responsive ──────────────────────────────── */
    @media (max-width: 520px) {
      header { padding: 2rem 1rem; }
      .papers-grid { grid-template-columns: 1fr; padding: 0 0.8rem 2rem; }
    }
  </style>
</head>
<body>

<header>
  <h1>מאמרים אקדמיים</h1>
  <p>עבודה סוציאלית, לקויות קוגניטיביות, AAC ותמיכה חברתית</p>
  <p class="meta">{{TOTAL}} מאמרים | עודכן לאחרונה: {{UPDATED}}</p>
</header>

<section class="filters" aria-label="סינון לפי נושא">
  <button class="filter-btn active" data-filter="all">הכל</button>
  <button class="filter-btn" data-filter="social work disability">עבודה סוציאלית ולקות</button>
  <button class="filter-btn" data-filter="cognitive disability intervention">התערבות קוגניטיבית</button>
  <button class="filter-btn" data-filter="AAC augmentative communication">AAC</button>
  <button class="filter-btn" data-filter="intellectual disability social support">תמיכה חברתית</button>
</section>

<main class="papers-grid" id="grid">
{{CARDS}}
</main>

<script>
  const buttons = document.querySelectorAll('.filter-btn');
  const cards   = document.querySelectorAll('.paper-card');

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const filter = btn.dataset.filter;
      let visible = 0;
      cards.forEach(card => {
        const kws = card.dataset.keywords || '';
        const show = filter === 'all' || kws.includes(filter);
        card.style.display = show ? '' : 'none';
        if (show) visible++;
      });
      // Show empty state if no cards match
      let emptyEl = document.getElementById('empty-state');
      if (visible === 0) {
        if (!emptyEl) {
          emptyEl = document.createElement('div');
          emptyEl.id = 'empty-state';
          emptyEl.className = 'empty-state';
          emptyEl.innerHTML = '<p>לא נמצאו מאמרים בקטגוריה זו.</p>';
          document.getElementById('grid').appendChild(emptyEl);
        }
      } else if (emptyEl) {
        emptyEl.remove();
      }
    });
  });
</script>

</body>
</html>
"""


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def generate_html(papers: list) -> None:
    cards_html = ""
    for p in reversed(papers):  # newest first
        authors = p.get("authors", [])
        authors_str = ", ".join(authors[:4])
        if len(authors) > 4:
            authors_str += " ועוד"

        kw_tags = "".join(
            f'<span class="kw-tag">{_escape_html(kw)}</span>'
            for kw in p.get("keywords", [])
        )
        source_class = "badge-pubmed" if p["source"] == "PubMed" else "badge-ss"
        title_he = _escape_html(p["title_he"] or p["title_en"])
        title_en = _escape_html(p["title_en"])
        summary = p.get("summary_he") or ""
        summary_html = (
            f'<p class="summary">{_escape_html(summary)}</p>'
            if summary
            else '<p class="no-summary">(סיכום אינו זמין — המאמר אינו כולל תקציר)</p>'
        )

        cards_html += f"""
<article class="paper-card" data-keywords="{' | '.join(p.get('keywords', []))}">
  <div class="card-header">
    <span class="source-badge {source_class}">{_escape_html(p['source'])}</span>
    <span class="year-badge">{p.get('year') or '—'}</span>
  </div>
  <h2 class="title-he">{title_he}</h2>
  <h3 class="title-en">{title_en}</h3>
  <p class="authors">{_escape_html(authors_str)}</p>
  <div class="keywords-row">{kw_tags}</div>
  {summary_html}
  <a class="read-link" href="{_escape_html(p['url'])}" target="_blank" rel="noopener noreferrer">
    קרא את המאמר המלא ←
  </a>
</article>"""

    total = len(papers)
    updated = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    html = HTML_TEMPLATE.replace("{{CARDS}}", cards_html)
    html = html.replace("{{TOTAL}}", str(total))
    html = html.replace("{{UPDATED}}", updated)
    INDEX_HTML.write_text(html, encoding="utf-8")
    print(f"  index.html written ({total} papers).")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def backfill_missing_summaries(papers: list) -> bool:
    """Fill in missing summaries for existing papers. Returns True if any were updated."""
    updated = False
    for p in papers:
        if p.get("summary_he"):
            continue
        print(f"  [Backfill] {p['title_en'][:70]}")
        doi = p["id"] if not p["id"].startswith(("pubmed-", "ss-")) else ""
        title_he, summary_he = summarize(p["title_en"], p.get("abstract_en", ""), doi=doi, pmc_id=p.get("pmc_id", ""))
        if title_he or summary_he:
            p["title_he"] = title_he or p.get("title_he", "")
            p["summary_he"] = summary_he
            updated = True
    return updated


def main() -> None:
    print("=== Academic Papers Fetcher ===")
    print(f"Loading {PAPERS_JSON} ...")
    existing = load_papers()
    print(f"  {len(existing)} existing papers.")

    # Fill summaries for papers that were saved without one
    print("\n[Backfill] Checking for papers without summaries ...")
    if backfill_missing_summaries(existing):
        print("  Backfill complete — saving.")
        save_papers(existing)
        generate_html(existing)

    existing_ids, existing_titles = build_dedup_sets(existing)
    new_papers: list = []

    for keyword in KEYWORDS:
        print(f"\n[Keyword] {keyword}")

        # PubMed
        print("  Fetching from PubMed ...")
        try:
            pmids = pubmed_search(keyword)
            print(f"  Found {len(pmids)} PMIDs.")
            pubmed_papers = pubmed_fetch(pmids, keyword)
            print(f"  Parsed {len(pubmed_papers)} PubMed articles.")
        except Exception as exc:
            print(f"  [PubMed] Error: {exc} — skipping.")
            pubmed_papers = []

        # Semantic Scholar
        print("  Fetching from Semantic Scholar ...")
        ss_papers = ss_search(keyword)
        print(f"  Found {len(ss_papers)} Semantic Scholar papers.")

        for paper in pubmed_papers + ss_papers:
            if is_duplicate(paper, existing_ids, existing_titles):
                continue

            # Already accepted this run from another keyword — just add keyword tag
            already = next((p for p in new_papers if p["id"] == paper["id"]), None)
            if already:
                if keyword not in already["keywords"]:
                    already["keywords"].append(keyword)
                continue

            print(f"  [NEW] {paper['title_en'][:75]}")
            doi = paper["id"] if not paper["id"].startswith(("pubmed-", "ss-")) else ""
            title_he, summary_he = summarize(paper["title_en"], paper["abstract_en"], doi=doi, pmc_id=paper.get("pmc_id", ""))
            paper["title_he"] = title_he
            paper["summary_he"] = summary_he

            # Mark seen so cross-source duplicates within this run are caught
            existing_ids.add(paper["id"])
            existing_titles.add(_normalize(paper["title_en"]))

            new_papers.append(paper)

    if not new_papers:
        print("\nNo new papers found. Skipping update.")
        SENTINEL.touch()
        return

    print(f"\nAdding {len(new_papers)} new papers.")
    all_papers = existing + new_papers
    save_papers(all_papers)
    generate_html(all_papers)
    print("Done.")


if __name__ == "__main__":
    main()
