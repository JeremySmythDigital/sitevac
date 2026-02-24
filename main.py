"""
SiteVac — Flatten any docs/wiki site into one AI-ready file.
FastAPI backend with SSE progress streaming + Stripe freemium.
"""

import asyncio
import json
import os
import re
import time
import uuid
import hashlib
import io
import zipfile
from typing import Optional
from urllib.parse import urljoin, urlparse
from collections import deque
from datetime import datetime, timezone
from html import escape

import stripe
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import requests as req_lib
from bs4 import BeautifulSoup
import html2text

def estimate_gpt_tokens(text: str) -> int:
    try:
        import tiktoken
        try:
            enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))


def estimate_claude_tokens(text: str) -> int:
    # Conservative heuristic: ~4 chars/token + 10% safety margin
    return int((len(text) / 4.0) * 1.10)

def split_blocks(text: str) -> list[str]:
    # Split text into logical paragraph blocks
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return blocks


def dedupe_boilerplate(pages: dict) -> dict:
    """
    Remove repeated paragraph blocks that appear across a large
    percentage of pages (likely nav, footer, repeated UI text).

    pages format:
    {
        url: {"title": str, "text": str}
    }
    """

    url_list = list(pages.keys())
    n = len(url_list)

    # Too small to dedupe safely
    if n < 3:
        return pages

    counts: dict[str, int] = {}
    per_page_hashes: dict[str, set[str]] = {}

    for url, d in pages.items():
        hashes = set()
        for blk in split_blocks(d["text"]):
            if len(blk) < BOILERPLATE_MIN_CHARS:
                continue
            h = hashlib.sha1(blk.encode("utf-8", "ignore")).hexdigest()
            hashes.add(h)
        per_page_hashes[url] = hashes

        for h in hashes:
            counts[h] = counts.get(h, 0) + 1

    threshold = max(2, int(n * BOILERPLATE_FRAC))
    boiler_hashes = {h for h, c in counts.items() if c >= threshold}

    if not boiler_hashes:
        return pages

    cleaned = {}
    for url, d in pages.items():
        blocks = split_blocks(d["text"])
        kept = []
        for blk in blocks:
            if len(blk) < BOILERPLATE_MIN_CHARS:
                kept.append(blk)
                continue
            h = hashlib.sha1(blk.encode("utf-8", "ignore")).hexdigest()
            if h in boiler_hashes:
                continue
            kept.append(blk)
        cleaned[url] = {
            "title": d["title"],
            "text": "\n\n".join(kept).strip(),
        }

    return cleaned

# ── Config ────────────────────────────────────────────────────────────────────
PRICE_MONTHLY   = "price_MONTHLY_ID_HERE"   # replace with Stripe price IDs
PRICE_ONETIME   = "price_ONETIME_ID_HERE"
SUCCESS_URL     = os.getenv("BASE_URL", "http://localhost:8000") + "/success?session_id={CHECKOUT_SESSION_ID}"
CANCEL_URL      = os.getenv("BASE_URL", "http://localhost:8000") + "/"

# === Lifetime-first caps ===
FREE_PAGE_LIMIT = 10
PRO_PAGE_LIMIT  = 200  # lifetime cap (protect hosting)

FREE_MAX_BYTES  = 2_000_000
PRO_MAX_BYTES   = 25_000_000

# Chunk targets (safe windows)
CHUNK_TARGETS = {
    "gpt": 80_000,
    "claude": 160_000,
    "openclaw": 40_000,
}

# Boilerplate dedupe thresholds
BOILERPLATE_FRAC = 0.60
BOILERPLATE_MIN_CHARS = 80

stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")

app = FastAPI(title="SiteVac")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory job store (for Render free tier — single process)
# For multi-worker, swap with Redis
jobs: dict[str, dict] = {}

# ── Scraper logic (same as CLI script) ───────────────────────────────────────

STRIP_SELECTORS = [
    "nav","header","footer","script","style","noscript",
    "[role='navigation']","[role='banner']","[role='contentinfo']",
    ".sidebar",".toc",".nav",".navbar",".menu",".breadcrumb",
    "#sidebar","#nav","#header","#footer",
    ".edit-page",".page-nav",".prev-next",
    ".theme-doc-sidebar-container",".md-sidebar",
]
CONTENT_SELECTORS = [
    "article","main","[role='main']",".markdown-body",
    ".md-content",".theme-doc-markdown",".content","#content","body",
]

def get_content_div(soup):
    for sel in CONTENT_SELECTORS:
        el = soup.select_one(sel)
        if el:
            for noise in STRIP_SELECTORS:
                for tag in el.select(noise):
                    tag.decompose()
            return el
    return soup.body or soup

def page_to_text(html_content, base_url):
    soup = BeautifulSoup(html_content, "html.parser")
    content = get_content_div(soup)
    h = html2text.HTML2Text()
    h.baseurl = base_url
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    h.unicode_snob = True
    h.skip_internal_links = True
    text = h.handle(str(content))
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def get_page_title(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    if soup.title:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text().strip() if h1 else "Untitled"

def collect_links(html_content, base_url, start_url):
    soup = BeautifulSoup(html_content, "html.parser")
    root_parsed = urlparse(start_url)
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(("#","mailto:","javascript:")):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if (parsed.scheme in ("http","https")
                and parsed.netloc == root_parsed.netloc
                and parsed.path.startswith(root_parsed.path)):
            clean = parsed._replace(fragment="", query="").geturl()
            links.add(clean)
    return links

async def scrape_job(job_id: str, start_url: str, max_pages: int, fmt: str):
    """Run scrape in a thread, emitting SSE-style updates into jobs[job_id]."""
    job = jobs[job_id]
    job["status"] = "running"
    job["pages"] = []
    job["log"] = []

    session = req_lib.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible; SiteVac/1.0)"

    visited = {}
    queue = deque([start_url])
    seen = {start_url}

    def push_log(msg):
        job["log"].append(msg)

    loop = asyncio.get_event_loop()

    def do_scrape():
        while queue and len(visited) < max_pages:
            url = queue.popleft()
            try:
                resp = session.get(url, timeout=15)
                resp.raise_for_status()
                if "text/html" not in resp.headers.get("content-type",""):
                    continue
                title = get_page_title(resp.text)
                text  = page_to_text(resp.text, url)
                visited[url] = {"title": title, "text": text}
                job["pages"].append({"url": url, "title": title})
                push_log(f"✓ [{len(visited)}] {title[:50]}")
                new_links = collect_links(resp.text, url, start_url)
                for link in sorted(new_links):
                    if link not in seen:
                        seen.add(link)
                        queue.append(link)
                time.sleep(0.4)
            except Exception as e:
                push_log(f"✗ SKIP {url} ({e})")
        return visited

    visited = await loop.run_in_executor(None, do_scrape)

    # Build output
    if fmt == "html":
        job["result"] = build_html(visited, start_url)
        job["ext"] = "html"
        job["mime"] = "text/html"
    elif fmt == "md":
        job["result"] = build_md(visited, start_url)
        job["ext"] = "md"
        job["mime"] = "text/markdown"
    else:
        job["result"] = build_txt(visited, start_url)
        job["ext"] = "txt"
        job["mime"] = "text/plain"

    job["status"] = "done"
    job["page_count"] = len(visited)


# ── Output builders ───────────────────────────────────────────────────────────

def build_txt(visited, start_url):
    lines = [f"# SITE DUMP: {start_url}", f"# Pages: {len(visited)}", "="*80, ""]
    for i, (url, d) in enumerate(visited.items(), 1):
        lines += [f"{'='*80}", f"PAGE {i} OF {len(visited)}", f"TITLE: {d['title']}", f"URL:   {url}", f"{'='*80}", "", d["text"], ""]
    return "\n".join(lines)

def build_md(visited, start_url):
    lines = [f"# Site Dump: {start_url}\n", f"> {len(visited)} pages scraped\n", "---\n"]
    for i, (url, d) in enumerate(visited.items(), 1):
        lines += [f"\n---\n", f"## {i}. {d['title']}\n", f"**URL:** {url}\n", d["text"], ""]
    return "\n".join(lines)

def build_html(visited, start_url):
    pages_html = []
    for i, (url, d) in enumerate(visited.items(), 1):
        body = escape(d["text"]).replace("\n","<br>")
        pages_html.append(f"""
        <div class="page" id="page-{i}">
          <div class="page-header">
            <span class="page-num">Page {i} / {len(visited)}</span>
            <h2>{escape(d['title'])}</h2>
            <a class="page-url" href="{escape(url)}" target="_blank">{escape(url)}</a>
          </div>
          <div class="page-body">{body}</div>
        </div>""")
    toc = "\n".join(f'<li><a href="#page-{i+1}">{escape(d["title"])}</a></li>'
                    for i,(_, d) in enumerate(visited.items()))
    plain = escape(build_txt(visited, start_url))
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>SiteVac — {escape(start_url)}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:system-ui,sans-serif;background:#0f0f0f;color:#e0e0e0;line-height:1.6}}
#hero{{padding:2rem;background:#1a1a2e;border-bottom:1px solid #333}}
#hero h1{{font-size:1.4rem;color:#7eb8f7}}#hero p{{color:#888;font-size:.9rem;margin-top:.3rem}}
#copy-btn{{margin-top:1rem;padding:.6rem 1.4rem;background:#2563eb;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:.95rem}}
#copy-btn:hover{{background:#1d4ed8}}#copy-btn.copied{{background:#16a34a}}
#layout{{display:flex;height:calc(100vh - 110px)}}
#toc{{width:250px;overflow-y:auto;background:#161622;border-right:1px solid #2a2a3a;padding:1rem 0}}
#toc h3{{padding:0 1rem .5rem;color:#888;font-size:.75rem;text-transform:uppercase;letter-spacing:.08em}}
#toc ol{{list-style:decimal;padding-left:2rem;padding-right:.5rem}}
#toc a{{color:#9eb8d9;text-decoration:none;font-size:.82rem;display:block;padding:.2rem 0}}#toc a:hover{{color:#7eb8f7}}
#main{{flex:1;overflow-y:auto;padding:2rem}}#content{{max-width:860px}}
.page{{margin-bottom:4rem;border:1px solid #2a2a3a;border-radius:8px;overflow:hidden}}
.page-header{{background:#1a1a2e;padding:1rem 1.5rem;border-bottom:1px solid #2a2a3a}}
.page-num{{font-size:.75rem;color:#666}}.page-header h2{{color:#c0d8f0;font-size:1.1rem;margin:.2rem 0}}
.page-url{{font-size:.75rem;color:#4a7ab5;word-break:break-all}}
.page-body{{padding:1.5rem;font-size:.88rem;white-space:pre-wrap;font-family:'Courier New',monospace;color:#ccc;background:#111}}
</style></head><body>
<div id="hero">
  <h1>📄 SiteVac — {escape(start_url)}</h1>
  <p>{len(visited)} pages • Select all &amp; paste into any AI</p>
  <button id="copy-btn" onclick="navigator.clipboard.writeText(document.getElementById('plain').value).then(()=>{{let b=document.getElementById('copy-btn');b.textContent='✅ Copied!';b.classList.add('copied');setTimeout(()=>{{b.textContent='📋 Copy All';b.classList.remove('copied')}},2500)}})">📋 Copy All to Clipboard</button>
  <textarea id="plain" style="position:absolute;left:-9999px">{plain}</textarea>
</div>
<div id="layout">
  <nav id="toc"><h3>Pages ({len(visited)})</h3><ol>{toc}</ol></nav>
  <div id="main"><div id="content">{"".join(pages_html)}</div></div>
</div></body></html>"""


# ── API Routes ────────────────────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    url: str
    format: str = "txt"
    pro_token: Optional[str] = None  # Stripe customer token for pro users

@app.post("/api/scrape")
async def start_scrape(body: ScrapeRequest):
    url = body.url.strip()
    if not url.startswith(("http://","https://")):
        raise HTTPException(400, "URL must start with http:// or https://")

    # Determine page limit
    is_pro = False
    if body.pro_token:
        # Verify token against Stripe (simplified — in prod use session/JWT)
        try:
            customer = stripe.Customer.list(email=body.pro_token, limit=1)
            if customer.data:
                subs = stripe.Subscription.list(customer=customer.data[0].id, status="active")
                is_pro = bool(subs.data)
        except Exception:
            pass

    max_pages = PRO_PAGE_LIMIT if is_pro else FREE_PAGE_LIMIT

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "pages": [], "log": [], "is_pro": is_pro, "max_pages": max_pages}

    asyncio.create_task(scrape_job(job_id, url, max_pages, body.format))

    return {"job_id": job_id, "max_pages": max_pages, "is_pro": is_pro}


@app.get("/api/progress/{job_id}")
async def stream_progress(job_id: str):
    """Server-Sent Events stream for live progress."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def event_stream():
        last_log_idx = 0
        while True:
            job = jobs.get(job_id, {})
            log = job.get("log", [])
            # Send new log lines
            for line in log[last_log_idx:]:
                yield f"data: {json.dumps({'type':'log','msg':line})}\n\n"
            last_log_idx = len(log)

            status = job.get("status","")
            page_count = len(job.get("pages",[]))
            yield f"data: {json.dumps({'type':'status','status':status,'count':page_count,'max':job.get('max_pages',10)})}\n\n"

            if status == "done":
                yield f"data: {json.dumps({'type':'done','count':page_count})}\n\n"
                break
            elif status == "error":
                yield f"data: {json.dumps({'type':'error','msg':job.get('error','Unknown error')})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Job not ready")
    result = job["result"]
    ext  = job["ext"]
    mime = job["mime"]
    slug = re.sub(r'[^\w\-]', '_', urlparse(job["pages"][0]["url"]).path.strip("/"))[:40] if job["pages"] else "dump"
    filename = f"sitevac_{slug}.{ext}"
    return PlainTextResponse(result, media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# ── Stripe checkout ───────────────────────────────────────────────────────────

@app.post("/api/checkout")
async def create_checkout(plan: str = Query(...)):
    if not stripe.api_key:
        raise HTTPException(503, "Stripe not configured")
    price_id = PRICE_MONTHLY if plan == "monthly" else PRICE_ONETIME
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription" if plan == "monthly" else "payment",
            success_url=SUCCESS_URL,
            cancel_url=CANCEL_URL,
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature","")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET","")
    try:
        event = stripe.Webhook.construct_event(payload, sig, webhook_secret)
    except Exception:
        raise HTTPException(400, "Invalid signature")
    # Handle completed checkout — in prod, save to DB and issue JWT/token
    if event["type"] == "checkout.session.completed":
        pass  # TODO: save customer email → pro status
    return {"ok": True}


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f:
        return f.read()

@app.get("/success", response_class=HTMLResponse)
async def success_page():
    return """<!DOCTYPE html><html><body style="font-family:sans-serif;text-align:center;padding:4rem;background:#0f0f0f;color:#e0e0e0">
    <h1 style="color:#7eb8f7">🎉 You're Pro!</h1>
    <p>Your account is active. <a href="/" style="color:#7eb8f7">Start scraping →</a></p>
    </body></html>"""
