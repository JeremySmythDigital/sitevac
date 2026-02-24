# 🌀 SiteVac

**Flatten any docs/wiki site into a single AI-ready file.**

Live scraping → SSE progress stream → download TXT/MD/HTML.

---

## Quick Deploy (Free on Render)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "initial"
gh repo create sitevac --public --push
```

### 2. Deploy to Render (free tier)
1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` — hit **Deploy**
4. Your app is live at `https://sitevac.onrender.com`

> **Note:** Render free tier sleeps after 15min of inactivity and takes ~30s to wake.
> Upgrade to Render Starter ($7/mo) to keep it always-on — worth it once you have users.

---

## Monetization Setup

### Option A: Carbon Ads (easiest, dev audience)
1. Apply at [carbonads.com](https://www.carbonads.com) — they accept small sites
2. Get your zone ID
3. In `templates/index.html`, uncomment the Carbon block and replace `YOUR_ID`
4. Typical revenue: **$1–3 CPM** for developer traffic = ~$20-100/mo at modest traffic

### Option B: Stripe Freemium (higher ceiling)

**1. Create Stripe products**
```
Dashboard → Products → Add Product

Product 1: "SiteVac Pro Monthly"
  → Price: $7.00 / month (recurring)
  → Copy price ID → paste as PRICE_MONTHLY in main.py

Product 2: "SiteVac Pro Lifetime"
  → Price: $19.00 (one-time)
  → Copy price ID → paste as PRICE_ONETIME in main.py
```

**2. Set environment variables in Render dashboard**
```
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
BASE_URL=https://your-sitevac-url.onrender.com
```

**3. Set up webhook**
- Stripe Dashboard → Webhooks → Add endpoint
- URL: `https://your-url.onrender.com/api/webhook`
- Events: `checkout.session.completed`

**4. Add a database to persist pro users** (optional but needed for proper auth)
- Render → New → PostgreSQL (free tier)
- Or use [Supabase](https://supabase.com) free tier
- Store `(stripe_customer_id, email, plan, active)` rows
- Issue a simple token/cookie on successful checkout

**Revenue projection:**
| Monthly visitors | Free→Pro conversion (2%) | Monthly revenue |
|-----------------|--------------------------|-----------------|
| 500             | 10 users                 | $70–$190        |
| 2,000           | 40 users                 | $280–$760       |
| 10,000          | 200 users                | $1,400–$3,800   |

---

## Traffic / SEO

Good landing page keywords:
- "flatten docs for ai"
- "scrape documentation into text"
- "deepwiki to text"
- "feed docs to chatgpt"
- "llm context from docs"

Post on:
- Hacker News "Show HN" (single biggest traffic driver for dev tools)
- r/LocalLLaMA, r/ChatGPT
- Twitter/X with a screen recording of it working
- ProductHunt

---

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Open http://localhost:8000
```

---

## Architecture

```
Browser
  │
  ├─ POST /api/scrape        → creates async job, returns job_id
  ├─ GET  /api/progress/:id  → SSE stream (live log lines + count)
  └─ GET  /api/download/:id  → streams completed file

FastAPI (single process, Render free tier)
  └─ asyncio.create_task() → scrape runs in thread pool
     └─ requests + BeautifulSoup + html2text
```

Jobs are in-memory (fine for free tier single process). For scale, swap with Redis + Celery.

---

## Upgrading for Scale

When you start hitting limits:

| Problem | Solution | Cost |
|---------|----------|------|
| Render sleeps | Render Starter tier | $7/mo |
| Jobs lost on restart | Redis job store | Upstash free tier |
| Multi-worker | Add Celery + Redis | ~$10/mo total |
| User auth | Supabase auth | Free tier |
| More pages | Async scraping (aiohttp) | Free (code change) |
