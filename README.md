# SiteVac 🌀

**Flatten any docs or wiki site into a single AI-ready file.**

Paste a URL. SiteVac crawls every page under that path, strips nav/boilerplate, and stitches the content into one clean file you can drop straight into Claude, ChatGPT, or any LLM.

🔗 **[sitevac.onrender.com](https://sitevac.onrender.com)**

---

## Why

Feeding documentation into an AI usually means copy-pasting dozens of pages manually, or writing a one-off scraping script you'll throw away. SiteVac is that script, but with a UI, output format options, and sensible limits so it doesn't hammer servers.

## What it does

- Crawls every link under a given URL prefix (stays on the same domain + path)
- Strips nav, sidebars, headers, footers — extracts just the content
- Deduplicates boilerplate that repeats across pages
- Outputs as **TXT**, **Markdown**, **HTML** (with copy button), or **ZIP Pack**
- ZIP Pack includes `clean.txt`, `clean.md`, `clean.html`, plus pre-chunked files sized for GPT (`chunks_gpt.txt`), Claude (`chunks_claude.txt`), and OpenAI-compatible JSONL (`chunks_openclaw.jsonl`)
- Streams live progress via SSE so you can watch pages being scraped in real time

## Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI + Python 3.11 |
| Scraping | requests + BeautifulSoup + html2text |
| Payments | Stripe (monthly + lifetime) |
| Database | Supabase (PostgreSQL) |
| Hosting | Render |
| Frontend | Vanilla JS, no framework |

## Plans

| | Free | Pro |
|---|---|---|
| Pages per scrape | 10 | 200 |
| Output formats | TXT, MD, HTML, ZIP | TXT, MD, HTML, ZIP |
| Scrapes | Unlimited | Unlimited |
| Price | $0 | $7/mo or $29 lifetime |

## Run locally

```bash
git clone https://github.com/yourhandle/sitevac
cd sitevac
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
BASE_URL=http://localhost:8000
```

```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

For Stripe webhooks locally, use the [Stripe CLI](https://stripe.com/docs/stripe-cli):

```bash
stripe listen --forward-to localhost:8000/api/webhook
```

## Supabase schema

```sql
-- Pro users
create table pro_users (
  id uuid primary key default gen_random_uuid(),
  email text unique not null,
  active boolean default true,
  plan text,
  created_at timestamptz default now()
);

-- Lifetime counter (atomic increment via RPC)
create table sitevac_counters (
  key text primary key,
  value_int int default 0
);

-- Webhook idempotency
create table processed_webhooks (
  session_id text primary key,
  created_at timestamptz default now()
);
```

Enable RLS on all tables and restrict to `service_role`. See the [Supabase RLS docs](https://supabase.com/docs/guides/auth/row-level-security).

## Deploy to Render

The repo includes a `render.yaml`. Connect the repo in the Render dashboard, set your environment variables, and it deploys automatically on push.

## Notes

- 0.4s delay between requests to avoid hammering servers
- Only scrape sites you have permission to access
- Free tier runs on Render's free plan — expect ~30s cold start after 15min idle

## License

MIT
