# SiteVac Specification

## Product Vision
- **One-sentence description**: Intelligent website scraper that converts sites into AI-readable markdown for LLM context loading
- **Target user**: AI developers, researchers, and power users who need to feed website content to LLMs
- **Core value proposition**: Clean, structured markdown extraction from any website, optimized for LLM token efficiency

## Technical Stack
- **Frontend framework**: N/A (API service)
- **Backend**: Python FastAPI on Render
- **Database**: None (stateless service)
- **Hosting**: Render (cloud platform)
- **Key dependencies**:
  - FastAPI for API framework
  - BeautifulSoup4 for HTML parsing
  - Trafilatura for content extraction
  - html2text for markdown conversion
  - requests for HTTP fetching

## User Stories

### Must-have Stories (MVP)
1. As a developer, I can submit any URL and get clean markdown output
2. As a developer, I can chunk large sites into token-optimized sections
3. As a developer, I can extract structured data (JSON) along with content
4. As a researcher, I can batch process multiple URLs at once
5. As a user, I can get usage statistics and rate limits

### Should-have Stories (v1.1)
1. As a user, I can authenticate via API key
2. As a user, I can set custom extraction rules (CSS selectors)
3. As a user, I can cache results for repeated requests
4. As a developer, I can use SDKs for Python/JavaScript

### Nice-to-have Stories (Future)
1. Browser automation for JavaScript-heavy sites (Playwright)
2. Scheduled scraping with webhook notifications
3. Full site crawling with sitemap generation
4. OCR for images within pages
5. PDF extraction alongside HTML

## Feature Specification

### Feature 1: URL-to-Markdown Conversion
- **Description**: Primary API endpoint for converting any URL to clean markdown
- **Input**: URL + extraction options
- **Output**: Clean markdown with metadata
- **Capabilities**:
  - Remove boilerplate (nav, footer, ads)
  - Preserve semantic structure (headings, lists, tables)
  - Extract metadata (title, description, author, date)
  - Handle various content types (HTML, XML)

### Feature 2: Intelligent Chunking
- **Description**: Split large content into token-optimized chunks
- **Capabilities**:
  - Configurable chunk size (default: 2000 tokens)
  - Preserve context at chunk boundaries
  - Overlap options for continuity
  - Return chunk count and metadata
  - Named chunks with headers

### Feature 3: Structured Data Extraction
- **Description**: Extract structured JSON alongside markdown
- **Extracted data**:
  - Open Graph metadata
  - Schema.org JSON-LD
  - Article metadata (author, date, reading time)
  - Links (internal/external)
  - Images with alt text
  - Tables as JSON arrays

### Feature 4: Batch Processing
- **Description**: Process multiple URLs in single request
- **Features**:
  - Concurrent fetching with rate limiting
  - Progress tracking via status endpoint
  - Aggregate results in single response
  - Error handling per URL
  - Priority queuing

### Feature 5: Output Formats
- **Description**: Multiple output format options
- **Formats**:
  - Clean markdown (default)
  - JSON with metadata
  - XML for structured documents
  - Plain text for simple extraction
  - HTML sanitization

## API Specification

### Endpoints

#### POST /api/scrape
- **Purpose**: Scrape single URL and return markdown
- **Body**:
  ```json
  {
    "url": "https://example.com",
    "options": {
      "chunk_size": 2000,
      "extract_metadata": true,
      "extract_links": false,
      "output_format": "markdown"
    }
  }
  ```
- **Response**:
  ```json
  {
    "url": "https://example.com",
    "title": "Example Article",
    "markdown": "# Example Article\n\nContent here...",
    "metadata": {
      "author": "John Doe",
      "date": "2026-03-16",
      "reading_time": 5,
      "word_count": 1200
    },
    "chunks": [
      {"id": 1, "text": "...", "token_count": 1950}
    ],
    "token_count": 1950,
    "processed_at": "2026-03-16T18:41:00Z"
  }
  ```

#### POST /api/batch
- **Purpose**: Batch process multiple URLs
- **Body**:
  ```json
  {
    "urls": ["url1", "url2", "url3"],
    "options": {
      "concurrency": 3,
      "chunk_size": 2000
    }
  }
  ```
- **Response**:
  ```json
  {
    "job_id": "uuid",
    "status": "processing",
    "total": 3,
    "completed": 0
  }
  ```

#### GET /api/batch/{job_id}
- **Purpose**: Get batch job status
- **Response**:
  ```json
  {
    "job_id": "uuid",
    "status": "completed",
    "results": [...],
    "errors": [],
    "total_tokens": 5000
  }
  ```

#### POST /api/extract
- **Purpose**: Extract structured data only (no markdown)
- **Response**:
  ```json
  {
    "metadata": {...},
    "links": [...],
    "images": [...],
    "schema_org": {...}
  }
  ```

#### GET /health
- **Purpose**: Service health check
- **Response**: `{ "status": "healthy", "version": "1.0.0" }`

### Auth Method
- **API Key**: X-API-Key header (optional for free tier)
- **Rate Limiting**:
  - Free: 100 requests/day
  - Paid: 10000 requests/day
  - Enterprise: Unlimited

### Rate Limits
- Free: 100 requests/day, 10 requests/minute
- Paid: 10000 requests/day, 100 requests/minute
- Enterprise: Custom

## Database Schema

### No Persistent Database
SiteVac is stateless by design. No database required.

### Future Database (for caching and accounts)
```sql
CREATE TABLE users (
  id UUID PRIMARY KEY,
  api_key TEXT UNIQUE NOT NULL,
  plan TEXT CHECK (plan IN ('free', 'paid', 'enterprise')),
  requests_today INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE cache (
  url_hash TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  result JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE jobs (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  urls TEXT[] NOT NULL,
  status TEXT NOT NULL,
  results JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);
```

## UI/UX Specification

### Key Pages
SiteVac is primarily an API service, but has a minimal landing page:

1. **Landing Page** (`/`)
   - Hero: "Transform Any Website into LLM-Ready Markdown"
   - Live demo input (paste URL, see result)
   - API documentation link
   - Pricing tiers
   - CTA: "Get API Key"

2. **Documentation Page** (`/docs`)
   - API endpoint reference
   - Code examples (Python, JavaScript, curl)
   - Best practices guide
   - Rate limits explanation

### User Flows

#### Flow 1: Quick Test
1. Visit landing page
2. Paste URL in demo input
3. See markdown output instantly
4. Copy API endpoint
5. Generate API key

#### Flow 2: API Integration
1. Read documentation
2. Get API key
3. Install SDK (npm install sitevac)
4. Call from application
5. Handle response

### Design System

#### Typography
- **Headlines**: Space Grotesk (weight 700, tracking -0.02em)
- **Body**: Inter Variable (weight 400-500) - **Note: Update to Satoshi**
- **Code**: JetBrains Mono (weight 400)

#### Colors
- **Primary**: #DC2626 (Red) - Developer tool aesthetic
- **Background**: #0A0A0A (Dark mode by default)
- **Card**: #1A1A1A
- **Text**: #F5F5F5
- **Accent**: #F59E0B (Amber for code highlights)

#### Spacing
- Base unit: 4px (tight for developer tools)
- Code blocks: 16px padding
- Section gaps: 24px

### Signature Moment
The **Live Demo with Token Counter** - An interactive URL input that shows:
1. URL being fetched in real-time (animated progress)
2. Token count updating as content loads
3. Markdown preview rendering with syntax highlighting
4. Copy button with "Copied!" animation
5. Token savings calculation vs raw HTML

This creates immediate understanding of the value proposition: "See exactly what your LLM will receive."

## Pricing Model

### Current Pricing
- Free tier with rate limits
- Contact for enterprise

### Recommended Pricing
- **Free**: $0/month
  - 100 requests/day
  - 10 requests/minute
  - Community support

- **Developer**: $19/month
  - 10,000 requests/day
  - 100 requests/minute
  - Priority support
  - Batch processing

- **Enterprise**: $99/month
  - Unlimited requests
  - No rate limits
  - Dedicated support
  - Custom integrations
  - SLA guarantee

### Rationale
- Developer tools need generous free tiers
- $19/mo is impulse-buy territory for solo developers
- Enterprise tier for companies using at scale
- Rate limits prevent abuse
- Usage-based pricing possible in future

## Marketing Strategy

### Target Keywords
- "website to markdown" (medium competition)
- "web scraper API" (high competition)
- "LLM context loader" (emerging, low competition)
- "content extraction API" (medium competition)
- "HTML to markdown converter" (medium competition)

### Competitors
1. **Jina Reader** - Similar concept, jina.ai/reader
2. **Firecrawl** - Full web scraping platform
3. **Readability.js** - Open source, requires implementation
4. **Custom scrapers** - Developers build their own

### Differentiation
1. **Token-optimized**: Designed specifically for LLMs, not general scraping
2. **Chunking built-in**: Split content intelligently for context windows
3. **Metadata extraction**: Rich metadata for better LLM understanding
4. **Developer-friendly**: Simple API, clear docs, generous free tier
5. **Fast and stateless**: No complex setup, instant results

### Distribution Channels
1. **Hacker News**: Developer tools forum (Show HN)
2. **Reddit**: r/languagelearning, r/LocalLLaMA, r/MachineLearning
3. **Product Hunt**: Launch to developer community
4. **Dev.to**: Technical tutorials on LLM context loading
5. **Twitter**: #BuildInPublic with AI Twitter
6. **GitHub**: Open source SDKs and integrations

## Success Metrics

### Key Metrics to Track
1. **Daily API Requests**: Volume of scraping requests
2. **Unique Users**: API keys in active use
3. **Token Efficiency**: Average token reduction ratio
4. **API Response Time**: Performance metric
5. **Error Rate**: Failed requests

### Goals (30/60/90 Days)

#### Day 30
- 100 API keys issued
- 10,000 total requests
- Average response time < 500ms
- 50% token reduction average
- Featured in 1 newsletter

#### Day 60
- 500 API keys issued
- 100,000 total requests
- 10 paying customers
- SDK for Python released
- Featured in 3 developer blogs

#### Day 90
- 1,000 API keys issued
- 1M total requests
- 25 paying customers
- SDK for JavaScript released
- $500 MRR
- Integration with 2 AI platforms

## Technical Debt / Known Issues

### Current Issues
1. **No authentication**: Open API, can be abused
2. **No caching**: Every request fetches fresh content
3. **JavaScript sites**: Can't render client-side JS
4. **Rate limiting**: Only basic rate limiting
5. **No SDKs**: Direct API calls only

### Priority Fixes Needed

#### High Priority
1. Implement API key authentication
2. Add Redis caching for repeated requests
3. Set up proper rate limiting (token bucket)
4. Add request logging for analytics

#### Medium Priority
1. Implement Playwright for JS-heavy sites
2. Create Python SDK
3. Create JavaScript SDK
4. Add webhook support for batch completion

#### Low Priority
1. Full site crawling capability
2. PDF extraction with OCR
3. Scheduled scraping
4. Custom extraction rules config
5. Team accounts and usage dashboard

## File Locations

### Repository Path
`/home/localadmin/.openclaw/workspace/projects/SiteVac/`

### Key Files to Modify
1. `main.py` - Main FastAPI application (core API logic)
2. `README.md` - Documentation and setup
3. `requirements.txt` - Python dependencies
4. `render.yaml` - Render deployment configuration

### Configuration Files
- `render.yaml` - Render deployment
- `requirements.txt` - Python package dependencies
- `.env.example` - Environment variables template