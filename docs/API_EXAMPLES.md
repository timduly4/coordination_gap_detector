# API Usage Examples

Complete guide to using the Coordination Gap Detection API with practical examples.

## Table of Contents

- [Quick Start](#quick-start)
- [Gap Detection](#gap-detection)
- [Listing Gaps](#listing-gaps)
- [Retrieving Specific Gaps](#retrieving-specific-gaps)
- [Filtering and Pagination](#filtering-and-pagination)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Integration Patterns](#integration-patterns)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

### Base URL

```
Local Development: http://localhost:8000
Production: https://your-domain.com
```

### API Version

Current API version: `v1`

All endpoints are prefixed with `/api/v1`

### Authentication

```bash
# Currently no authentication required for local development
# Production will use API keys:
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/api/v1/gaps
```

---

## Gap Detection

### Basic Detection

Detect coordination gaps across the last 30 days:

```bash
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30
  }'
```

**Response**:
```json
{
  "gaps": [
    {
      "id": "gap_abc123",
      "type": "duplicate_work",
      "title": "Two teams building OAuth integration",
      "topic": "OAuth2 integration",
      "teams_involved": ["platform-team", "auth-team"],
      "impact_score": 0.89,
      "impact_tier": "HIGH",
      "confidence": 0.87,
      "detected_at": "2024-12-27T10:30:00Z"
    }
  ],
  "metadata": {
    "total_gaps": 1,
    "critical_gaps": 0,
    "high_gaps": 1,
    "medium_gaps": 0,
    "low_gaps": 0,
    "detection_time_ms": 3842,
    "messages_analyzed": 1250,
    "clusters_found": 15
  }
}
```

### Detection with Filters

Detect high-impact gaps from specific sources:

```bash
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "sources": ["slack", "github"],
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.7,
    "teams": ["engineering", "platform-team", "auth-team"]
  }'
```

**Parameters**:
- `timeframe_days` (int, 1-365): How far back to analyze (default: 30)
- `sources` (array): Data sources to include (default: ["slack"])
- `gap_types` (array): Types of gaps to detect (default: ["duplicate_work"])
- `min_impact_score` (float, 0-1): Minimum impact threshold (default: 0.0)
- `teams` (array, optional): Filter to specific teams

### Detection with Evidence

Include detailed evidence in results:

```bash
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.6,
    "include_evidence": true
  }'
```

**Response with Evidence**:
```json
{
  "gaps": [
    {
      "id": "gap_abc123",
      "type": "duplicate_work",
      "title": "Two teams building OAuth integration",
      "impact_score": 0.89,
      "evidence": [
        {
          "source": "slack",
          "channel": "#platform",
          "content": "Starting OAuth2 implementation for API gateway",
          "author": "alice@company.com",
          "team": "platform-team",
          "timestamp": "2024-12-01T09:00:00Z",
          "relevance_score": 0.95
        },
        {
          "source": "slack",
          "channel": "#auth-team",
          "content": "We're building OAuth support for the auth service",
          "author": "bob@company.com",
          "team": "auth-team",
          "timestamp": "2024-12-01T14:20:00Z",
          "relevance_score": 0.92
        }
      ],
      "temporal_overlap": {
        "start": "2024-12-01T00:00:00Z",
        "end": "2024-12-15T00:00:00Z",
        "overlap_days": 14
      },
      "verification": {
        "is_duplicate": true,
        "confidence": 0.85,
        "reasoning": "Both teams are implementing OAuth2 independently without coordination",
        "evidence": [
          "Platform team started OAuth implementation on Dec 1",
          "Auth team began parallel OAuth work 4 hours later"
        ],
        "recommendation": "Connect alice@company.com and bob@company.com immediately",
        "overlap_ratio": 0.8
      },
      "insight": "Platform and Auth teams independently implementing OAuth2",
      "recommendation": "Consolidate efforts by having one team lead",
      "estimated_cost": {
        "engineering_hours": 85,
        "dollar_value": 8500,
        "explanation": "2 teams × ~40 hours each + coordination overhead"
      }
    }
  ],
  "metadata": {
    "total_gaps": 1,
    "high_gaps": 1,
    "detection_time_ms": 4123,
    "messages_analyzed": 1250
  }
}
```

### Focused Detection

Detect gaps for specific teams only:

```bash
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 14,
    "teams": ["platform-team", "auth-team"],
    "min_impact_score": 0.5
  }'
```

---

## Listing Gaps

### List All Gaps

```bash
curl http://localhost:8000/api/v1/gaps
```

**Response**:
```json
{
  "gaps": [
    {
      "id": "gap_abc123",
      "type": "duplicate_work",
      "title": "Two teams building OAuth integration",
      "impact_score": 0.89,
      "teams_involved": ["platform-team", "auth-team"],
      "detected_at": "2024-12-27T10:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "limit": 10,
  "has_more": false
}
```

### List with Filters

Filter gaps by type and impact:

```bash
curl "http://localhost:8000/api/v1/gaps?gap_type=duplicate_work&min_impact_score=0.7"
```

**Query Parameters**:
- `gap_type` (string): Filter by gap type
- `min_impact_score` (float, 0-1): Minimum impact score
- `teams` (array): Filter by team involvement
- `page` (int, ≥1): Page number (default: 1)
- `limit` (int, 1-100): Results per page (default: 10)

### List with Pagination

```bash
# Page 1 (first 5 results)
curl "http://localhost:8000/api/v1/gaps?page=1&limit=5"

# Page 2 (next 5 results)
curl "http://localhost:8000/api/v1/gaps?page=2&limit=5"
```

**Response**:
```json
{
  "gaps": [...],
  "total": 23,
  "page": 2,
  "limit": 5,
  "has_more": true
}
```

### List High-Impact Gaps Only

```bash
curl "http://localhost:8000/api/v1/gaps?min_impact_score=0.8&limit=20"
```

---

## Retrieving Specific Gaps

### Get Gap by ID

```bash
curl http://localhost:8000/api/v1/gaps/gap_abc123
```

**Response** (full gap object):
```json
{
  "id": "gap_abc123",
  "type": "duplicate_work",
  "title": "Two teams building OAuth integration",
  "topic": "OAuth2 integration",
  "teams_involved": ["platform-team", "auth-team"],
  "impact_score": 0.89,
  "impact_tier": "HIGH",
  "confidence": 0.87,
  "evidence": [
    {
      "source": "slack",
      "content": "Starting OAuth2 implementation...",
      "author": "alice@company.com",
      "timestamp": "2024-12-01T09:00:00Z",
      "relevance_score": 0.95
    }
  ],
  "temporal_overlap": {
    "start": "2024-12-01T00:00:00Z",
    "end": "2024-12-15T00:00:00Z",
    "overlap_days": 14
  },
  "verification": {
    "is_duplicate": true,
    "confidence": 0.85,
    "reasoning": "Both teams implementing OAuth2 independently",
    "recommendation": "Connect teams immediately"
  },
  "insight": "Platform and Auth teams independently implementing OAuth2",
  "recommendation": "Consolidate efforts under one team lead",
  "estimated_cost": {
    "engineering_hours": 85,
    "dollar_value": 8500
  },
  "detected_at": "2024-12-27T10:30:00Z"
}
```

### Gap Not Found

```bash
curl http://localhost:8000/api/v1/gaps/nonexistent_gap
```

**Response** (404):
```json
{
  "detail": "Gap with ID 'nonexistent_gap' not found"
}
```

---

## Filtering and Pagination

### Complex Filtering

Combine multiple filters:

```bash
curl -G "http://localhost:8000/api/v1/gaps" \
  --data-urlencode "gap_type=duplicate_work" \
  --data-urlencode "min_impact_score=0.6" \
  --data-urlencode "teams=platform-team" \
  --data-urlencode "teams=auth-team" \
  --data-urlencode "limit=20"
```

### Pagination Pattern

Iterate through all gaps:

```bash
#!/bin/bash
# Fetch all gaps with pagination

page=1
limit=10
has_more=true

while [ "$has_more" = "true" ]; do
  echo "Fetching page $page..."

  response=$(curl -s "http://localhost:8000/api/v1/gaps?page=$page&limit=$limit")

  # Extract has_more field
  has_more=$(echo "$response" | jq -r '.has_more')

  # Process gaps
  echo "$response" | jq '.gaps[]'

  ((page++))
done
```

### Sort and Filter

```bash
# Get top 10 highest-impact gaps
curl "http://localhost:8000/api/v1/gaps?min_impact_score=0.7&limit=10"

# Get recent gaps for specific team
curl "http://localhost:8000/api/v1/gaps?teams=platform-team&limit=5"
```

---

## Error Handling

### Validation Errors (400)

**Invalid timeframe**:
```bash
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{"timeframe_days": 0}'
```

**Response**:
```json
{
  "detail": [
    {
      "loc": ["body", "timeframe_days"],
      "msg": "ensure this value is greater than or equal to 1",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

**Invalid impact score**:
```bash
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{"min_impact_score": 1.5}'
```

**Response**:
```json
{
  "detail": [
    {
      "loc": ["body", "min_impact_score"],
      "msg": "ensure this value is less than or equal to 1.0",
      "type": "value_error.number.not_le"
    }
  ]
}
```

### Server Errors (500)

```bash
# Internal server error
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{"timeframe_days": 30}'
```

**Response**:
```json
{
  "detail": "Gap detection failed. Please try again later."
}
```

### Service Unavailable (503)

**LLM rate limit exceeded**:
```json
{
  "detail": "Gap detection service temporarily unavailable. Please try again in 60 seconds."
}
```

### Error Handling Pattern

```python
import requests

def detect_gaps_with_retry(params, max_retries=3):
    """Detect gaps with automatic retry on transient errors."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/gaps/detect",
                json=params,
                timeout=30
            )

            # Success
            if response.status_code == 200:
                return response.json()

            # Client error (don't retry)
            if 400 <= response.status_code < 500:
                raise ValueError(f"Invalid request: {response.json()}")

            # Server error (retry)
            if response.status_code >= 500:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise Exception(f"Server error: {response.json()}")

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise Exception("Request timed out after retries")

    raise Exception(f"Failed after {max_retries} attempts")
```

---

## Rate Limiting

### Current Limits

```
Development: No rate limits
Production:
  - 60 requests per minute per IP
  - 1000 requests per hour per API key
  - 10 concurrent detection requests
```

### Rate Limit Headers

```bash
curl -i http://localhost:8000/api/v1/gaps
```

**Response Headers**:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1703682000
```

### Handling Rate Limits

```python
import time
import requests

def api_call_with_rate_limit(url):
    """Make API call respecting rate limits."""
    response = requests.get(url)

    # Check if rate limited
    if response.status_code == 429:
        # Get reset time from header
        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
        wait_seconds = reset_time - int(time.time())

        print(f"Rate limited. Waiting {wait_seconds} seconds...")
        time.sleep(wait_seconds + 1)

        # Retry
        return api_call_with_rate_limit(url)

    return response.json()
```

---

## Integration Patterns

### Python Client Example

```python
import requests
from typing import List, Dict, Optional

class GapDetectionClient:
    """Client for Gap Detection API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def detect_gaps(
        self,
        timeframe_days: int = 30,
        sources: Optional[List[str]] = None,
        min_impact_score: float = 0.6,
        include_evidence: bool = True
    ) -> Dict:
        """Detect coordination gaps."""
        response = self.session.post(
            f"{self.base_url}/api/v1/gaps/detect",
            json={
                "timeframe_days": timeframe_days,
                "sources": sources or ["slack"],
                "gap_types": ["duplicate_work"],
                "min_impact_score": min_impact_score,
                "include_evidence": include_evidence
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def list_gaps(
        self,
        gap_type: Optional[str] = None,
        min_impact_score: float = 0.0,
        teams: Optional[List[str]] = None,
        page: int = 1,
        limit: int = 10
    ) -> Dict:
        """List detected gaps with pagination."""
        params = {
            "page": page,
            "limit": limit,
            "min_impact_score": min_impact_score
        }
        if gap_type:
            params["gap_type"] = gap_type
        if teams:
            params["teams"] = teams

        response = self.session.get(
            f"{self.base_url}/api/v1/gaps",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def get_gap(self, gap_id: str) -> Dict:
        """Get specific gap by ID."""
        response = self.session.get(
            f"{self.base_url}/api/v1/gaps/{gap_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()

# Usage
client = GapDetectionClient()

# Detect gaps
result = client.detect_gaps(timeframe_days=30, min_impact_score=0.7)
print(f"Found {result['metadata']['total_gaps']} gaps")

# List high-impact gaps
gaps = client.list_gaps(min_impact_score=0.8, limit=5)
for gap in gaps['gaps']:
    print(f"- {gap['title']} (impact: {gap['impact_score']})")

# Get specific gap
gap = client.get_gap("gap_abc123")
print(f"Gap: {gap['title']}")
print(f"Teams: {', '.join(gap['teams_involved'])}")
```

### JavaScript/TypeScript Client

```typescript
class GapDetectionClient {
  constructor(private baseUrl: string = "http://localhost:8000") {}

  async detectGaps(params: {
    timeframeDays?: number;
    sources?: string[];
    minImpactScore?: number;
    includeEvidence?: boolean;
  }): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/v1/gaps/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        timeframe_days: params.timeframeDays || 30,
        sources: params.sources || ["slack"],
        gap_types: ["duplicate_work"],
        min_impact_score: params.minImpactScore || 0.6,
        include_evidence: params.includeEvidence ?? true,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  }

  async listGaps(params: {
    gapType?: string;
    minImpactScore?: number;
    teams?: string[];
    page?: number;
    limit?: number;
  }): Promise<any> {
    const queryParams = new URLSearchParams({
      page: String(params.page || 1),
      limit: String(params.limit || 10),
      min_impact_score: String(params.minImpactScore || 0.0),
    });

    if (params.gapType) queryParams.append("gap_type", params.gapType);
    if (params.teams) {
      params.teams.forEach(team => queryParams.append("teams", team));
    }

    const response = await fetch(
      `${this.baseUrl}/api/v1/gaps?${queryParams}`
    );

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  }
}

// Usage
const client = new GapDetectionClient();

// Detect gaps
const result = await client.detectGaps({
  timeframeDays: 30,
  minImpactScore: 0.7,
});
console.log(`Found ${result.metadata.total_gaps} gaps`);

// List gaps
const gaps = await client.list_gaps({
  minImpactScore: 0.8,
  limit: 5,
});
gaps.gaps.forEach(gap => {
  console.log(`- ${gap.title} (impact: ${gap.impact_score})`);
});
```

### Webhook Integration

Receive gap notifications via webhook:

```python
from flask import Flask, request

app = Flask(__name__)

@app.route("/webhook/gaps", methods=["POST"])
def gap_webhook():
    """Receive gap detection notifications."""
    data = request.json

    gap = data["gap"]
    print(f"New gap detected: {gap['title']}")
    print(f"Impact: {gap['impact_score']} ({gap['impact_tier']})")
    print(f"Teams: {', '.join(gap['teams_involved'])}")

    # Send notification (Slack, email, etc.)
    if gap["impact_tier"] in ["CRITICAL", "HIGH"]:
        send_alert_to_slack(gap)

    return {"status": "received"}

def send_alert_to_slack(gap):
    """Send Slack notification for high-impact gaps."""
    import requests

    requests.post(
        "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        json={
            "text": f"🚨 High-impact coordination gap detected: {gap['title']}",
            "attachments": [{
                "color": "danger",
                "fields": [
                    {"title": "Impact", "value": gap["impact_tier"], "short": True},
                    {"title": "Teams", "value": ", ".join(gap["teams_involved"]), "short": True},
                    {"title": "Recommendation", "value": gap["recommendation"]},
                ]
            }]
        }
    )
```

---

## Advanced Usage

### Batch Detection

Detect gaps across multiple timeframes:

```python
def detect_gaps_batch(timeframes: List[int]) -> Dict[int, Dict]:
    """Detect gaps for multiple timeframes."""
    client = GapDetectionClient()
    results = {}

    for days in timeframes:
        print(f"Detecting gaps for last {days} days...")
        results[days] = client.detect_gaps(
            timeframe_days=days,
            min_impact_score=0.6
        )

    return results

# Detect for 7, 14, 30 days
results = detect_gaps_batch([7, 14, 30])

for days, result in results.items():
    total = result['metadata']['total_gaps']
    print(f"Last {days} days: {total} gaps")
```

### Custom Impact Thresholds

Different thresholds for different teams:

```python
def detect_critical_gaps_by_team(teams: List[str]) -> Dict[str, List]:
    """Detect critical gaps for specific teams."""
    client = GapDetectionClient()
    critical_gaps = {}

    for team in teams:
        result = client.detect_gaps(
            timeframe_days=30,
            teams=[team],
            min_impact_score=0.8  # Critical only
        )

        critical_gaps[team] = result['gaps']

    return critical_gaps

# Find critical gaps for each team
gaps = detect_critical_gaps_by_team([
    "platform-team",
    "auth-team",
    "backend-team"
])

for team, team_gaps in gaps.items():
    print(f"{team}: {len(team_gaps)} critical gaps")
```

### Continuous Monitoring

Monitor for new gaps periodically:

```python
import time
from datetime import datetime

def monitor_gaps(check_interval_seconds: int = 3600):
    """Monitor for new gaps continuously."""
    client = GapDetectionClient()
    known_gap_ids = set()

    while True:
        print(f"[{datetime.now()}] Checking for gaps...")

        result = client.detect_gaps(
            timeframe_days=7,  # Check last week
            min_impact_score=0.7
        )

        # Find new gaps
        for gap in result['gaps']:
            if gap['id'] not in known_gap_ids:
                print(f"NEW GAP: {gap['title']}")
                print(f"  Impact: {gap['impact_score']}")
                print(f"  Teams: {', '.join(gap['teams_involved'])}")

                # Alert on new high-impact gaps
                if gap['impact_tier'] in ['CRITICAL', 'HIGH']:
                    send_alert(gap)

                known_gap_ids.add(gap['id'])

        # Wait before next check
        time.sleep(check_interval_seconds)

# Monitor every hour
monitor_gaps(check_interval_seconds=3600)
```

---

## Summary

**Quick Reference**:

```bash
# Detect gaps
POST /api/v1/gaps/detect

# List gaps
GET /api/v1/gaps

# Get specific gap
GET /api/v1/gaps/{gap_id}

# Health check
GET /api/v1/gaps/health
```

**Best Practices**:
1. Start with high impact threshold (0.7+)
2. Include evidence for actionable insights
3. Use pagination for large result sets
4. Implement retry logic for transient errors
5. Respect rate limits
6. Monitor for high-impact gaps

**Next Steps**:
- [Gap Detection Methodology](GAP_DETECTION.md)
- [Entity Extraction Guide](ENTITY_EXTRACTION.md)
- [Main README](../README.md)

---

**Last Updated**: December 2024
**Version**: 1.0
**Milestone**: 3H - Testing & Documentation
