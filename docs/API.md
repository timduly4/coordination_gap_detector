# API Documentation

Comprehensive documentation for the Coordination Gap Detector REST API.

## Base URL

```
http://localhost:8000
```

For production deployments, replace with your production URL.

## Authentication

*Authentication is not yet implemented. This will be added in future milestones.*

## API Versioning

All API endpoints are versioned under `/api/v1/`.

## Interactive Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Available Endpoints

### Health & Status

#### GET /health

Basic health check endpoint for the application.

**Response:**
```json
{
  "status": "healthy",
  "environment": "development"
}
```

**Status Codes:**
- `200 OK` - Application is healthy

---

#### GET /health/detailed

Detailed health check with service connectivity status.

**Response:**
```json
{
  "status": "healthy",
  "environment": "development",
  "services": {
    "postgres": {
      "status": "connected",
      "url": "localhost:5432/coordination"
    },
    "redis": {
      "status": "not_implemented",
      "url": "redis://localhost:6379"
    },
    "chromadb": {
      "status": "connected",
      "collection": "coordination_messages",
      "document_count": 24,
      "persist_dir": "./data/chroma"
    }
  }
}
```

**Status Codes:**
- `200 OK` - Health check completed (check `status` field for actual health)

**Status Values:**
- `healthy` - All services are operational
- `degraded` - Some services are experiencing issues
- `unhealthy` - Critical services are down

---

### Search API

#### POST /api/v1/search/

Search for messages using semantic similarity with flexible filtering options.

**Request Body:**

```json
{
  "query": "OAuth implementation",
  "limit": 10,
  "threshold": 0.7,
  "source_types": ["slack"],
  "channels": ["#engineering"],
  "date_from": "2024-11-01T00:00:00Z",
  "date_to": "2024-12-01T23:59:59Z"
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text (1-1000 characters) |
| `limit` | integer | No | 10 | Maximum number of results (1-100) |
| `threshold` | float | No | 0.0 | Minimum similarity score (0.0-1.0) |
| `source_types` | array[string] | No | null | Filter by source types (e.g., ["slack", "github"]) |
| `channels` | array[string] | No | null | Filter by specific channels |
| `date_from` | datetime | No | null | Filter messages from this date (ISO 8601 format) |
| `date_to` | datetime | No | null | Filter messages until this date (ISO 8601 format) |

**Response:**

```json
{
  "results": [
    {
      "content": "Starting OAuth2 integration today. We need this for the mobile app authentication.",
      "source": "slack",
      "channel": "#engineering",
      "author": "alice@company.com",
      "timestamp": "2024-12-01T09:00:00Z",
      "score": 0.89,
      "message_id": 123,
      "external_id": "slack_msg_abc123",
      "message_metadata": {
        "reactions": [
          {"emoji": "thumbsup", "count": 3}
        ],
        "mentions": ["bob@company.com"]
      }
    }
  ],
  "total": 1,
  "query": "OAuth implementation",
  "query_time_ms": 45,
  "threshold": 0.7
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | List of matching messages |
| `results[].content` | string | Message content |
| `results[].source` | string | Source type (slack, github, etc.) |
| `results[].channel` | string | Channel or location name |
| `results[].author` | string | Message author (email or username) |
| `results[].timestamp` | datetime | When the message was created |
| `results[].score` | float | Similarity score (0.0-1.0) |
| `results[].message_id` | integer | Database message ID |
| `results[].external_id` | string | ID in the source system |
| `results[].message_metadata` | object | Additional metadata (reactions, mentions, etc.) |
| `total` | integer | Total number of results returned |
| `query` | string | The search query that was executed |
| `query_time_ms` | integer | Query execution time in milliseconds |
| `threshold` | float | Similarity threshold used |

**Status Codes:**
- `200 OK` - Search completed successfully
- `400 Bad Request` - Invalid request parameters
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error during search

**Examples:**

```bash
# Basic search
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth implementation"}'

# Search with filters
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication",
    "limit": 5,
    "threshold": 0.8,
    "source_types": ["slack"],
    "channels": ["#engineering", "#security"]
  }'

# Search with date range
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deployment",
    "date_from": "2024-11-01T00:00:00Z",
    "date_to": "2024-11-30T23:59:59Z"
  }'
```

**Use Cases:**

1. **Find relevant discussions**: Search for messages about a specific topic
2. **Filter by channel**: Limit results to specific channels or teams
3. **Time-based analysis**: Search within a specific date range
4. **High-relevance search**: Use threshold to get only highly similar results
5. **Cross-source search**: Find related content across multiple sources

---

#### GET /api/v1/search/health

Check the health and status of the search service and its dependencies.

**Response:**

```json
{
  "status": "healthy",
  "vector_store": {
    "connected": true,
    "collection": "coordination_messages",
    "document_count": 24
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Service status: "healthy", "degraded", or "unhealthy" |
| `vector_store.connected` | boolean | Whether vector store is accessible |
| `vector_store.collection` | string | Name of the vector store collection |
| `vector_store.document_count` | integer | Number of documents in the collection |

**Status Codes:**
- `200 OK` - Health check completed
- `500 Internal Server Error` - Health check failed

**Example:**

```bash
curl http://localhost:8000/api/v1/search/health
```

---

## Error Handling

The API uses standard HTTP status codes and returns error details in a consistent format.

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

For validation errors (422):

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Common Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Endpoint not found |
| 422 | Unprocessable Entity | Validation error |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Examples

**Missing required field:**
```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Invalid parameter value:**
```json
{
  "detail": [
    {
      "loc": ["body", "limit"],
      "msg": "ensure this value is greater than or equal to 1",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

**Query too long:**
```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "ensure this value has at most 1000 characters",
      "type": "value_error.any_str.max_length"
    }
  ]
}
```

---

## Rate Limiting

*Rate limiting is not yet implemented. This will be added in future milestones.*

Planned rate limits:
- **Search API**: 100 requests per minute per IP
- **Health endpoints**: Unlimited

---

## Pagination

Currently, the search API uses a simple limit-based approach. Pagination will be improved in future milestones.

**Current approach:**
```json
{
  "query": "example",
  "limit": 10
}
```

**Planned improvements:**
- Cursor-based pagination for large result sets
- Offset-based pagination as alternative
- Total count of all matching results

---

## Best Practices

### Query Optimization

1. **Use specific queries**: More specific queries return better results
   - Good: "OAuth2 authentication implementation"
   - Less good: "auth"

2. **Set appropriate thresholds**: Higher thresholds for precision, lower for recall
   - High precision: `threshold: 0.8`
   - High recall: `threshold: 0.3`

3. **Use filters**: Narrow results with source types and channels
   ```json
   {
     "query": "deployment",
     "source_types": ["slack"],
     "channels": ["#engineering"]
   }
   ```

4. **Limit results**: Request only what you need
   ```json
   {
     "query": "example",
     "limit": 5
   }
   ```

### Performance Tips

1. **Batch requests**: Combine related searches when possible
2. **Use health checks**: Verify service status before heavy operations
3. **Monitor query times**: Use `query_time_ms` to track performance
4. **Cache results**: Cache search results for frequently used queries

### Error Handling

1. **Always check status codes**: Don't assume 200 OK
2. **Parse error details**: Use error messages for debugging
3. **Implement retries**: Retry on 5xx errors with exponential backoff
4. **Validate inputs**: Validate on client side to reduce 422 errors

---

## Future Endpoints

The following endpoints are planned for future milestones:

### Gap Detection API (Coming Soon)

- `POST /api/v1/gaps/detect` - Detect coordination gaps
- `GET /api/v1/gaps/{gap_id}` - Get gap details
- `PUT /api/v1/gaps/{gap_id}` - Update gap status
- `GET /api/v1/gaps/` - List all gaps

### Insights API (Coming Soon)

- `GET /api/v1/insights/` - Get AI-generated insights
- `POST /api/v1/insights/generate` - Generate insights for data

### Metrics API (Coming Soon)

- `GET /api/v1/metrics/detection-quality` - Detection quality metrics
- `GET /api/v1/metrics/search-quality` - Search quality metrics

---

## Changelog

### v0.1.0 (December 2024)

**Added:**
- Search API endpoint with semantic similarity
- Health check endpoints
- Interactive API documentation

**Coming Next:**
- Gap detection endpoints
- Authentication and authorization
- Rate limiting

---

## Support

For questions, issues, or feature requests:

- **GitHub Issues**: https://github.com/timduly4/coordination_gap_detector/issues
- **Documentation**: See [README.md](../README.md)
- **Testing Guide**: See [TESTING.md](../TESTING.md)

---

Built with [Claude Code](https://claude.com/claude-code) for AI-native development.
