# Streamlit Dashboard

Interactive web dashboard for visualizing and exploring coordination gaps detected by the system.

## Features

### 🎯 Interactive Gap Detection
- Select from multiple pre-built scenarios (OAuth duplication, API redesign, auth migration, etc.)
- Configure detection settings (minimum impact score, evidence display)
- Run detection with a single click

### 📊 Rich Visualizations
- Impact distribution charts
- Color-coded gap cards by severity (Critical, High, Medium, Low)
- Evidence timeline display
- Cost estimation breakdown

### 💡 AI-Powered Insights
- Claude API-generated insights for each gap
- LLM verification results
- Actionable recommendations
- Temporal overlap analysis

### 💾 Export Capabilities
- Download results as JSON
- Include full gap details and evidence
- Timestamped exports for tracking

## Installation

Install the dashboard dependencies:

```bash
# Using uv (recommended)
uv pip install -e ".[dashboard]"

# Or using pip
pip install -e ".[dashboard]"
```

## Running the Dashboard

From the project root directory:

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your default browser at `http://localhost:8501`.

## Usage

1. **Select a Scenario** from the sidebar dropdown
   - `oauth_duplication` - Platform and Auth teams independently implementing OAuth (HIGH impact)
   - `api_redesign_duplication` - Two teams redesigning the API structure (HIGH impact)
   - `auth_migration_duplication` - Security and Platform teams migrating auth independently (MEDIUM-HIGH impact)
   - `similar_topics_different_scope` - Edge case: Similar topics but different scope (should NOT detect)
   - `sequential_work` - Edge case: Sequential work, no overlap (should NOT detect)
   - `intentional_collaboration` - Negative example: Teams collaborating properly (should NOT detect)

2. **Configure Settings**
   - Adjust minimum impact score threshold
   - Toggle evidence display on/off

3. **Run Detection**
   - Click "🚀 Detect Gaps" button
   - View real-time analysis progress

4. **Explore Results**
   - Review summary metrics (total gaps, critical count, estimated cost)
   - Filter gaps by impact tier (All, Critical, High, Medium, Low)
   - Expand gap cards to see detailed evidence
   - Review AI insights and recommendations

5. **Export Data**
   - Download results as JSON for further analysis
   - Share findings with stakeholders

## Dashboard Sections

### Header
- Main title and description
- Quick introduction to gap detection capabilities

### Sidebar
- **Configuration Panel**
  - Scenario selection
  - Detection settings
  - About section with tech stack details

### Main Content Area

#### Welcome Screen (before detection)
- How-to guide
- Available scenarios preview
- Feature highlights

#### Results Screen (after detection)
- **Summary Metrics**
  - Total gaps detected
  - Critical/High/Medium/Low counts
  - Estimated total cost

- **Impact Distribution**
  - Bar chart of gaps by severity
  - Average impact score
  - Average confidence
  - Total teams involved

- **Gap Details**
  - Tabbed interface by severity
  - Expandable gap cards with:
    - Impact score, confidence, teams involved
    - AI-generated insight
    - Actionable recommendation
    - Cost breakdown (hours + dollar value)
    - LLM verification results
    - Temporal overlap analysis
    - Evidence items (up to 10 shown)

- **Export Section**
  - JSON download button
  - Timestamped filenames

## Available Scenarios

### Positive Examples (Should Detect Gaps)

**OAuth Duplication** (`oauth_duplication`)
- Platform and Auth teams independently implementing OAuth2
- 24+ messages over 14 days
- Both teams: authorization code flow, PKCE, JWT tokens, database schemas
- Expected: HIGH impact duplicate work gap

**API Redesign** (`api_redesign_duplication`)
- Platform and Backend teams independently redesigning REST API
- 16+ messages over 18 days
- Both teams: REST principles, versioning, error handling, pagination
- Expected: HIGH impact duplicate work gap

**Auth Migration** (`auth_migration_duplication`)
- Security and Platform teams migrating from sessions to JWT
- 12+ messages over 9 days
- Both teams: RS256, middleware, migration strategy
- Expected: MEDIUM-HIGH impact duplicate work gap

### Negative Examples (Should NOT Detect Gaps)

**Similar Topics, Different Scope** (`similar_topics_different_scope`)
- User authentication (social logins) vs service-to-service auth (mTLS)
- Same word "authentication" but completely different contexts
- Expected: NO gap detected

**Sequential Work** (`sequential_work`)
- Web team implements OAuth, then 60 days later mobile team does
- No temporal overlap
- Expected: NO gap detected

**Intentional Collaboration** (`intentional_collaboration`)
- Platform and Auth teams explicitly coordinating on OAuth
- Clear division of labor, cross-references, sync meetings
- Expected: NO gap detected

## Technical Details

### Detection Pipeline

The dashboard runs the same detection pipeline as the API:

1. **Message Retrieval** - Load mock Slack messages from selected scenario
2. **Embedding Generation** - Create semantic embeddings using sentence-transformers
3. **Clustering** - DBSCAN clustering with 0.85 similarity threshold
4. **Entity Extraction** - Identify teams, projects, people
5. **Temporal Overlap** - Check if teams worked simultaneously
6. **LLM Verification** - Claude API validates if it's true duplication
7. **Impact Scoring** - Multi-signal scoring (team size, time, criticality)
8. **Ranking** - Sort gaps by impact score

### Data Flow

```
Scenario Selection
    ↓
MockSlackClient.get_scenario_messages()
    ↓
DuplicateWorkDetector.detect()
    ↓
[Clustering → Entity Extraction → LLM Verification → Impact Scoring]
    ↓
List[CoordinationGap]
    ↓
Dashboard Display
```

### Technology Stack

- **Frontend**: Streamlit (Python-based web framework)
- **Detection**: Same modules as FastAPI backend
  - `src/detection/duplicate_work.py`
  - `src/detection/clustering.py`
  - `src/detection/impact_scoring.py`
  - `src/models/llm.py` (Claude API)
- **Mock Data**: `src/ingestion/slack/mock_client.py`

## Customization

### Adding New Scenarios

Edit `src/ingestion/slack/mock_client.py`:

```python
def _generate_your_scenario(self) -> List[MockMessage]:
    """Generate your custom scenario."""
    base_time = datetime.utcnow() - timedelta(days=X)

    messages = [
        MockMessage(
            content="Your message content",
            author="user@company.com",
            channel="#channel-name",
            timestamp=base_time,
            thread_id="thread_id",
            metadata={"team": "team-name"}
        ),
        # ... more messages
    ]

    return messages
```

Then add to the `scenarios` dict in `__init__`:

```python
self.scenarios = {
    # ... existing scenarios
    "your_scenario": self._generate_your_scenario,
}
```

### Styling

Custom CSS is in `streamlit_app.py`:

```python
st.markdown("""
<style>
    .main-header { ... }
    .gap-card { ... }
    .critical { border-left-color: #dc3545; }
    # ... modify as needed
</style>
""", unsafe_allow_html=True)
```

## Performance

- **Loading Time**: ~2-5 seconds for scenario with 20-30 messages
- **Detection Time**: ~3-8 seconds depending on:
  - Number of messages (more messages = more clustering time)
  - Number of clusters (more clusters = more LLM calls)
  - LLM API latency (Claude API response time)

### Optimization Tips

1. **Reduce LLM Calls**: Filter clusters before LLM verification
2. **Cache Embeddings**: Embeddings are computed once per run
3. **Limit Evidence**: Display only top 10 evidence items per gap
4. **Background Processing**: Consider using `@st.cache_data` for expensive operations

## Troubleshooting

### Dashboard won't start
```bash
# Make sure Streamlit is installed
uv pip install streamlit

# Check Python version (requires 3.11+)
python --version
```

### "No module named 'src'"
```bash
# Run from project root directory
cd /path/to/coordination_gap_detector
streamlit run streamlit_app.py
```

### Gaps not detecting
- Check scenario selection (some scenarios are negative examples)
- Lower minimum impact score threshold
- Review detection logs (shown in terminal)

### Slow performance
- Reduce number of messages in scenario
- Use scenarios with fewer clusters
- Check network latency to Claude API

## Screenshots

(Screenshots would go here after dashboard is deployed)

- Welcome screen
- Scenario selection
- Detection results summary
- Gap detail card
- Evidence timeline
- Export functionality

## Demo Video

See [DEMO.md](./DEMO.md) for a step-by-step walkthrough of the dashboard features.

## API Integration

This dashboard uses **mock data only**. To integrate with the FastAPI backend:

1. Start the API server: `uvicorn src.main:app --reload`
2. Use `httpx` to call `/api/v1/gaps/detect` endpoint
3. Display results in the dashboard

Example integration:

```python
import httpx
import streamlit as st

async def detect_via_api(timeframe_days: int):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/gaps/detect",
            json={
                "timeframe_days": timeframe_days,
                "gap_types": ["duplicate_work"],
                "min_impact_score": 0.0
            }
        )
        return response.json()

# Use in dashboard
gaps = asyncio.run(detect_via_api(30))
```

## Contributing

To improve the dashboard:

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/dashboard-improvement`
3. Make changes to `streamlit_app.py`
4. Test locally: `streamlit run streamlit_app.py`
5. Submit a pull request

## License

AGPL-3.0 - See [LICENSE](./LICENSE) for details.
