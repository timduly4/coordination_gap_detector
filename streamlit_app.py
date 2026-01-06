"""
Streamlit Dashboard for Coordination Gap Detector

Interactive web interface for detecting and visualizing coordination gaps across teams.
"""
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.slack.mock_client import MockSlackClient
from src.detection.duplicate_work import DuplicateWorkDetector
from src.models.schemas import CoordinationGap


# Page configuration
st.set_page_config(
    page_title="Coordination Gap Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .gap-card {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
        border-radius: 0.3rem;
    }
    .critical { border-left-color: #dc3545; }
    .high { border-left-color: #fd7e14; }
    .medium { border-left-color: #ffc107; }
    .low { border-left-color: #28a745; }
    .evidence-item {
        background-color: #fff;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #6c757d;
    }
    .stat-box {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def get_impact_color(impact_tier: str) -> str:
    """Get color based on impact tier."""
    colors = {
        "CRITICAL": "#dc3545",
        "HIGH": "#fd7e14",
        "MEDIUM": "#ffc107",
        "LOW": "#28a745"
    }
    return colors.get(impact_tier, "#6c757d")


def get_impact_emoji(impact_tier: str) -> str:
    """Get emoji based on impact tier."""
    emojis = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }
    return emojis.get(impact_tier, "⚪")


def display_gap_card(gap: CoordinationGap, index: int) -> None:
    """Display a single gap as a styled card."""
    impact_class = gap.impact_tier.lower()

    with st.expander(
        f"{get_impact_emoji(gap.impact_tier)} **Gap {index + 1}: {gap.title}**",
        expanded=(index == 0)  # Expand first gap by default
    ):
        # Header metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Impact Score", f"{gap.impact_score:.2f}")
        with col2:
            st.metric("Confidence", f"{gap.confidence:.0%}")
        with col3:
            st.metric("Teams Involved", len(gap.teams_involved))
        with col4:
            if gap.estimated_cost:
                st.metric("Est. Cost", f"${gap.estimated_cost.dollar_value:,.0f}")

        # Gap details
        st.markdown("---")
        st.markdown(f"**🎯 Type:** `{gap.type}`")
        st.markdown(f"**📋 Topic:** {gap.topic}")
        st.markdown(f"**👥 Teams:** {', '.join(gap.teams_involved)}")

        if gap.timespan_days:
            st.markdown(f"**⏱️ Duration:** {gap.timespan_days} days")

        # Insight
        st.markdown("### 💡 AI Insight")
        st.info(gap.insight)

        # Recommendation
        st.markdown("### 🎯 Recommendation")
        st.success(gap.recommendation)

        # Cost breakdown
        if gap.estimated_cost:
            st.markdown("### 💰 Cost Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Engineering Hours", f"{gap.estimated_cost.engineering_hours:.0f} hrs")
            with col2:
                st.metric("Dollar Value", f"${gap.estimated_cost.dollar_value:,.0f}")
            st.caption(gap.estimated_cost.explanation)

        # LLM Verification
        if gap.verification:
            st.markdown("### 🤖 LLM Verification")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Duplicate Work?", "Yes" if gap.verification.is_duplicate else "No")
            with col2:
                st.metric("Overlap Ratio", f"{gap.verification.overlap_ratio:.0%}")

            st.markdown(f"**Reasoning:** {gap.verification.reasoning}")

        # Temporal overlap
        if gap.temporal_overlap:
            st.markdown("### 📅 Temporal Overlap")
            st.markdown(
                f"**Overlap Period:** {gap.temporal_overlap.start.strftime('%Y-%m-%d')} "
                f"to {gap.temporal_overlap.end.strftime('%Y-%m-%d')} "
                f"({gap.temporal_overlap.overlap_days} days)"
            )

        # Evidence
        if gap.evidence:
            st.markdown("### 📊 Evidence")
            st.caption(f"Showing {len(gap.evidence)} pieces of evidence")

            for i, evidence in enumerate(gap.evidence[:10]):  # Limit to 10 for readability
                with st.container():
                    st.markdown(f"""
                    <div class="evidence-item">
                        <strong>#{i+1} - {evidence.source}</strong> |
                        {evidence.channel or 'Unknown channel'} |
                        {evidence.author or 'Unknown author'} |
                        {evidence.timestamp.strftime('%Y-%m-%d %H:%M')}
                        <br>
                        <em>"{evidence.content[:200]}{'...' if len(evidence.content) > 200 else ''}"</em>
                        <br>
                        <small>Relevance: {evidence.relevance_score:.2f} | Team: {evidence.team or 'Unknown'}</small>
                    </div>
                    """, unsafe_allow_html=True)

            if len(gap.evidence) > 10:
                st.caption(f"... and {len(gap.evidence) - 10} more pieces of evidence")


async def detect_gaps_async(messages, scenario_name: str) -> List[CoordinationGap]:
    """Run gap detection asynchronously."""
    detector = DuplicateWorkDetector()

    # Convert mock messages to the format expected by detector
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "content": msg.content,
            "author": msg.author,
            "channel": msg.channel,
            "timestamp": msg.timestamp,
            "external_id": msg.external_id,
            "thread_id": msg.thread_id,
            "metadata": msg.metadata
        })

    # Detect gaps (this will generate embeddings internally)
    gaps = await detector.detect(messages=formatted_messages)

    return gaps


def main():
    """Main dashboard application."""

    # Header
    st.markdown('<div class="main-header">🔍 Coordination Gap Detector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-powered system that detects when teams are duplicating work, '
        'missing context, or working at cross-purposes</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Initialize mock client
        mock_client = MockSlackClient()
        scenario_descriptions = mock_client.get_scenario_descriptions()

        # Scenario selection
        st.subheader("📁 Select Scenario")
        scenario = st.selectbox(
            "Choose a mock data scenario:",
            options=list(scenario_descriptions.keys()),
            format_func=lambda x: scenario_descriptions[x],
            help="Select a pre-built scenario to analyze"
        )

        st.markdown("---")

        # Detection settings
        st.subheader("🎛️ Detection Settings")
        min_impact_score = st.slider(
            "Minimum Impact Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Filter gaps by minimum impact score"
        )

        include_evidence = st.checkbox(
            "Show Evidence",
            value=True,
            help="Include evidence details in results"
        )

        st.markdown("---")

        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Coordination Gap Detector** identifies organizational inefficiencies:

            - 🔄 **Duplicate Work** - Teams solving the same problem independently
            - ❓ **Missing Context** - Decisions made without key stakeholders
            - 📄 **Stale Docs** - Documentation contradicting current code
            - 🏝️ **Knowledge Silos** - Critical knowledge trapped in teams

            **Tech Stack:**
            - FastAPI, PostgreSQL, ChromaDB
            - Claude API for LLM reasoning
            - DBSCAN clustering
            - Hybrid search (BM25 + semantic)
            """)

        st.markdown("---")

        # Run detection button
        detect_button = st.button(
            "🚀 Detect Gaps",
            type="primary",
            use_container_width=True
        )

    # Main content area
    if detect_button:
        with st.spinner(f"🔍 Analyzing {scenario_descriptions[scenario]}..."):
            # Get messages from selected scenario
            messages = mock_client.get_scenario_messages(scenario)

            st.info(f"📨 Loaded **{len(messages)}** messages from scenario: **{scenario}**")

            # Show message preview
            with st.expander("📋 Message Preview", expanded=False):
                st.caption("First 5 messages from the scenario:")
                for i, msg in enumerate(messages[:5]):
                    st.markdown(f"""
                    **Message {i+1}** | {msg.channel} | {msg.author} | {msg.timestamp.strftime('%Y-%m-%d %H:%M')}
                    > {msg.content[:150]}{'...' if len(msg.content) > 150 else ''}
                    """)

            # Run detection
            try:
                # Run async detection
                gaps = asyncio.run(detect_gaps_async(messages, scenario))

                # Filter by impact score
                filtered_gaps = [g for g in gaps if g.impact_score >= min_impact_score]

                # Sort by impact score
                filtered_gaps.sort(key=lambda x: x.impact_score, reverse=True)

                # Display results
                st.markdown("---")
                st.markdown("## 📊 Detection Results")

                if filtered_gaps:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(
                            f'<div class="stat-box">'
                            f'<h2>{len(filtered_gaps)}</h2>'
                            f'<p>Total Gaps</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with col2:
                        critical_count = sum(1 for g in filtered_gaps if g.impact_tier == "CRITICAL")
                        st.markdown(
                            f'<div class="stat-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">'
                            f'<h2>{critical_count}</h2>'
                            f'<p>Critical</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with col3:
                        high_count = sum(1 for g in filtered_gaps if g.impact_tier == "HIGH")
                        st.markdown(
                            f'<div class="stat-box" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">'
                            f'<h2>{high_count}</h2>'
                            f'<p>High Priority</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with col4:
                        total_cost = sum(
                            g.estimated_cost.dollar_value for g in filtered_gaps
                            if g.estimated_cost
                        )
                        st.markdown(
                            f'<div class="stat-box" style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);">'
                            f'<h2>${total_cost:,.0f}</h2>'
                            f'<p>Est. Total Cost</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("---")

                    # Impact distribution
                    st.subheader("📈 Impact Distribution")
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Create distribution chart data
                        impact_counts = {
                            "CRITICAL": sum(1 for g in filtered_gaps if g.impact_tier == "CRITICAL"),
                            "HIGH": sum(1 for g in filtered_gaps if g.impact_tier == "HIGH"),
                            "MEDIUM": sum(1 for g in filtered_gaps if g.impact_tier == "MEDIUM"),
                            "LOW": sum(1 for g in filtered_gaps if g.impact_tier == "LOW"),
                        }
                        st.bar_chart(impact_counts)

                    with col2:
                        st.metric("Avg Impact Score", f"{sum(g.impact_score for g in filtered_gaps) / len(filtered_gaps):.2f}")
                        st.metric("Avg Confidence", f"{sum(g.confidence for g in filtered_gaps) / len(filtered_gaps):.0%}")
                        st.metric("Total Teams", len(set(team for g in filtered_gaps for team in g.teams_involved)))

                    st.markdown("---")

                    # Display gaps
                    st.subheader(f"🔍 Detected Gaps ({len(filtered_gaps)})")

                    # Add filter tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "🔴 All Gaps",
                        "🔴 Critical",
                        "🟠 High",
                        "🟡 Medium",
                        "🟢 Low"
                    ])

                    with tab1:
                        for i, gap in enumerate(filtered_gaps):
                            display_gap_card(gap, i)

                    with tab2:
                        critical_gaps = [g for g in filtered_gaps if g.impact_tier == "CRITICAL"]
                        if critical_gaps:
                            for i, gap in enumerate(critical_gaps):
                                display_gap_card(gap, i)
                        else:
                            st.info("No critical gaps detected")

                    with tab3:
                        high_gaps = [g for g in filtered_gaps if g.impact_tier == "HIGH"]
                        if high_gaps:
                            for i, gap in enumerate(high_gaps):
                                display_gap_card(gap, i)
                        else:
                            st.info("No high priority gaps detected")

                    with tab4:
                        medium_gaps = [g for g in filtered_gaps if g.impact_tier == "MEDIUM"]
                        if medium_gaps:
                            for i, gap in enumerate(medium_gaps):
                                display_gap_card(gap, i)
                        else:
                            st.info("No medium priority gaps detected")

                    with tab5:
                        low_gaps = [g for g in filtered_gaps if g.impact_tier == "LOW"]
                        if low_gaps:
                            for i, gap in enumerate(low_gaps):
                                display_gap_card(gap, i)
                        else:
                            st.info("No low priority gaps detected")

                    # Export option
                    st.markdown("---")
                    st.subheader("💾 Export Results")

                    # Convert gaps to JSON
                    export_data = {
                        "scenario": scenario,
                        "detected_at": datetime.utcnow().isoformat(),
                        "total_gaps": len(filtered_gaps),
                        "gaps": [g.dict() for g in filtered_gaps]
                    }

                    st.download_button(
                        label="⬇️ Download Results (JSON)",
                        data=json.dumps(export_data, indent=2, default=str),
                        file_name=f"coordination_gaps_{scenario}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                else:
                    st.warning(f"No gaps detected with impact score >= {min_impact_score}")
                    st.info("Try lowering the minimum impact score threshold in the sidebar.")

            except Exception as e:
                st.error(f"Error during gap detection: {e}")
                st.exception(e)

    else:
        # Welcome screen
        st.markdown("---")
        st.markdown("## 👋 Welcome!")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🎯 How It Works

            1. **Select a Scenario** from the sidebar
            2. **Configure Settings** (impact threshold, evidence display)
            3. **Click "Detect Gaps"** to run the analysis
            4. **Review Results** with AI-powered insights
            5. **Export Data** for further analysis
            """)

        with col2:
            st.markdown("""
            ### 📊 What You'll See

            - **Impact Scores** - Severity of coordination gaps
            - **AI Insights** - Claude-powered analysis
            - **Cost Estimates** - Dollar value of wasted effort
            - **Evidence** - Supporting messages from teams
            - **Recommendations** - Actionable next steps
            """)

        st.markdown("---")

        # Demo scenario preview
        st.markdown("### 🎬 Available Scenarios")

        mock_client = MockSlackClient()
        scenarios = mock_client.get_scenario_descriptions()

        for scenario_key, description in scenarios.items():
            with st.expander(f"📁 {description}"):
                messages = mock_client.get_scenario_messages(scenario_key)
                st.caption(f"**{len(messages)} messages** | Various channels and teams")

                # Show first message as preview
                if messages:
                    first_msg = messages[0]
                    st.markdown(f"""
                    **Preview:**
                    > {first_msg.content[:200]}{'...' if len(first_msg.content) > 200 else ''}

                    *— {first_msg.author} in {first_msg.channel}*
                    """)


if __name__ == "__main__":
    main()
