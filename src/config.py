"""
Configuration management for the coordination gap detector.
"""
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    environment: str = Field(default="development", description="Environment (development, production)")
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    log_level: str = Field(default="INFO", description="Logging level")
    max_workers: int = Field(default=8, description="Maximum worker processes")

    # API Keys
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    slack_bot_token: Optional[str] = Field(default=None, description="Slack bot token")
    slack_app_token: Optional[str] = Field(default=None, description="Slack app token")
    github_token: Optional[str] = Field(default=None, description="GitHub API token")
    jira_api_token: Optional[str] = Field(default=None, description="Jira API token")

    # Search Infrastructure
    elasticsearch_url: str = Field(
        default="http://localhost:9200", description="Elasticsearch URL"
    )
    elasticsearch_api_key: Optional[str] = Field(
        default=None, description="Elasticsearch API key"
    )

    # Database
    postgres_url: str = Field(
        default="postgresql://user:pass@localhost:5432/coordination",
        description="PostgreSQL connection URL",
    )
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    neo4j_uri: Optional[str] = Field(default=None, description="Neo4j URI")
    neo4j_user: Optional[str] = Field(default=None, description="Neo4j user")
    neo4j_password: Optional[str] = Field(default=None, description="Neo4j password")

    # Vector Store
    chroma_persist_dir: str = Field(
        default="./data/chroma", description="ChromaDB persistence directory"
    )

    # Event Streaming
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092", description="Kafka bootstrap servers"
    )
    kafka_topic_slack: str = Field(
        default="slack-events", description="Kafka topic for Slack events"
    )
    kafka_topic_github: str = Field(
        default="github-events", description="Kafka topic for GitHub events"
    )

    # Kubernetes (Production)
    k8s_namespace: str = Field(
        default="coordination-prod", description="Kubernetes namespace"
    )
    k8s_context: Optional[str] = Field(
        default=None, description="Kubernetes context"
    )

    # Observability
    prometheus_endpoint: Optional[str] = Field(
        default=None, description="Prometheus endpoint"
    )
    grafana_api_key: Optional[str] = Field(default=None, description="Grafana API key")
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN")

    # Feature Flags
    enable_realtime_detection: bool = Field(
        default=True, description="Enable real-time gap detection"
    )
    enable_duplicate_work_detection: bool = Field(
        default=True, description="Enable duplicate work detection"
    )
    enable_missing_context_detection: bool = Field(
        default=True, description="Enable missing context detection"
    )
    enable_stale_docs_detection: bool = Field(
        default=True, description="Enable stale documentation detection"
    )
    gap_detection_model_version: str = Field(
        default="v3.1", description="Gap detection model version"
    )
    ranking_model_version: str = Field(
        default="v2.7", description="Ranking model version"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
