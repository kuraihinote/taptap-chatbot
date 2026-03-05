from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database Configuration
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "taptap_db"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    DB_SSL: str = "require"

    @property
    def DATABASE_URL(self) -> str:
        """asyncpg-compatible URL used by the connection pool."""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def SQLALCHEMY_DATABASE_URL(self) -> str:
        """SQLAlchemy async URL used for schema introspection."""
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o-mini"
    AZURE_OPENAI_API_VERSION: str = "2025-01-01-preview"

    # Application Configuration
    APP_NAME: str = "TapTap Analytics Chatbot"
    VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Query Configuration
    MAX_QUERY_RESULTS: int = 100
    DEFAULT_LIMIT: int = 50

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()