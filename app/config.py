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
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode={self.DB_SSL}"
        )

    # LLM Configuration — Gemini only
    LLM_API_KEY: str = ""

    # Application Configuration
    APP_NAME: str = "TapTap Analytics Chatbot"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Query Configuration
    MAX_QUERY_RESULTS: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # ignore extra keys in .env like LLM_PROVIDER


settings = Settings()