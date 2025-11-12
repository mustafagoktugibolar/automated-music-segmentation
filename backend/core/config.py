from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field

class DB_Settings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    
    # Pooling settings
    PG_MIN_CONN: int = 1
    PG_MAX_CONN: int = 10
    PG_RETRY_COUNT: int = 5
    PG_RETRY_WAIT: int = 3
    PG_STATEMENT_TIMEOUT_MS: int = 15000
    
    @computed_field
    @property
    def DB_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')
    
settings = DB_Settings()
