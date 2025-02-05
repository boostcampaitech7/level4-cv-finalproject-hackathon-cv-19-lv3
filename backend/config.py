from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    service_api_key: str
    clova_host: str
    request_id: str

    class Config:
        env_file = '.env'

settings = Settings()