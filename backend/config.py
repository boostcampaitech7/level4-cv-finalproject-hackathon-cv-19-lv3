import logging
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    service_api_key: str
    clova_host: str
    request_id: str

    class Config:
        env_file = '.env'

settings = Settings()

logging.basicConfig(
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)