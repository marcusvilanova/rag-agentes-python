"""Configuração lida de variáveis de ambiente ou do arquivo .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuracao(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gemini: um único provedor para chat e embeddings, com camada gratuita.
    google_api_key: SecretStr | None = None
    modelo_chat: str = "gemini-3.8-flash"
    modelo_embeddings: str = "gemini-embedding-2"
    dimensoes_embeddings: int = Field(default=768, ge=128, le=3072)

    armazenamento: Literal["mongodb", "milvus"] = "mongodb"
    # URIs como SecretStr: no Atlas elas carregam usuário e senha.
    mongodb_uri: SecretStr = SecretStr("mongodb://localhost:27017/?directConnection=true")
    mongodb_banco: str = "rag_agentes"
    mongodb_colecao: str = "trechos"
    mongodb_indice: str = "indice_vetorial"
    milvus_uri: SecretStr = SecretStr("http://localhost:19530")
    milvus_colecao: str = "trechos"

    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    cache_ttl_segundos: int = Field(default=24 * 60 * 60, ge=0)

    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    k_trechos: int = Field(default=5, ge=1, le=20)
    pontuacao_minima: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tentativas_redacao: int = Field(default=2, ge=1, le=5)
    pasta_corpus: Path = Path("dados/corpus")

    def chave_google(self) -> str:
        # `GOOGLE_API_KEY=` vazio, como no .env.example, conta como ausente.
        chave = self.google_api_key.get_secret_value() if self.google_api_key else ""
        if not chave:
            raise RuntimeError("Defina GOOGLE_API_KEY no ambiente ou no arquivo .env.")
        return chave


@lru_cache
def obter_configuracao() -> Configuracao:
    return Configuracao()
