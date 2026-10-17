"""Testes contra MongoDB Atlas Local e Redis reais (docker compose ou serviços do CI).

Executar com: `uv run pytest -m integracao`. O vetorizador é o de teste, então
nenhuma chave de API é necessária.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
import redis
from apoio import VetorizadorDeTeste, aguardar, verificar_contrato_de_armazenamento
from pymongo import MongoClient

from rag_agentes.armazenamento.mongodb import ArmazenamentoMongoDB
from rag_agentes.cache import CacheDeRespostas
from rag_agentes.modelos import RespostaDaPergunta

pytestmark = pytest.mark.integracao

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/?directConnection=true")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/15")
_TEMPO_LIMITE_DE_INDEXACAO_SEGUNDOS = 90


@pytest.fixture
def armazenamento_mongodb() -> Iterator[ArmazenamentoMongoDB]:
    cliente: MongoClient[dict[str, object]] = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    banco = f"teste_{uuid4().hex[:8]}"
    try:
        yield ArmazenamentoMongoDB(cliente, banco=banco, colecao="trechos", indice="indice_vetorial")
    finally:
        cliente.drop_database(banco)
        cliente.close()


def test_mongodb_atlas_vector_search_cumpre_o_contrato(armazenamento_mongodb: ArmazenamentoMongoDB) -> None:
    consulta = VetorizadorDeTeste().vetorizar_consulta("dados")

    def aguardar_indexacao(quantidade: int) -> None:
        # O índice do Atlas é atualizado de forma assíncrona após cada escrita.
        aguardar(
            lambda: len(armazenamento_mongodb.buscar(consulta, k=10)) == quantidade,
            tempo_limite=_TEMPO_LIMITE_DE_INDEXACAO_SEGUNDOS,
        )

    verificar_contrato_de_armazenamento(armazenamento_mongodb, aguardar_indexacao)


def test_cache_no_redis_respeita_o_ttl() -> None:
    cliente = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    cache = CacheDeRespostas(cliente, ttl_segundos=30, assinatura=f"teste-{uuid4().hex}")
    resposta = RespostaDaPergunta(pergunta="p?", situacao="respondida", resposta="r [1]", fontes=[], tentativas=1)

    try:
        cache.gravar(resposta)
        assert cache.obter("P") == resposta
        assert 0 < cliente.ttl(cache.chave("p")) <= 30
    finally:
        cache.limpar()
        cliente.close()
