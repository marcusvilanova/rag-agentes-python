"""Montagem dos componentes a partir da configuração e o serviço de perguntas.

Este é o único módulo que conhece as implementações concretas (Gemini,
MongoDB/Milvus, Redis, Langfuse); os demais dependem apenas de interfaces.
"""

from __future__ import annotations

from typing import Any, cast

import redis
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
from pymongo import MongoClient

from rag_agentes.agentes.grafo import EstadoDoGrafo, construir_grafo, montar_resposta
from rag_agentes.armazenamento import ArmazenamentoVetorial
from rag_agentes.armazenamento.mongodb import ArmazenamentoMongoDB
from rag_agentes.cache import CacheDeRespostas
from rag_agentes.configuracao import Configuracao
from rag_agentes.corpus import carregar_fontes
from rag_agentes.modelos import RespostaDaPergunta
from rag_agentes.observabilidade import Rastreamento
from rag_agentes.recuperacao import Recuperador
from rag_agentes.vetorizacao import VetorizadorGemini

# Incrementar quando prompts ou o formato da resposta mudarem, para invalidar o cache.
VERSAO_DAS_RESPOSTAS = "1"
_TEMPO_LIMITE_REDIS_SEGUNDOS = 2


class ServicoDePerguntas:
    def __init__(
        self,
        grafo: CompiledStateGraph[Any, Any, Any, Any],
        rastreamento: Rastreamento,
        cache: CacheDeRespostas | None = None,
    ) -> None:
        self._grafo = grafo
        self._rastreamento = rastreamento
        self._cache = cache

    def perguntar(self, pergunta: str) -> RespostaDaPergunta:
        with self._rastreamento.observar("perguntar", entrada=pergunta) as observacao:
            resposta = self._responder(pergunta)
            observacao.registrar_saida(resposta.model_dump(include={"situacao", "resposta", "em_cache"}))
        return resposta

    def encerrar(self) -> None:
        self._rastreamento.descarregar()

    def _responder(self, pergunta: str) -> RespostaDaPergunta:
        if self._cache is not None:
            anterior = self._cache.obter(pergunta)
            if anterior is not None:
                # A pergunta guardada pode diferir na caixa ou na pontuação; devolve a recebida.
                return anterior.model_copy(update={"em_cache": True, "pergunta": pergunta})

        estado = cast(
            EstadoDoGrafo,
            self._grafo.invoke(
                {"pergunta": pergunta},
                config={"callbacks": self._rastreamento.callbacks(), "run_name": "grafo_de_agentes"},
            ),
        )
        resposta = montar_resposta(estado)
        if self._cache is not None and resposta.situacao == "respondida":
            self._cache.gravar(resposta)
        return resposta


def criar_rastreamento(configuracao: Configuracao) -> Rastreamento:
    segredo = configuracao.langfuse_secret_key
    return Rastreamento(
        public_key=configuracao.langfuse_public_key,
        secret_key=segredo.get_secret_value() if segredo else None,
        base_url=configuracao.langfuse_base_url,
    )


def criar_vetorizador(configuracao: Configuracao) -> VetorizadorGemini:
    return VetorizadorGemini(
        chave_api=configuracao.chave_google(),
        modelo=configuracao.modelo_embeddings,
        dimensoes=configuracao.dimensoes_embeddings,
    )


def criar_armazenamento(configuracao: Configuracao) -> ArmazenamentoVetorial:
    if configuracao.armazenamento == "milvus":
        # Import tardio: pymilvus é dependência opcional.
        from pymilvus import MilvusClient

        from rag_agentes.armazenamento.milvus import ArmazenamentoMilvus

        return ArmazenamentoMilvus(
            MilvusClient(uri=configuracao.milvus_uri.get_secret_value()), configuracao.milvus_colecao
        )
    return ArmazenamentoMongoDB(
        MongoClient(configuracao.mongodb_uri.get_secret_value()),
        banco=configuracao.mongodb_banco,
        colecao=configuracao.mongodb_colecao,
        indice=configuracao.mongodb_indice,
    )


def criar_recuperador(configuracao: Configuracao, rastreamento: Rastreamento) -> Recuperador:
    return Recuperador(
        vetorizador=criar_vetorizador(configuracao),
        armazenamento=criar_armazenamento(configuracao),
        k=configuracao.k_trechos,
        pontuacao_minima=configuracao.pontuacao_minima,
        rastreamento=rastreamento,
        modelo_embeddings=configuracao.modelo_embeddings,
    )


def criar_cache(configuracao: Configuracao) -> CacheDeRespostas:
    cliente = redis.Redis.from_url(
        configuracao.redis_url.get_secret_value(),
        decode_responses=True,
        socket_timeout=_TEMPO_LIMITE_REDIS_SEGUNDOS,
        socket_connect_timeout=_TEMPO_LIMITE_REDIS_SEGUNDOS,
    )
    assinatura = "|".join(
        str(parte)
        for parte in (
            VERSAO_DAS_RESPOSTAS,
            configuracao.modelo_chat,
            configuracao.modelo_embeddings,
            configuracao.dimensoes_embeddings,
            configuracao.k_trechos,
            configuracao.pontuacao_minima,
            configuracao.max_tentativas_redacao,
        )
    )
    return CacheDeRespostas(cliente, configuracao.cache_ttl_segundos, assinatura)


def criar_servico(configuracao: Configuracao, usar_cache: bool = True) -> ServicoDePerguntas:
    rastreamento = criar_rastreamento(configuracao)
    modelo = ChatGoogleGenerativeAI(model=configuracao.modelo_chat, google_api_key=configuracao.chave_google())
    titulos = [fonte.titulo for fonte in carregar_fontes(configuracao.pasta_corpus / "fontes.toml")]
    grafo = construir_grafo(
        modelo=modelo,
        recuperador=criar_recuperador(configuracao, rastreamento),
        documentos=titulos,
        max_tentativas=configuracao.max_tentativas_redacao,
    )
    return ServicoDePerguntas(grafo, rastreamento, criar_cache(configuracao) if usar_cache else None)
