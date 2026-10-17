"""API HTTP. Documentação interativa em /docs quando o servidor está no ar."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field, StringConstraints

from rag_agentes.configuracao import obter_configuracao
from rag_agentes.modelos import RespostaDaPergunta
from rag_agentes.servico import ServicoDePerguntas, criar_servico


class PedidoDePergunta(BaseModel):
    pergunta: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=1000)] = Field(
        examples=["Em que hipóteses a LGPD permite tratar dados pessoais?"],
    )


def obter_servico(request: Request) -> ServicoDePerguntas:
    servico: ServicoDePerguntas = request.app.state.servico
    return servico


def criar_app(construir_servico: Callable[[], ServicoDePerguntas]) -> FastAPI:
    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
        app.state.servico = construir_servico()
        yield
        app.state.servico.encerrar()

    app = FastAPI(
        title="RAG com agentes sobre legislação brasileira",
        version="0.1.0",
        lifespan=ciclo_de_vida,
    )

    # Rotas síncronas: o FastAPI as executa num pool de threads, o que basta
    # aqui porque o trabalho é dominado por E/S em clientes síncronos.
    @app.post("/perguntar", response_model=RespostaDaPergunta)
    def perguntar(
        pedido: PedidoDePergunta,
        servico: Annotated[ServicoDePerguntas, Depends(obter_servico)],
    ) -> RespostaDaPergunta:
        return servico.perguntar(pedido.pergunta)

    @app.get("/saude")
    def saude() -> dict[str, str]:
        return {"situacao": "ok"}

    return app


def _servico_padrao() -> ServicoDePerguntas:
    return criar_servico(obter_configuracao())


app = criar_app(_servico_padrao)
