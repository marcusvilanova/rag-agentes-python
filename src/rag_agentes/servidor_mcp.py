"""Servidor MCP que expõe a busca vetorial como ferramenta.

Um cliente MCP (Claude Desktop, Claude Code, IDEs) pode chamar
`buscar_trechos` e montar a própria resposta a partir dos trechos, usando o
mesmo índice da API. Transporte padrão: stdio.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field, StringConstraints

from rag_agentes.recuperacao import Recuperador

_K_MAXIMO = 20
_CONSULTA_MAXIMA = 1000


class TrechoEncontrado(BaseModel):
    id: str
    referencia: str
    secao: str
    texto: str
    pontuacao: float


def criar_servidor_mcp(recuperador: Recuperador, documentos: Sequence[str]) -> MCPServer:
    """`documentos` são os identificadores aceitos no filtro (ex.: "lgpd")."""
    servidor = MCPServer(
        name="rag-legislacao",
        instructions=(
            "Busca semântica em leis brasileiras: "
            + ", ".join(documentos)
            + ". Cite a referência de cada trecho usado na resposta."
        ),
    )
    documentos_validos = frozenset(documentos)

    @servidor.tool()
    def buscar_trechos(
        consulta: Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=3, max_length=_CONSULTA_MAXIMA),
            Field(description="Pergunta ou termos a buscar."),
        ],
        k: Annotated[int, Field(ge=1, le=_K_MAXIMO, description="Quantidade de trechos.")] = 5,
        documentos: Annotated[list[str] | None, Field(description="Restringe a busca a estes documentos.")] = None,
    ) -> list[TrechoEncontrado]:
        """Devolve os artigos de lei mais próximos da consulta, com referência e pontuação."""
        desconhecidos = set(documentos or []) - documentos_validos
        if desconhecidos:
            raise ValueError(
                f"Documentos desconhecidos: {sorted(desconhecidos)}. Válidos: {sorted(documentos_validos)}."
            )
        return [
            TrechoEncontrado(
                id=item.trecho.id,
                referencia=item.trecho.referencia,
                secao=item.trecho.secao,
                texto=item.trecho.texto,
                pontuacao=item.pontuacao,
            )
            for item in recuperador.recuperar(consulta, k=k, documentos=documentos)
        ]

    return servidor
