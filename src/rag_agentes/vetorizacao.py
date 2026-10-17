"""Geração de embeddings com o Gemini.

O `gemini-embedding-2` não aceita o parâmetro `task_type`; a documentação do
modelo pede que a tarefa seja indicada por um prefixo no próprio texto, um para
documentos e outro para consultas. Os dois prefixos abaixo seguem esse formato.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import batched
from typing import Protocol

from google import genai
from google.genai import types

from rag_agentes.modelos import Trecho

_TAMANHO_DO_LOTE = 50
_TENTATIVAS_HTTP = 5


class Vetorizador(Protocol):
    @property
    def dimensoes(self) -> int: ...

    def vetorizar_trechos(self, trechos: Sequence[Trecho]) -> list[list[float]]: ...

    def vetorizar_consulta(self, consulta: str) -> list[float]: ...


def texto_para_indexacao(trecho: Trecho) -> str:
    """Texto enviado ao modelo para um trecho: inclui lei e seção como título."""
    titulo = " - ".join(parte for parte in (trecho.titulo_documento, trecho.secao) if parte)
    return f"title: {titulo} | text: {trecho.texto}"


def texto_para_consulta(consulta: str) -> str:
    return f"task: search result | query: {consulta}"


class VetorizadorGemini:
    def __init__(self, chave_api: str, modelo: str, dimensoes: int) -> None:
        opcoes_http = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=_TENTATIVAS_HTTP))
        self._cliente = genai.Client(api_key=chave_api, http_options=opcoes_http)
        self._modelo = modelo
        self._dimensoes = dimensoes

    @property
    def dimensoes(self) -> int:
        return self._dimensoes

    def vetorizar_trechos(self, trechos: Sequence[Trecho]) -> list[list[float]]:
        vetores: list[list[float]] = []
        for lote in batched(trechos, _TAMANHO_DO_LOTE):
            vetores.extend(self._vetorizar([texto_para_indexacao(trecho) for trecho in lote]))
        return vetores

    def vetorizar_consulta(self, consulta: str) -> list[float]:
        return self._vetorizar([texto_para_consulta(consulta)])[0]

    def _vetorizar(self, textos: list[str]) -> list[list[float]]:
        # Cada texto vai num Content próprio: uma lista de strings em `contents`
        # faria o gemini-embedding-2 devolver um único vetor agregado.
        conteudos = [types.Content(parts=[types.Part.from_text(text=texto)]) for texto in textos]
        resposta = self._cliente.models.embed_content(
            model=self._modelo,
            contents=conteudos,
            config=types.EmbedContentConfig(output_dimensionality=self._dimensoes),
        )
        embeddings = resposta.embeddings or []
        if len(embeddings) != len(textos):
            raise RuntimeError(f"Esperados {len(textos)} vetores, recebidos {len(embeddings)}.")
        return [list(embedding.values or []) for embedding in embeddings]
