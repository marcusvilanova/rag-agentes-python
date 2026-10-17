"""Índice vetorial no MongoDB com Atlas Vector Search.

O mesmo código atende o Atlas na nuvem e a imagem `mongodb/mongodb-atlas-local`
do docker-compose, que roda o mesmo motor de busca (mongot) localmente.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from pymongo import MongoClient, ReplaceOne
from pymongo.operations import SearchIndexModel

from rag_agentes.modelos import Trecho, TrechoRecuperado

# A documentação do Atlas recomenda examinar ao menos 20 vezes mais candidatos
# que o limite pedido, para manter a revocação da busca aproximada (HNSW).
_FATOR_DE_CANDIDATOS = 20
_MINIMO_DE_CANDIDATOS = 100
_TEMPO_LIMITE_DO_INDICE_SEGUNDOS = 180


class ArmazenamentoMongoDB:
    def __init__(self, cliente: MongoClient[dict[str, Any]], banco: str, colecao: str, indice: str) -> None:
        self._colecao = cliente[banco][colecao]
        self._indice = indice

    def preparar(self, dimensoes: int) -> None:
        banco = self._colecao.database
        if self._colecao.name not in banco.list_collection_names():
            banco.create_collection(self._colecao.name)

        definicao = definicao_do_indice(dimensoes)
        existentes = list(self._colecao.list_search_indexes(self._indice))
        if not existentes:
            modelo = SearchIndexModel(definition=definicao, name=self._indice, type="vectorSearch")
            self._colecao.create_search_index(modelo)
        elif existentes[0].get("latestDefinition") != definicao:
            self._colecao.update_search_index(self._indice, definicao)
        self._aguardar_indice_pronto()

    def substituir_documento(
        self, documento: str, trechos: Sequence[Trecho], vetores: Sequence[Sequence[float]]
    ) -> None:
        operacoes = [
            ReplaceOne(
                {"_id": trecho.id},
                {**trecho.model_dump(exclude={"id"}), "embedding": list(vetor)},
                upsert=True,
            )
            for trecho, vetor in zip(trechos, vetores, strict=True)
        ]
        if operacoes:
            self._colecao.bulk_write(operacoes, ordered=False)
        self._colecao.delete_many({"documento": documento, "_id": {"$nin": [t.id for t in trechos]}})

    def buscar(self, vetor: Sequence[float], k: int, documentos: Sequence[str] | None = None) -> list[TrechoRecuperado]:
        estagio: dict[str, Any] = {
            "index": self._indice,
            "path": "embedding",
            "queryVector": list(vetor),
            "numCandidates": max(k * _FATOR_DE_CANDIDATOS, _MINIMO_DE_CANDIDATOS),
            "limit": k,
        }
        if documentos:
            estagio["filter"] = {"documento": {"$in": list(documentos)}}
        pipeline = [
            {"$vectorSearch": estagio},
            {"$project": {"embedding": 0, "pontuacao": {"$meta": "vectorSearchScore"}}},
        ]
        return [_para_trecho_recuperado(registro) for registro in self._colecao.aggregate(pipeline)]

    def contar(self) -> int:
        return self._colecao.count_documents({})

    def _aguardar_indice_pronto(self) -> None:
        # A criação do índice é assíncrona no Atlas; a busca falha até ele ficar pronto.
        prazo = time.monotonic() + _TEMPO_LIMITE_DO_INDICE_SEGUNDOS
        while time.monotonic() < prazo:
            indices = list(self._colecao.list_search_indexes(self._indice))
            if indices and indices[0].get("queryable") and indices[0].get("status") == "READY":
                return
            time.sleep(1)
        raise TimeoutError(f"O índice {self._indice!r} não ficou pronto em {_TEMPO_LIMITE_DO_INDICE_SEGUNDOS}s.")


def definicao_do_indice(dimensoes: int) -> dict[str, Any]:
    return {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": dimensoes, "similarity": "cosine"},
            {"type": "filter", "path": "documento"},
        ]
    }


def _para_trecho_recuperado(registro: Mapping[str, Any]) -> TrechoRecuperado:
    campos = {chave: valor for chave, valor in registro.items() if chave not in {"_id", "pontuacao"}}
    return TrechoRecuperado(
        trecho=Trecho(id=registro["_id"], **campos),
        pontuacao=float(registro["pontuacao"]),
    )
