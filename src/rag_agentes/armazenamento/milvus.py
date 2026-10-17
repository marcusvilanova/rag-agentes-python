"""Índice vetorial no Milvus, atrás da mesma interface do MongoDB.

Dependência opcional: `uv sync --extra milvus`. `MILVUS_URI` aceita o endereço
de um servidor Milvus ou o caminho de um arquivo local (Milvus Lite, disponível
só em Linux e macOS), que é o modo usado nos testes de CI.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pymilvus import DataType, MilvusClient

from rag_agentes.modelos import Trecho, TrechoRecuperado

_TAMANHO_MAXIMO_ID = 256
_TAMANHO_MAXIMO_DOCUMENTO = 64


class ArmazenamentoMilvus:
    def __init__(self, cliente: MilvusClient, colecao: str) -> None:
        self._cliente = cliente
        self._colecao = colecao

    def preparar(self, dimensoes: int) -> None:
        if self._cliente.has_collection(self._colecao):
            self._cliente.load_collection(self._colecao)
            return
        esquema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        esquema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=_TAMANHO_MAXIMO_ID)
        esquema.add_field("documento", DataType.VARCHAR, max_length=_TAMANHO_MAXIMO_DOCUMENTO)
        esquema.add_field("vetor", DataType.FLOAT_VECTOR, dim=dimensoes)
        esquema.add_field("trecho", DataType.JSON)
        indices = MilvusClient.prepare_index_params()
        indices.add_index(field_name="vetor", index_type="AUTOINDEX", metric_type="COSINE")
        # Consistência forte: uma busca logo após a ingestão já enxerga as escritas.
        self._cliente.create_collection(self._colecao, schema=esquema, index_params=indices, consistency_level="Strong")

    def substituir_documento(
        self, documento: str, trechos: Sequence[Trecho], vetores: Sequence[Sequence[float]]
    ) -> None:
        registros = [
            {
                "id": trecho.id,
                "documento": trecho.documento,
                "vetor": list(vetor),
                "trecho": trecho.model_dump(exclude={"id"}),
            }
            for trecho, vetor in zip(trechos, vetores, strict=True)
        ]
        # json.dumps produz literais com aspas e escapes válidos na expressão do Milvus.
        filtro = f"documento == {json.dumps(documento)} and id not in {json.dumps([t.id for t in trechos])}"
        self._cliente.delete(self._colecao, filter=filtro)
        if registros:
            self._cliente.upsert(self._colecao, registros)

    def buscar(self, vetor: Sequence[float], k: int, documentos: Sequence[str] | None = None) -> list[TrechoRecuperado]:
        filtro = f"documento in {json.dumps(list(documentos))}" if documentos else ""
        resultados = self._cliente.search(
            self._colecao,
            data=[list(vetor)],
            limit=k,
            filter=filtro,
            anns_field="vetor",
            output_fields=["trecho"],
            search_params={"metric_type": "COSINE"},
        )
        return [_para_trecho_recuperado(acerto) for acerto in resultados[0]]

    def contar(self) -> int:
        resultado = self._cliente.query(self._colecao, filter="", output_fields=["count(*)"])
        return int(resultado[0]["count(*)"])


def _para_trecho_recuperado(acerto: dict[str, Any]) -> TrechoRecuperado:
    # O Milvus devolve o cosseno em [-1, 1]; a interface usa a escala [0, 1] do MongoDB.
    return TrechoRecuperado(
        trecho=Trecho(id=acerto["id"], **acerto["entity"]["trecho"]),
        pontuacao=(1 + float(acerto["distance"])) / 2,
    )
