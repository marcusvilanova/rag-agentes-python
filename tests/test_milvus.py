"""O adaptador do Milvus cumpre o mesmo contrato do MongoDB, usando Milvus Lite.

Milvus Lite só existe para Linux e macOS; no Windows este módulo é ignorado.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from apoio import verificar_contrato_de_armazenamento

pytestmark = pytest.mark.milvus

pymilvus = pytest.importorskip("pymilvus")
pytest.importorskip("milvus_lite")

from rag_agentes.armazenamento.milvus import ArmazenamentoMilvus  # noqa: E402


def test_milvus_cumpre_o_contrato(tmp_path: Path) -> None:
    cliente = pymilvus.MilvusClient(uri=str(tmp_path / "milvus.db"))
    try:
        verificar_contrato_de_armazenamento(ArmazenamentoMilvus(cliente, "trechos"))
    finally:
        cliente.close()
