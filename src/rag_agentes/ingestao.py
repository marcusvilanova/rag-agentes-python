"""Ingestão: lê o corpus, fragmenta, vetoriza e grava no índice vetorial."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag_agentes.armazenamento import ArmazenamentoVetorial
from rag_agentes.corpus import carregar_fontes
from rag_agentes.fragmentacao import fragmentar
from rag_agentes.modelos import Trecho
from rag_agentes.vetorizacao import Vetorizador


def carregar_trechos(pasta_corpus: Path) -> dict[str, list[Trecho]]:
    """Fragmenta todos os documentos listados em `fontes.toml`, sem vetorizar."""
    return {
        fonte.id: fragmentar(fonte.id, fonte.titulo, (pasta_corpus / f"{fonte.id}.txt").read_text(encoding="utf-8"))
        for fonte in carregar_fontes(pasta_corpus / "fontes.toml")
    }


def ingerir(
    pasta_corpus: Path,
    vetorizador: Vetorizador,
    armazenamento: ArmazenamentoVetorial,
    ao_concluir_documento: Callable[[str, int], None] = lambda _documento, _quantidade: None,
) -> dict[str, int]:
    """Indexa o corpus inteiro. É idempotente: reexecutar substitui os trechos."""
    armazenamento.preparar(vetorizador.dimensoes)
    totais: dict[str, int] = {}
    for documento, trechos in carregar_trechos(pasta_corpus).items():
        vetores = vetorizador.vetorizar_trechos(trechos)
        armazenamento.substituir_documento(documento, trechos, vetores)
        totais[documento] = len(trechos)
        ao_concluir_documento(documento, len(trechos))
    return totais
