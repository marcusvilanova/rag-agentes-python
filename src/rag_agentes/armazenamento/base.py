from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_agentes.modelos import Trecho, TrechoRecuperado


class ArmazenamentoVetorial(Protocol):
    """Contrato que o restante do sistema conhece; MongoDB e Milvus o implementam.

    A pontuação devolvida por `buscar` é a similaridade de cosseno normalizada
    para [0, 1], isto é, (1 + cosseno) / 2, que é a escala nativa do MongoDB.
    """

    def preparar(self, dimensoes: int) -> None:
        """Cria coleção e índice vetorial, se ainda não existirem."""
        ...

    def substituir_documento(
        self, documento: str, trechos: Sequence[Trecho], vetores: Sequence[Sequence[float]]
    ) -> None:
        """Grava os trechos de um documento e apaga os que não existem mais nele."""
        ...

    def buscar(self, vetor: Sequence[float], k: int, documentos: Sequence[str] | None = None) -> list[TrechoRecuperado]:
        """Devolve os `k` trechos mais próximos, opcionalmente só dos documentos indicados."""
        ...

    def contar(self) -> int: ...
