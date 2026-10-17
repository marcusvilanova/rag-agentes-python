from __future__ import annotations

from collections.abc import Sequence

from rag_agentes.modelos import TrechoRecuperado


def formatar_trechos(trechos: Sequence[TrechoRecuperado]) -> str:
    """Numera os trechos a partir de 1; é essa numeração que as citações [n] usam."""
    blocos = []
    for numero, item in enumerate(trechos, start=1):
        trecho = item.trecho
        cabecalho = f"[{numero}] {trecho.referencia}"
        if trecho.secao:
            cabecalho += f" ({trecho.secao})"
        blocos.append(f"{cabecalho}\n{trecho.texto}")
    return "\n\n".join(blocos)
