from __future__ import annotations

from collections.abc import Sequence

from rag_agentes.armazenamento import ArmazenamentoVetorial
from rag_agentes.modelos import TrechoRecuperado
from rag_agentes.observabilidade import Rastreamento
from rag_agentes.vetorizacao import Vetorizador


class Recuperador:
    """Vetoriza a consulta, busca no índice e descarta resultados abaixo do limiar."""

    def __init__(
        self,
        vetorizador: Vetorizador,
        armazenamento: ArmazenamentoVetorial,
        *,
        k: int,
        pontuacao_minima: float = 0.0,
        rastreamento: Rastreamento | None = None,
        modelo_embeddings: str | None = None,
    ) -> None:
        self._vetorizador = vetorizador
        self._armazenamento = armazenamento
        self._k = k
        self._pontuacao_minima = pontuacao_minima
        self._rastreamento = rastreamento or Rastreamento()
        self._modelo_embeddings = modelo_embeddings

    def recuperar(
        self, consulta: str, k: int | None = None, documentos: Sequence[str] | None = None
    ) -> list[TrechoRecuperado]:
        with self._rastreamento.observar("recuperar_trechos", tipo="retriever", entrada=consulta) as observacao:
            with self._rastreamento.observar(
                "vetorizar_consulta", tipo="embedding", entrada=consulta, modelo=self._modelo_embeddings
            ):
                vetor = self._vetorizador.vetorizar_consulta(consulta)
            encontrados = self._armazenamento.buscar(vetor, k or self._k, documentos)
            aceitos = [item for item in encontrados if item.pontuacao >= self._pontuacao_minima]
            observacao.registrar_saida([{"id": item.trecho.id, "pontuacao": item.pontuacao} for item in aceitos])
        return aceitos
