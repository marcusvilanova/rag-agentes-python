"""Avaliação com um conjunto versionado de perguntas e fontes esperadas.

Duas modalidades:
- recuperação (padrão): só embeddings e índice; mede se os artigos esperados
  aparecem entre os k primeiros resultados. Barata e determinística dado o índice.
- completa: executa o grafo inteiro e mede classificação, aprovação pelo
  verificador e se as citações incluem as fontes esperadas. Consome cota do LLM.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, model_validator

from rag_agentes.modelos import RespostaDaPergunta
from rag_agentes.recuperacao import Recuperador


class ItemDeAvaliacao(BaseModel):
    id: str
    pergunta: str
    categoria: Literal["consulta_legislacao", "fora_do_escopo"]
    fontes_esperadas: list[str]

    @model_validator(mode="after")
    def _fontes_coerentes_com_categoria(self) -> Self:
        if self.categoria == "consulta_legislacao" and not self.fontes_esperadas:
            raise ValueError(f"{self.id}: perguntas no escopo precisam de fontes esperadas.")
        if self.categoria == "fora_do_escopo" and self.fontes_esperadas:
            raise ValueError(f"{self.id}: perguntas fora do escopo não têm fontes esperadas.")
        return self


@dataclass(frozen=True)
class ResultadoDeRecuperacao:
    id: str
    revocacao: float
    posicao_do_primeiro_acerto: int | None


@dataclass(frozen=True)
class ResultadoCompleto:
    id: str
    classificacao_correta: bool
    situacao: str
    citou_fonte_esperada: bool | None


def carregar_conjunto(caminho: Path) -> list[ItemDeAvaliacao]:
    linhas = caminho.read_text(encoding="utf-8").splitlines()
    return [ItemDeAvaliacao.model_validate_json(linha) for linha in linhas if linha.strip()]


def corresponde(id_do_trecho: str, fonte_esperada: str) -> bool:
    """`lgpd:art-7` corresponde ao trecho inteiro e a qualquer parte dele (`lgpd:art-7:parte-2`)."""
    return id_do_trecho == fonte_esperada or id_do_trecho.startswith(f"{fonte_esperada}:")


def avaliar_ids(item: ItemDeAvaliacao, ids_recuperados: Sequence[str]) -> ResultadoDeRecuperacao:
    encontradas = [
        esperada for esperada in item.fontes_esperadas if any(corresponde(i, esperada) for i in ids_recuperados)
    ]
    posicao = next(
        (
            indice
            for indice, id_recuperado in enumerate(ids_recuperados, start=1)
            if any(corresponde(id_recuperado, esperada) for esperada in item.fontes_esperadas)
        ),
        None,
    )
    return ResultadoDeRecuperacao(
        id=item.id,
        revocacao=len(encontradas) / len(item.fontes_esperadas),
        posicao_do_primeiro_acerto=posicao,
    )


def resumir_recuperacao(resultados: Sequence[ResultadoDeRecuperacao]) -> dict[str, float]:
    total = len(resultados)
    if total == 0:
        return {"perguntas": 0, "acerto_em_k": 0.0, "revocacao_em_k": 0.0, "mrr": 0.0}
    return {
        "perguntas": total,
        "acerto_em_k": sum(r.posicao_do_primeiro_acerto is not None for r in resultados) / total,
        "revocacao_em_k": sum(r.revocacao for r in resultados) / total,
        "mrr": sum(1 / r.posicao_do_primeiro_acerto for r in resultados if r.posicao_do_primeiro_acerto) / total,
    }


def avaliar_recuperacao(
    itens: Sequence[ItemDeAvaliacao], recuperador: Recuperador, k: int
) -> list[ResultadoDeRecuperacao]:
    no_escopo = [item for item in itens if item.categoria == "consulta_legislacao"]
    return [avaliar_ids(item, [r.trecho.id for r in recuperador.recuperar(item.pergunta, k=k)]) for item in no_escopo]


def avaliar_completo(
    itens: Sequence[ItemDeAvaliacao], perguntar: Callable[[str], RespostaDaPergunta]
) -> list[ResultadoCompleto]:
    resultados = []
    for item in itens:
        resposta = perguntar(item.pergunta)
        foi_rejeitada = resposta.situacao == "fora_do_escopo"
        citou = None
        if resposta.situacao == "respondida":
            citou = any(corresponde(f.id, esperada) for f in resposta.fontes for esperada in item.fontes_esperadas)
        resultados.append(
            ResultadoCompleto(
                id=item.id,
                classificacao_correta=foi_rejeitada == (item.categoria == "fora_do_escopo"),
                situacao=resposta.situacao,
                citou_fonte_esperada=citou,
            )
        )
    return resultados


def resumir_completo(itens: Sequence[ItemDeAvaliacao], resultados: Sequence[ResultadoCompleto]) -> dict[str, float]:
    categorias = {item.id: item.categoria for item in itens}
    no_escopo = [r for r in resultados if categorias[r.id] == "consulta_legislacao"]
    respondidas = [r for r in no_escopo if r.situacao == "respondida"]
    return {
        "perguntas": len(resultados),
        "classificacao_correta": sum(r.classificacao_correta for r in resultados) / max(len(resultados), 1),
        "respondidas_no_escopo": len(respondidas) / max(len(no_escopo), 1),
        "citou_fonte_esperada": sum(bool(r.citou_fonte_esperada) for r in respondidas) / max(len(respondidas), 1),
    }


def gravar_relatorio(
    caminho: Path,
    resumo: dict[str, float],
    detalhes: Sequence[ResultadoDeRecuperacao | ResultadoCompleto],
) -> None:
    conteudo = {"resumo": resumo, "detalhes": [asdict(d) for d in detalhes]}
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2), encoding="utf-8")
