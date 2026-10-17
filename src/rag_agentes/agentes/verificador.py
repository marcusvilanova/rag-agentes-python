"""Agente verificador: rejeita respostas que não estejam ancoradas nos trechos recuperados.

A verificação tem duas etapas. A primeira é determinística e barata: exige ao
menos uma citação e que toda citação [n] aponte para um trecho que de fato foi
recuperado. Só se ela passar é feita a segunda, em que um LLM confere se cada
afirmação é sustentada pelo trecho citado.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from rag_agentes.agentes.contexto import formatar_trechos
from rag_agentes.modelos import TrechoRecuperado

_CITACAO = re.compile(r"\[(\d+)\]")

_INSTRUCOES = """Você audita respostas escritas a partir de trechos de lei. Verifique se cada afirmação da resposta é sustentada pelo trecho que ela cita entre colchetes.

Uma afirmação é sustentada quando o trecho citado diz o mesmo, de forma explícita ou por paráfrase fiel. Não é sustentada quando acrescenta informação ausente do trecho, generaliza além do texto, altera sujeito, prazo, condição ou exceção, ou cita um trecho que trata de outro assunto.

Seja rigoroso: na dúvida, considere a afirmação sem sustentação."""


class Julgamento(BaseModel):
    afirmacoes_sem_suporte: list[str] = Field(
        description="Afirmações que os trechos citados não sustentam; lista vazia se todas forem sustentadas."
    )
    justificativa: str = Field(description="Explicação curta do veredito.")
    fundamentada: bool = Field(description="Verdadeiro somente se todas as afirmações forem sustentadas.")


class ResultadoVerificacao(BaseModel):
    aprovada: bool
    motivo: str
    afirmacoes_sem_suporte: list[str] = Field(default_factory=list)


def extrair_citacoes(resposta: str) -> list[int]:
    return sorted({int(numero) for numero in _CITACAO.findall(resposta)})


def checar_citacoes(resposta: str, quantidade_de_trechos: int) -> ResultadoVerificacao | None:
    """Etapa determinística. Devolve a reprovação, ou None se as citações forem válidas."""
    citacoes = extrair_citacoes(resposta)
    if not citacoes:
        return ResultadoVerificacao(aprovada=False, motivo="A resposta não cita nenhum trecho.")
    invalidas = [n for n in citacoes if not 1 <= n <= quantidade_de_trechos]
    if invalidas:
        lista = ", ".join(f"[{n}]" for n in invalidas)
        return ResultadoVerificacao(
            aprovada=False,
            motivo=f"A resposta cita {lista}, mas só existem os trechos [1] a [{quantidade_de_trechos}].",
        )
    return None


def verificar(
    modelo: BaseChatModel, pergunta: str, resposta: str, trechos: Sequence[TrechoRecuperado]
) -> ResultadoVerificacao:
    reprovacao = checar_citacoes(resposta, len(trechos))
    if reprovacao is not None:
        return reprovacao

    conteudo = f"Pergunta: {pergunta}\n\nTrechos:\n{formatar_trechos(trechos)}\n\nResposta a auditar:\n{resposta}"
    julgamento = modelo.with_structured_output(Julgamento).invoke([SystemMessage(_INSTRUCOES), HumanMessage(conteudo)])
    if not isinstance(julgamento, Julgamento):
        raise TypeError(f"Saída inesperada do verificador: {julgamento!r}")

    # Veredito positivo com afirmações sem suporte é contraditório; prevalece a reprovação.
    aprovada = julgamento.fundamentada and not julgamento.afirmacoes_sem_suporte
    return ResultadoVerificacao(
        aprovada=aprovada,
        motivo=julgamento.justificativa,
        afirmacoes_sem_suporte=julgamento.afirmacoes_sem_suporte,
    )
