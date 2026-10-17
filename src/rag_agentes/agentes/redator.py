"""Agente redator: escreve a resposta usando só os trechos recuperados, com citações [n]."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rag_agentes.agentes.contexto import formatar_trechos
from rag_agentes.agentes.verificador import ResultadoVerificacao
from rag_agentes.modelos import TrechoRecuperado

SEM_FUNDAMENTO = "SEM_FUNDAMENTO"

_INSTRUCOES = f"""Você responde perguntas sobre legislação brasileira usando exclusivamente os trechos numerados fornecidos.

Regras:
1. Use apenas o que está escrito nos trechos. Não use conhecimento externo nem complete lacunas com suposições.
2. Depois de cada afirmação, indique entre colchetes o número do trecho que a sustenta, por exemplo [2] ou [1][3].
3. Mencione o dispositivo pelo nome (por exemplo, "art. 7º da Lei nº 13.709/2018") quando isso ajudar o leitor.
4. Se os trechos não forem suficientes para responder, escreva apenas {SEM_FUNDAMENTO}.
5. Responda em português, de forma direta, em no máximo três parágrafos."""


def redigir(
    modelo: BaseChatModel,
    pergunta: str,
    trechos: Sequence[TrechoRecuperado],
    verificacao_anterior: ResultadoVerificacao | None = None,
) -> str:
    conteudo = f"Pergunta: {pergunta}\n\nTrechos:\n{formatar_trechos(trechos)}"
    if verificacao_anterior is not None and not verificacao_anterior.aprovada:
        conteudo += (
            f"\n\nUma versão anterior da resposta foi rejeitada pelo verificador. Motivo: {verificacao_anterior.motivo}"
        )
        if verificacao_anterior.afirmacoes_sem_suporte:
            itens = "\n".join(f"- {a}" for a in verificacao_anterior.afirmacoes_sem_suporte)
            conteudo += f"\nAfirmações sem sustentação nos trechos:\n{itens}"
        conteudo += "\nEscreva uma nova resposta que corrija esses pontos."

    mensagem = modelo.invoke([SystemMessage(_INSTRUCOES), HumanMessage(conteudo)])
    return mensagem.text.strip()
