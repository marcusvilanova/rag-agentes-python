"""Agente classificador: decide se a pergunta cabe no corpus e prepara a consulta de busca."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

_INSTRUCOES = """Você classifica perguntas enviadas a um assistente que responde apenas com base nos seguintes atos normativos brasileiros:
{documentos}

Classifique como "consulta_legislacao" toda pergunta que esses textos possam responder, mesmo em parte. Classifique como "fora_do_escopo" as demais, incluindo pedidos que não são perguntas (conversa, tarefas, código).

Em "consulta_de_busca", reescreva a pergunta como uma consulta objetiva para busca nesses textos: sem saudações nem contexto pessoal, preservando os termos técnicos e jurídicos usados na pergunta."""


class Classificacao(BaseModel):
    justificativa: str = Field(description="Uma frase explicando a classificação.")
    categoria: Literal["consulta_legislacao", "fora_do_escopo"]
    consulta_de_busca: str = Field(description="Pergunta reescrita como consulta de busca.")


def classificar(modelo: BaseChatModel, pergunta: str, documentos: Sequence[str]) -> Classificacao:
    lista = "\n".join(f"- {titulo}" for titulo in documentos)
    mensagens = [
        SystemMessage(_INSTRUCOES.format(documentos=lista)),
        HumanMessage(pergunta),
    ]
    resultado = modelo.with_structured_output(Classificacao).invoke(mensagens)
    if not isinstance(resultado, Classificacao):
        raise TypeError(f"Saída inesperada do classificador: {resultado!r}")
    return resultado
