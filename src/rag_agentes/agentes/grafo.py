"""Orquestração dos agentes com LangGraph.

O classificador encerra perguntas fora do escopo. As demais seguem para a
recuperação e para o redator; o verificador aprova a resposta ou a devolve ao
redator com o motivo da reprovação, até `max_tentativas`. O diagrama do grafo
compilado sai de `rag diagrama`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_agentes.agentes import classificador, redator, verificador
from rag_agentes.agentes.verificador import ResultadoVerificacao, extrair_citacoes
from rag_agentes.modelos import FonteCitada, RespostaDaPergunta, TrechoRecuperado
from rag_agentes.recuperacao import Recuperador

Situacao = Literal["respondida", "fora_do_escopo", "sem_fundamento"]

MENSAGEM_FORA_DO_ESCOPO = "A pergunta não trata dos documentos indexados, por isso não foi respondida."
MENSAGEM_SEM_FUNDAMENTO = "Não encontrei nos documentos indexados fundamento suficiente para responder com segurança."


class EstadoDoGrafo(TypedDict, total=False):
    pergunta: str
    categoria: str
    consulta_de_busca: str
    trechos: list[TrechoRecuperado]
    resposta: str
    tentativas: int
    verificacao: ResultadoVerificacao
    situacao: Situacao


def construir_grafo(
    modelo: BaseChatModel,
    recuperador: Recuperador,
    documentos: Sequence[str],
    max_tentativas: int,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Monta o grafo. `documentos` são os títulos do corpus, usados pelo classificador."""

    # Os nós recebem `state`: é o nome de parâmetro que o protocolo de nós do LangGraph exige.
    def classificar(state: EstadoDoGrafo) -> EstadoDoGrafo:
        resultado = classificador.classificar(modelo, state["pergunta"], documentos)
        return {"categoria": resultado.categoria, "consulta_de_busca": resultado.consulta_de_busca}

    def recuperar(state: EstadoDoGrafo) -> EstadoDoGrafo:
        return {"trechos": recuperador.recuperar(state["consulta_de_busca"])}

    def redigir(state: EstadoDoGrafo) -> EstadoDoGrafo:
        resposta = redator.redigir(modelo, state["pergunta"], state["trechos"], state.get("verificacao"))
        return {"resposta": resposta, "tentativas": state.get("tentativas", 0) + 1}

    def verificar(state: EstadoDoGrafo) -> EstadoDoGrafo:
        resultado = verificador.verificar(modelo, state["pergunta"], state["resposta"], state["trechos"])
        return {"verificacao": resultado}

    def apos_classificar(estado: EstadoDoGrafo) -> str:
        return "recuperar" if estado["categoria"] == "consulta_legislacao" else "fora_do_escopo"

    def apos_recuperar(estado: EstadoDoGrafo) -> str:
        return "redigir" if estado["trechos"] else "sem_fundamento"

    def apos_redigir(estado: EstadoDoGrafo) -> str:
        resposta = estado["resposta"]
        # Tolera variações como "SEM_FUNDAMENTO." ou "**SEM_FUNDAMENTO**", desde que sem citações.
        declarou_sem_base = redator.SEM_FUNDAMENTO in resposta and not extrair_citacoes(resposta)
        return "sem_fundamento" if declarou_sem_base else "verificar"

    def apos_verificar(estado: EstadoDoGrafo) -> str:
        if estado["verificacao"].aprovada:
            return "respondida"
        return "redigir" if estado["tentativas"] < max_tentativas else "sem_fundamento"

    grafo = StateGraph(EstadoDoGrafo)
    grafo.add_node("classificar", classificar)
    grafo.add_node("recuperar", recuperar)
    grafo.add_node("redigir", redigir)
    grafo.add_node("verificar", verificar)
    grafo.add_node("respondida", lambda _estado: {"situacao": "respondida"})
    grafo.add_node(
        "fora_do_escopo",
        lambda _estado: {"situacao": "fora_do_escopo", "resposta": MENSAGEM_FORA_DO_ESCOPO},
    )
    grafo.add_node(
        "sem_fundamento",
        lambda _estado: {"situacao": "sem_fundamento", "resposta": MENSAGEM_SEM_FUNDAMENTO},
    )

    grafo.add_edge(START, "classificar")
    grafo.add_conditional_edges("classificar", apos_classificar, ["recuperar", "fora_do_escopo"])
    grafo.add_conditional_edges("recuperar", apos_recuperar, ["redigir", "sem_fundamento"])
    grafo.add_conditional_edges("redigir", apos_redigir, ["verificar", "sem_fundamento"])
    grafo.add_conditional_edges("verificar", apos_verificar, ["respondida", "redigir", "sem_fundamento"])
    for final in ("respondida", "fora_do_escopo", "sem_fundamento"):
        grafo.add_edge(final, END)
    return grafo.compile()


def desenhar_grafo() -> str:
    """Diagrama Mermaid do grafo compilado. Os nós não são executados, então
    modelo e recuperador não são necessários."""
    grafo = construir_grafo(
        modelo=cast(BaseChatModel, None),
        recuperador=cast(Recuperador, None),
        documentos=[],
        max_tentativas=2,
    )
    return grafo.get_graph().draw_mermaid()


def montar_resposta(estado: EstadoDoGrafo) -> RespostaDaPergunta:
    """Converte o estado final do grafo na resposta pública da API."""
    situacao = estado["situacao"]
    trechos = estado.get("trechos", [])
    fontes = []
    if situacao == "respondida":
        fontes = [
            FonteCitada(
                numero=numero,
                id=trechos[numero - 1].trecho.id,
                referencia=trechos[numero - 1].trecho.referencia,
                texto=trechos[numero - 1].trecho.texto,
                pontuacao=trechos[numero - 1].pontuacao,
            )
            for numero in extrair_citacoes(estado["resposta"])
        ]
    verificacao = estado.get("verificacao")
    return RespostaDaPergunta(
        pergunta=estado["pergunta"],
        situacao=situacao,
        resposta=estado["resposta"],
        fontes=fontes,
        tentativas=estado.get("tentativas", 0),
        motivo_da_verificacao=verificacao.motivo if verificacao else None,
    )
