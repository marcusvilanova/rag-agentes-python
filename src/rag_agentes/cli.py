"""Linha de comando: `rag <comando>`. Use `rag --help` para a lista."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag_agentes.configuracao import Configuracao, obter_configuracao

_CONJUNTO_PADRAO = Path("avaliacao/perguntas_v1.jsonl")


def _corpus(configuracao: Configuracao, _argumentos: argparse.Namespace) -> None:
    from rag_agentes.corpus import gerar_corpus

    for caminho in gerar_corpus(configuracao.pasta_corpus / "fontes.toml", configuracao.pasta_corpus):
        print(f"gravado {caminho}")


def _ingerir(configuracao: Configuracao, _argumentos: argparse.Namespace) -> None:
    from rag_agentes.ingestao import ingerir
    from rag_agentes.servico import criar_armazenamento, criar_cache, criar_vetorizador

    try:
        totais = ingerir(
            configuracao.pasta_corpus,
            criar_vetorizador(configuracao),
            criar_armazenamento(configuracao),
            ao_concluir_documento=lambda documento, quantidade: print(f"{documento}: {quantidade} trechos"),
        )
        print(f"total: {sum(totais.values())} trechos indexados")
    finally:
        # Mesmo uma ingestão interrompida pode ter substituído documentos.
        removidas = criar_cache(configuracao).limpar()
        print(f"cache: {removidas} respostas removidas")


def _perguntar(configuracao: Configuracao, argumentos: argparse.Namespace) -> None:
    from rag_agentes.servico import criar_servico

    servico = criar_servico(configuracao)
    try:
        resposta = servico.perguntar(argumentos.pergunta)
    finally:
        servico.encerrar()
    print(f"[{resposta.situacao}; tentativas: {resposta.tentativas}; cache: {resposta.em_cache}]\n")
    print(resposta.resposta)
    if resposta.fontes:
        print("\nFontes:")
        for fonte in resposta.fontes:
            print(f"  [{fonte.numero}] {fonte.referencia} ({fonte.id}, pontuação {fonte.pontuacao:.3f})")


def _avaliar(configuracao: Configuracao, argumentos: argparse.Namespace) -> None:
    from rag_agentes import avaliacao
    from rag_agentes.servico import criar_rastreamento, criar_recuperador, criar_servico

    itens = avaliacao.carregar_conjunto(argumentos.conjunto)
    detalhes: list[avaliacao.ResultadoDeRecuperacao | avaliacao.ResultadoCompleto]
    if argumentos.completo:
        # Sem cache: a avaliação precisa medir o grafo, não respostas guardadas.
        servico = criar_servico(configuracao, usar_cache=False)
        try:
            completos = avaliacao.avaliar_completo(itens, servico.perguntar)
        finally:
            servico.encerrar()
        resumo = avaliacao.resumir_completo(itens, completos)
        detalhes = list(completos)
    else:
        rastreamento = criar_rastreamento(configuracao)
        try:
            recuperador = criar_recuperador(configuracao, rastreamento)
            de_recuperacao = avaliacao.avaliar_recuperacao(itens, recuperador, argumentos.k)
        finally:
            rastreamento.descarregar()
        resumo = avaliacao.resumir_recuperacao(de_recuperacao)
        detalhes = list(de_recuperacao)

    for nome, valor in resumo.items():
        print(f"{nome}: {valor:.3f}" if isinstance(valor, float) else f"{nome}: {valor}")
    if argumentos.saida:
        avaliacao.gravar_relatorio(argumentos.saida, resumo, detalhes)
        print(f"relatório: {argumentos.saida}")


def _api(_configuracao: Configuracao, argumentos: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("rag_agentes.api:app", host=argumentos.host, port=argumentos.porta)


def _mcp(configuracao: Configuracao, _argumentos: argparse.Namespace) -> None:
    from rag_agentes.corpus import carregar_fontes
    from rag_agentes.servico import criar_rastreamento, criar_recuperador
    from rag_agentes.servidor_mcp import criar_servidor_mcp

    documentos = [fonte.id for fonte in carregar_fontes(configuracao.pasta_corpus / "fontes.toml")]
    recuperador = criar_recuperador(configuracao, criar_rastreamento(configuracao))
    criar_servidor_mcp(recuperador, documentos).run()


def _diagrama(_configuracao: Configuracao, _argumentos: argparse.Namespace) -> None:
    from rag_agentes.agentes.grafo import desenhar_grafo

    print(desenhar_grafo())


def _analisador() -> argparse.ArgumentParser:
    analisador = argparse.ArgumentParser(prog="rag", description=__doc__)
    comandos = analisador.add_subparsers(dest="comando", required=True)

    comandos.add_parser("corpus", help="baixa as leis do Planalto e grava o texto limpo").set_defaults(acao=_corpus)
    comandos.add_parser("ingerir", help="fragmenta, vetoriza e indexa o corpus").set_defaults(acao=_ingerir)

    perguntar = comandos.add_parser("perguntar", help="responde uma pergunta pelo grafo de agentes")
    perguntar.add_argument("pergunta")
    perguntar.set_defaults(acao=_perguntar)

    avaliar = comandos.add_parser("avaliar", help="mede a recuperação (ou o fluxo completo) no conjunto de avaliação")
    avaliar.add_argument("--conjunto", type=Path, default=_CONJUNTO_PADRAO)
    avaliar.add_argument("--k", type=int, default=5)
    avaliar.add_argument("--completo", action="store_true", help="executa o grafo inteiro (consome cota do LLM)")
    avaliar.add_argument("--saida", type=Path, help="grava o relatório em JSON")
    avaliar.set_defaults(acao=_avaliar)

    api = comandos.add_parser("api", help="sobe a API HTTP")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--porta", type=int, default=8000)
    api.set_defaults(acao=_api)

    comandos.add_parser("mcp", help="sobe o servidor MCP (stdio)").set_defaults(acao=_mcp)
    comandos.add_parser("diagrama", help="imprime o grafo de agentes em Mermaid").set_defaults(acao=_diagrama)
    return analisador


def main(argumentos: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    analisador = _analisador()
    lidos = analisador.parse_args(argumentos)
    try:
        lidos.acao(obter_configuracao(), lidos)
    except RuntimeError as erro:
        print(f"erro: {erro}", file=sys.stderr)
        raise SystemExit(1) from erro
