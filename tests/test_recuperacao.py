from __future__ import annotations

from pathlib import Path

from apoio import ArmazenamentoEmMemoria, VetorizadorDeTeste, verificar_contrato_de_armazenamento

from rag_agentes.ingestao import carregar_trechos, ingerir
from rag_agentes.recuperacao import Recuperador


def _indice_do_corpus(pasta_corpus: Path) -> tuple[VetorizadorDeTeste, ArmazenamentoEmMemoria]:
    vetorizador, armazenamento = VetorizadorDeTeste(), ArmazenamentoEmMemoria()
    ingerir(pasta_corpus, vetorizador, armazenamento)
    return vetorizador, armazenamento


def test_armazenamento_em_memoria_cumpre_o_contrato() -> None:
    verificar_contrato_de_armazenamento(ArmazenamentoEmMemoria())


def test_ingestao_indexa_todos_os_trechos_e_e_idempotente(pasta_corpus: Path) -> None:
    vetorizador, armazenamento = VetorizadorDeTeste(), ArmazenamentoEmMemoria()
    concluidos: list[str] = []

    totais = ingerir(pasta_corpus, vetorizador, armazenamento, lambda doc, _n: concluidos.append(doc))
    ingerir(pasta_corpus, vetorizador, armazenamento)

    esperado = sum(len(t) for t in carregar_trechos(pasta_corpus).values())
    assert sum(totais.values()) == esperado
    assert armazenamento.contar() == esperado
    assert armazenamento.dimensoes == vetorizador.dimensoes
    assert concluidos == list(totais)


def test_recupera_o_artigo_que_contem_os_termos_da_consulta(pasta_corpus: Path) -> None:
    vetorizador, armazenamento = _indice_do_corpus(pasta_corpus)
    recuperador = Recuperador(vetorizador, armazenamento, k=3)

    resultados = recuperador.recuperar("registros de conexão sob sigilo pelo prazo de 1 (um) ano")

    assert len(resultados) == 3
    assert resultados[0].trecho.id == "marco-civil:art-13"


def test_filtro_por_documento_restringe_os_resultados(pasta_corpus: Path) -> None:
    vetorizador, armazenamento = _indice_do_corpus(pasta_corpus)
    recuperador = Recuperador(vetorizador, armazenamento, k=5)

    resultados = recuperador.recuperar("tratamento de dados pessoais", documentos=["lai"])

    assert resultados
    assert {r.trecho.documento for r in resultados} == {"lai"}


def test_k_da_chamada_prevalece_sobre_o_padrao(pasta_corpus: Path) -> None:
    vetorizador, armazenamento = _indice_do_corpus(pasta_corpus)
    recuperador = Recuperador(vetorizador, armazenamento, k=5)

    assert len(recuperador.recuperar("educação profissional", k=2)) == 2


def test_limiar_descarta_resultados_pouco_similares(pasta_corpus: Path) -> None:
    vetorizador, armazenamento = _indice_do_corpus(pasta_corpus)
    recuperador = Recuperador(vetorizador, armazenamento, k=5, pontuacao_minima=0.99)

    assert recuperador.recuperar("palavras inexistentes zzzz qqqq wwww") == []
