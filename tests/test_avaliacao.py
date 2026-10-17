from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_agentes.avaliacao import (
    ItemDeAvaliacao,
    ResultadoDeRecuperacao,
    avaliar_completo,
    avaliar_ids,
    carregar_conjunto,
    corresponde,
    resumir_completo,
    resumir_recuperacao,
)
from rag_agentes.ingestao import carregar_trechos
from rag_agentes.modelos import FonteCitada, RespostaDaPergunta


def _item(fontes: list[str], categoria: str = "consulta_legislacao", id_: str = "q") -> ItemDeAvaliacao:
    return ItemDeAvaliacao(id=id_, pergunta="p", categoria=categoria, fontes_esperadas=fontes)  # type: ignore[arg-type]


def test_conjunto_versionado_e_valido(arquivo_de_avaliacao: Path) -> None:
    itens = carregar_conjunto(arquivo_de_avaliacao)

    assert len(itens) >= 25
    assert len({i.id for i in itens}) == len(itens)
    assert len({i.pergunta for i in itens}) == len(itens)
    assert any(i.categoria == "fora_do_escopo" for i in itens)


def test_toda_fonte_esperada_existe_no_corpus(arquivo_de_avaliacao: Path, pasta_corpus: Path) -> None:
    # Protege o conjunto contra mudanças na fragmentação que renomeiem trechos.
    ids = [t.id for trechos in carregar_trechos(pasta_corpus).values() for t in trechos]
    esperadas = {f for item in carregar_conjunto(arquivo_de_avaliacao) for f in item.fontes_esperadas}

    ausentes = sorted(f for f in esperadas if not any(corresponde(i, f) for i in ids))

    assert ausentes == []


def test_item_no_escopo_sem_fonte_e_invalido() -> None:
    with pytest.raises(ValidationError, match="precisam de fontes"):
        _item([])


def test_item_fora_do_escopo_com_fonte_e_invalido() -> None:
    with pytest.raises(ValidationError, match="não têm fontes"):
        _item(["lgpd:art-7"], categoria="fora_do_escopo")


def test_correspondencia_aceita_partes_do_artigo_mas_nao_prefixos_numericos() -> None:
    assert corresponde("lgpd:art-7", "lgpd:art-7")
    assert corresponde("lgpd:art-7:parte-2", "lgpd:art-7")
    assert not corresponde("lgpd:art-70", "lgpd:art-7")


def test_avaliar_ids_calcula_revocacao_e_posicao_do_primeiro_acerto() -> None:
    resultado = avaliar_ids(_item(["lgpd:art-7", "lgpd:art-11"]), ["lgpd:art-5", "lgpd:art-7:parte-1", "lgpd:art-8"])

    assert resultado.revocacao == 0.5
    assert resultado.posicao_do_primeiro_acerto == 2


def test_resumo_da_recuperacao() -> None:
    resumo = resumir_recuperacao(
        [
            ResultadoDeRecuperacao("a", revocacao=1.0, posicao_do_primeiro_acerto=1),
            ResultadoDeRecuperacao("b", revocacao=0.5, posicao_do_primeiro_acerto=2),
            ResultadoDeRecuperacao("c", revocacao=0.0, posicao_do_primeiro_acerto=None),
            ResultadoDeRecuperacao("d", revocacao=1.0, posicao_do_primeiro_acerto=4),
        ]
    )

    assert resumo == {"perguntas": 4, "acerto_em_k": 0.75, "revocacao_em_k": 0.625, "mrr": pytest.approx(0.4375)}


def test_avaliacao_completa_mede_classificacao_e_citacoes() -> None:
    itens = [
        _item(["lgpd:art-7"], id_="certa"),
        _item(["lgpd:art-48"], id_="citou-outra"),
        _item([], categoria="fora_do_escopo", id_="fora"),
    ]
    respostas = {
        "certa": ("respondida", "lgpd:art-7:parte-1"),
        "citou-outra": ("respondida", "lgpd:art-7"),
        "fora": ("respondida", "lgpd:art-1"),
    }

    def perguntar_por_id(id_: str) -> RespostaDaPergunta:
        situacao, fonte = respostas[id_]
        return RespostaDaPergunta(
            pergunta=id_,
            situacao=situacao,  # type: ignore[arg-type]
            resposta="r [1]",
            fontes=[FonteCitada(numero=1, id=fonte, referencia="r", texto="t", pontuacao=0.9)],
            tentativas=1,
        )

    itens_com_pergunta = [i.model_copy(update={"pergunta": i.id}) for i in itens]
    resultados = avaliar_completo(itens_com_pergunta, perguntar_por_id)
    resumo = resumir_completo(itens_com_pergunta, resultados)

    assert [r.citou_fonte_esperada for r in resultados] == [True, False, False]
    assert resumo["classificacao_correta"] == pytest.approx(2 / 3)
    assert resumo["respondidas_no_escopo"] == 1.0
    assert resumo["citou_fonte_esperada"] == 0.5
