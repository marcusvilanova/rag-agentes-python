from __future__ import annotations

from pathlib import Path

import pytest

from rag_agentes.fragmentacao import fragmentar
from rag_agentes.ingestao import carregar_trechos
from rag_agentes.modelos import formatar_artigo

TEXTO = """LEI Nº 99.999, DE 1º DE JANEIRO DE 2030
Dispõe sobre um assunto de teste.
O PRESIDENTE DA REPÚBLICA Faço saber que o Congresso Nacional decreta e eu sanciono a seguinte Lei:
CAPÍTULO I
DISPOSIÇÕES PRELIMINARES
Art. 1º Esta Lei trata de um assunto de teste.
Parágrafo único. O parágrafo complementa o caput.
Seção I
Das Definições
Art. 2º Para os fins desta Lei, considera-se:
I - termo: palavra com significado próprio;
II - definição: enunciado do significado.
CAPÍTULO II
DAS OBRIGAÇÕES
Art. 3º (VETADO).
Art. 4º-A Fica incluída uma obrigação. (Incluído pela Lei nº 99.998, de 2031)
Art. 10. Este artigo tem numeração cardinal.
Art. 11. (Revogado pela Lei nº 99.997, de 2032)
Art. 55-B. Este artigo tem sufixo.
Brasília, 1º de janeiro de 2030; 209º da Independência.
FULANO DE TAL
"""


def test_cada_artigo_vira_um_trecho_com_id_estavel() -> None:
    trechos = fragmentar("teste", "Lei de Teste", TEXTO)

    assert [t.id for t in trechos] == [
        "teste:preambulo",
        "teste:art-1",
        "teste:art-2",
        "teste:art-4-A",
        "teste:art-10",
        "teste:art-55-B",
    ]
    assert [t.ordem for t in trechos] == list(range(len(trechos)))


def test_paragrafos_e_incisos_ficam_no_mesmo_trecho_do_caput() -> None:
    art2 = fragmentar("teste", "Lei de Teste", TEXTO)[2]

    assert art2.texto.splitlines() == [
        "Art. 2º Para os fins desta Lei, considera-se:",
        "I - termo: palavra com significado próprio;",
        "II - definição: enunciado do significado.",
    ]


def test_secao_registra_o_caminho_estrutural_e_reinicia_niveis_inferiores() -> None:
    trechos = {t.id: t for t in fragmentar("teste", "Lei de Teste", TEXTO)}

    assert trechos["teste:art-1"].secao == "CAPÍTULO I - DISPOSIÇÕES PRELIMINARES"
    assert trechos["teste:art-2"].secao == "CAPÍTULO I - DISPOSIÇÕES PRELIMINARES > Seção I - Das Definições"
    assert trechos["teste:art-10"].secao == "CAPÍTULO II - DAS OBRIGAÇÕES"


def test_artigos_vetados_ou_revogados_sao_descartados() -> None:
    ids = {t.id for t in fragmentar("teste", "Lei de Teste", TEXTO)}

    assert "teste:art-3" not in ids
    assert "teste:art-11" not in ids


def test_vetos_e_revogacoes_com_anotacao_ou_em_varias_linhas_sao_descartados() -> None:
    # Formas reais: LDB art. 79-A e LGPD art. 55.
    texto = (
        "Art. 1º Texto com conteúdo.\n"
        "Art. 79-A. (VETADO) (Incluído pela Lei nº 10.639, de 9.1.2003)\n"
        "Art. 55. (VETADO).\n"
        "(Revogado pela Lei nº 14.460, de 2022)\n"
        "(Revogado pela Lei nº 14.460, de 2022)\n"
        "Art. 56. Texto vigente. (Incluído pela Lei nº 1, de 2000)\n"
        "Parágrafo único. (VETADO)."
    )

    ids = [t.id for t in fragmentar("teste", "Lei", texto)]

    assert ids == ["teste:art-1", "teste:art-56"]


def test_titulo_com_sufixo_e_anotacao_nao_vaza_para_o_artigo_anterior() -> None:
    # Formas reais da LDB: "Seção IV-A" e "CAPÍTULO V-A", com a anotação antes ou depois do nome.
    texto = (
        "Art. 36. Texto do artigo 36.\n"
        "Seção IV-A\n"
        "Da Educação Profissional Técnica de Nível Médio\n"
        "(Incluído pela Lei nº 11.741, de 2008)\n"
        "Art. 36-A. Texto do artigo 36-A.\n"
        "CAPÍTULO V-A\n"
        "(Incluído pela Lei nº 14.191, de 2021)\n"
        "DA EDUCAÇÃO BILÍNGUE DE SURDOS\n"
        "Art. 60-A. Texto do artigo 60-A."
    )

    trechos = {t.id: t for t in fragmentar("ldb", "LDB", texto)}

    assert trechos["ldb:art-36"].texto == "Art. 36. Texto do artigo 36."
    assert trechos["ldb:art-36-A"].secao == "Seção IV-A - Da Educação Profissional Técnica de Nível Médio"
    assert trechos["ldb:art-36-A"].texto == "Art. 36-A. Texto do artigo 36-A."
    assert trechos["ldb:art-60-A"].secao == "CAPÍTULO V-A - DA EDUCAÇÃO BILÍNGUE DE SURDOS"


def test_hifen_separado_do_numero_nao_e_sufixo_de_artigo() -> None:
    ids = [t.id for t in fragmentar("teste", "Lei", "Art. 1º - O Senac terá sede.\nArt. 2º-A Incluído.")]

    assert ids == ["teste:art-1", "teste:art-2-A"]


def test_assinatura_encerra_o_texto_da_lei() -> None:
    ultimo = fragmentar("teste", "Lei de Teste", TEXTO)[-1]

    assert "FULANO" not in ultimo.texto
    assert "Brasília" not in ultimo.texto


def test_artigo_longo_e_dividido_repetindo_o_caput() -> None:
    incisos = [f"{numero} - hipótese número {numero} " + "x" * 80 for numero in range(1, 21)]
    texto = "Art. 7º O tratamento somente poderá ser realizado nas seguintes hipóteses:\n" + "\n".join(incisos)

    partes = fragmentar("lgpd", "LGPD", texto, tamanho_maximo=500)

    assert len(partes) > 1
    assert [p.id for p in partes] == [f"lgpd:art-7:parte-{n}" for n in range(1, len(partes) + 1)]
    assert all(p.texto.startswith("Art. 7º O tratamento") for p in partes)
    assert all(len(p.texto) <= 500 for p in partes)
    linhas_sem_caput = [linha for p in partes for linha in p.texto.splitlines()[1:]]
    assert linhas_sem_caput == incisos


def test_linha_maior_que_o_limite_nao_e_cortada() -> None:
    texto = "Art. 1º Caput.\n" + "§ 1º " + "palavra " * 100

    partes = fragmentar("teste", "Lei", texto, tamanho_maximo=100)

    assert len(partes) == 1
    assert partes[0].texto.endswith("palavra")


def test_ids_repetidos_recebem_sufixo() -> None:
    texto = "Art. 1º Primeiro.\nArt. 1º Repetido por erro de origem."

    ids = [t.id for t in fragmentar("teste", "Lei", texto)]

    assert ids == ["teste:art-1", "teste:art-1~2"]


@pytest.mark.parametrize(
    ("numero", "esperado"),
    [("1", "art. 1º"), ("9", "art. 9º"), ("10", "art. 10"), ("4-A", "art. 4º-A"), ("55-B", "art. 55-B")],
)
def test_formatar_artigo_segue_a_tecnica_legislativa(numero: str, esperado: str) -> None:
    assert formatar_artigo(numero) == esperado


def test_referencia_omite_o_apelido_da_lei() -> None:
    trecho = fragmentar("lgpd", "Lei nº 13.709/2018 (Lei Geral de Proteção de Dados Pessoais)", TEXTO)[1]

    assert trecho.referencia == "Lei nº 13.709/2018, art. 1º"


def test_corpus_real_gera_ids_unicos_e_trechos_dentro_do_limite(pasta_corpus: Path) -> None:
    por_documento = carregar_trechos(pasta_corpus)
    todos = [t for trechos in por_documento.values() for t in trechos]
    ids = [t.id for t in todos]

    assert set(por_documento) == {"lgpd", "marco-civil", "lai", "ldb", "senac"}
    assert len(ids) == len(set(ids))
    assert not [i for i in ids if "~" in i], "o corpus não deveria ter artigos com número repetido"
    assert all(t.texto.strip() for t in todos)
    assert max(len(t.texto) for t in todos) <= 1500
    assert any(i.startswith("lgpd:art-7") for i in ids)
