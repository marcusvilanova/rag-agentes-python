from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from apoio import ModeloRoteirizado, recuperado

from rag_agentes.agentes.classificador import Classificacao
from rag_agentes.agentes.grafo import (
    MENSAGEM_FORA_DO_ESCOPO,
    MENSAGEM_SEM_FUNDAMENTO,
    construir_grafo,
    desenhar_grafo,
    montar_resposta,
)
from rag_agentes.agentes.redator import SEM_FUNDAMENTO
from rag_agentes.agentes.verificador import Julgamento
from rag_agentes.modelos import TrechoRecuperado

TRECHOS = [
    recuperado("lgpd:art-7", "Art. 7º O tratamento somente poderá ser realizado mediante consentimento.", 0.91),
    recuperado("lgpd:art-48", "Art. 48. O controlador deverá comunicar incidente de segurança.", 0.85),
]

NO_ESCOPO = Classificacao(
    justificativa="Trata da LGPD.", categoria="consulta_legislacao", consulta_de_busca="hipóteses de tratamento"
)
FORA = Classificacao(justificativa="Culinária.", categoria="fora_do_escopo", consulta_de_busca="bolo")
APROVA = Julgamento(afirmacoes_sem_suporte=[], justificativa="Sustentada.", fundamentada=True)
REPROVA = Julgamento(
    afirmacoes_sem_suporte=["Exige contrato escrito."], justificativa="Não consta.", fundamentada=False
)


class RecuperadorFixo:
    def __init__(self, trechos: Sequence[TrechoRecuperado]) -> None:
        self._trechos = list(trechos)
        self.consultas: list[str] = []

    def recuperar(self, consulta: str, **_: Any) -> list[TrechoRecuperado]:
        self.consultas.append(consulta)
        return self._trechos


def _executar(modelo: ModeloRoteirizado, recuperador: RecuperadorFixo, max_tentativas: int = 2) -> dict[str, Any]:
    grafo = construir_grafo(modelo, recuperador, ["Lei nº 13.709/2018"], max_tentativas)  # type: ignore[arg-type]
    return grafo.invoke({"pergunta": "Quando posso tratar dados pessoais?"})


def test_fluxo_completo_com_resposta_aprovada() -> None:
    modelo = ModeloRoteirizado(estruturadas=[NO_ESCOPO, APROVA], textos=["Mediante consentimento [1]."])
    recuperador = RecuperadorFixo(TRECHOS)

    estado = _executar(modelo, recuperador)

    assert estado["situacao"] == "respondida"
    assert estado["tentativas"] == 1
    assert recuperador.consultas == ["hipóteses de tratamento"]
    assert modelo.tipos_chamados() == ["Classificacao", "texto", "Julgamento"]


def test_pergunta_fora_do_escopo_nao_consulta_o_indice() -> None:
    modelo = ModeloRoteirizado(estruturadas=[FORA])
    recuperador = RecuperadorFixo(TRECHOS)

    estado = _executar(modelo, recuperador)

    assert estado["situacao"] == "fora_do_escopo"
    assert estado["resposta"] == MENSAGEM_FORA_DO_ESCOPO
    assert recuperador.consultas == []


def test_sem_trechos_recuperados_nao_chama_o_redator() -> None:
    modelo = ModeloRoteirizado(estruturadas=[NO_ESCOPO])

    estado = _executar(modelo, RecuperadorFixo([]))

    assert estado["situacao"] == "sem_fundamento"
    assert estado["resposta"] == MENSAGEM_SEM_FUNDAMENTO
    assert modelo.tipos_chamados() == ["Classificacao"]


def test_redator_que_declara_falta_de_base_encerra_sem_verificar() -> None:
    modelo = ModeloRoteirizado(estruturadas=[NO_ESCOPO], textos=[SEM_FUNDAMENTO])

    estado = _executar(modelo, RecuperadorFixo(TRECHOS))

    assert estado["situacao"] == "sem_fundamento"
    assert modelo.tipos_chamados() == ["Classificacao", "texto"]


def test_variacoes_de_sem_fundamento_sem_citacao_tambem_encerram() -> None:
    modelo = ModeloRoteirizado(estruturadas=[NO_ESCOPO], textos=[f"**{SEM_FUNDAMENTO}**."])

    estado = _executar(modelo, RecuperadorFixo(TRECHOS))

    assert estado["situacao"] == "sem_fundamento"
    assert modelo.tipos_chamados() == ["Classificacao", "texto"]


def test_reprovacao_devolve_ao_redator_com_o_motivo() -> None:
    modelo = ModeloRoteirizado(
        estruturadas=[NO_ESCOPO, REPROVA, APROVA],
        textos=["Exige contrato escrito [1].", "Mediante consentimento [1]."],
    )

    estado = _executar(modelo, RecuperadorFixo(TRECHOS))

    assert estado["situacao"] == "respondida"
    assert estado["tentativas"] == 2
    assert estado["resposta"] == "Mediante consentimento [1]."
    segunda_redacao = modelo.chamadas[3][1][-1].content
    assert "rejeitada pelo verificador" in segunda_redacao
    assert "Exige contrato escrito." in segunda_redacao


def test_reprovacoes_esgotam_as_tentativas() -> None:
    modelo = ModeloRoteirizado(
        estruturadas=[NO_ESCOPO, REPROVA, REPROVA],
        textos=["Exige contrato escrito [1].", "Exige contrato escrito [2]."],
    )

    estado = _executar(modelo, RecuperadorFixo(TRECHOS), max_tentativas=2)

    assert estado["situacao"] == "sem_fundamento"
    assert estado["resposta"] == MENSAGEM_SEM_FUNDAMENTO
    assert estado["tentativas"] == 2


def test_montar_resposta_lista_apenas_os_trechos_citados() -> None:
    estado = {
        "pergunta": "p",
        "situacao": "respondida",
        "resposta": "Deve comunicar [2].",
        "trechos": TRECHOS,
        "tentativas": 1,
    }

    resposta = montar_resposta(estado)  # type: ignore[arg-type]

    assert [(f.numero, f.id, f.pontuacao) for f in resposta.fontes] == [(2, "lgpd:art-48", 0.85)]
    assert resposta.fontes[0].referencia == "Lei nº 13.709/2018, art. 1º"


def test_resposta_nao_aprovada_nao_expoe_fontes() -> None:
    estado = {"pergunta": "p", "situacao": "sem_fundamento", "resposta": MENSAGEM_SEM_FUNDAMENTO, "trechos": TRECHOS}

    assert montar_resposta(estado).fontes == []  # type: ignore[arg-type]


def test_diagrama_do_grafo_contem_todos_os_nos() -> None:
    diagrama = desenhar_grafo()

    for no in ("classificar", "recuperar", "redigir", "verificar", "respondida", "fora_do_escopo", "sem_fundamento"):
        assert no in diagrama
