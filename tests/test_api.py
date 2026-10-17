from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag_agentes.api import criar_app
from rag_agentes.modelos import FonteCitada, RespostaDaPergunta


class ServicoFalso:
    def __init__(self) -> None:
        self.perguntas: list[str] = []
        self.encerrado = False

    def perguntar(self, pergunta: str) -> RespostaDaPergunta:
        self.perguntas.append(pergunta)
        return RespostaDaPergunta(
            pergunta=pergunta,
            situacao="respondida",
            resposta="Mediante consentimento [1].",
            fontes=[
                FonteCitada(
                    numero=1, id="lgpd:art-7", referencia="Lei nº 13.709/2018, art. 7º", texto="t", pontuacao=0.9
                )
            ],
            tentativas=1,
        )

    def encerrar(self) -> None:
        self.encerrado = True


@pytest.fixture
def servico() -> ServicoFalso:
    return ServicoFalso()


@pytest.fixture
def cliente(servico: ServicoFalso) -> Iterator[TestClient]:
    with TestClient(criar_app(lambda: servico)) as cliente:  # type: ignore[arg-type,return-value]
        yield cliente


def test_perguntar_devolve_resposta_com_fontes(cliente: TestClient, servico: ServicoFalso) -> None:
    resposta = cliente.post("/perguntar", json={"pergunta": "Quando posso tratar dados pessoais?"})

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["situacao"] == "respondida"
    assert corpo["fontes"][0]["id"] == "lgpd:art-7"
    assert corpo["em_cache"] is False
    assert servico.perguntas == ["Quando posso tratar dados pessoais?"]


@pytest.mark.parametrize("corpo", [{}, {"pergunta": "oi"}, {"pergunta": "      "}, {"pergunta": "x" * 1001}])
def test_perguntar_valida_a_entrada(cliente: TestClient, servico: ServicoFalso, corpo: dict[str, str]) -> None:
    resposta = cliente.post("/perguntar", json=corpo)

    assert resposta.status_code == 422
    assert servico.perguntas == []


def test_saude(cliente: TestClient) -> None:
    assert cliente.get("/saude").json() == {"situacao": "ok"}


def test_encerrar_o_app_descarrega_o_servico(servico: ServicoFalso) -> None:
    with TestClient(criar_app(lambda: servico)):  # type: ignore[arg-type,return-value]
        pass

    assert servico.encerrado
