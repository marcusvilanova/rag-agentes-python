from __future__ import annotations

from typing import Any

import pytest
from apoio import recuperado
from mcp import Client

from rag_agentes.modelos import TrechoRecuperado
from rag_agentes.servidor_mcp import criar_servidor_mcp

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class RecuperadorRegistrado:
    def __init__(self) -> None:
        self.chamadas: list[dict[str, Any]] = []

    def recuperar(
        self, consulta: str, k: int | None = None, documentos: list[str] | None = None
    ) -> list[TrechoRecuperado]:
        self.chamadas.append({"consulta": consulta, "k": k, "documentos": documentos})
        return [recuperado("lgpd:art-7", "Art. 7º O tratamento...", 0.88, artigo="7")]


@pytest.fixture
def recuperador() -> RecuperadorRegistrado:
    return RecuperadorRegistrado()


@pytest.fixture
def servidor(recuperador: RecuperadorRegistrado) -> Any:
    return criar_servidor_mcp(recuperador, ["lgpd", "lai"])  # type: ignore[arg-type]


async def test_ferramenta_de_busca_e_anunciada(servidor: Any) -> None:
    async with Client(servidor) as cliente:
        ferramentas = await cliente.list_tools()

    [busca] = [f for f in ferramentas.tools if f.name == "buscar_trechos"]
    assert set(busca.input_schema["properties"]) == {"consulta", "k", "documentos"}
    assert busca.input_schema["required"] == ["consulta"]


async def test_busca_devolve_trechos_estruturados(servidor: Any, recuperador: RecuperadorRegistrado) -> None:
    async with Client(servidor) as cliente:
        resultado = await cliente.call_tool(
            "buscar_trechos", {"consulta": "consentimento", "k": 3, "documentos": ["lgpd"]}
        )

    assert not resultado.is_error
    [trecho] = resultado.structured_content["result"]
    assert trecho["referencia"] == "Lei nº 13.709/2018, art. 7º"
    assert trecho["pontuacao"] == pytest.approx(0.88)
    assert recuperador.chamadas == [{"consulta": "consentimento", "k": 3, "documentos": ["lgpd"]}]


async def test_documento_desconhecido_e_recusado(servidor: Any, recuperador: RecuperadorRegistrado) -> None:
    async with Client(servidor) as cliente:
        resultado = await cliente.call_tool("buscar_trechos", {"consulta": "consentimento", "documentos": ["cdc"]})

    assert resultado.is_error
    assert recuperador.chamadas == []


async def test_k_fora_do_intervalo_e_recusado(servidor: Any, recuperador: RecuperadorRegistrado) -> None:
    async with Client(servidor) as cliente:
        resultado = await cliente.call_tool("buscar_trechos", {"consulta": "consentimento", "k": 50})

    assert resultado.is_error
    assert recuperador.chamadas == []


async def test_consulta_longa_demais_e_recusada(servidor: Any, recuperador: RecuperadorRegistrado) -> None:
    async with Client(servidor) as cliente:
        resultado = await cliente.call_tool("buscar_trechos", {"consulta": "x" * 1001})

    assert resultado.is_error
    assert recuperador.chamadas == []
