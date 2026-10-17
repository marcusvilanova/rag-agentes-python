"""Dublês de teste: vetorizador determinístico, índice em memória e modelo de chat roteirizado.

Nenhum deles é usado fora dos testes. O vetorizador por hash de palavras não
captura significado; serve para exercitar o encadeamento (vetorizar, indexar,
buscar, filtrar) sem chamar uma API paga e com resultado reproduzível.
"""

from __future__ import annotations

import math
import re
import time
import zlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from rag_agentes.armazenamento import ArmazenamentoVetorial
from rag_agentes.modelos import Trecho, TrechoRecuperado

DIMENSOES_DE_TESTE = 256
_PALAVRA = re.compile(r"\w{3,}")


class VetorizadorDeTeste:
    """Saco de palavras com hashing (crc32), normalizado. Determinístico entre execuções."""

    @property
    def dimensoes(self) -> int:
        return DIMENSOES_DE_TESTE

    def vetorizar_trechos(self, trechos: Sequence[Trecho]) -> list[list[float]]:
        return [self._vetorizar(t.texto) for t in trechos]

    def vetorizar_consulta(self, consulta: str) -> list[float]:
        return self._vetorizar(consulta)

    @staticmethod
    def _vetorizar(texto: str) -> list[float]:
        vetor = [0.0] * DIMENSOES_DE_TESTE
        for palavra in _PALAVRA.findall(texto.casefold()):
            vetor[zlib.crc32(palavra.encode()) % DIMENSOES_DE_TESTE] += 1.0
        norma = math.sqrt(sum(v * v for v in vetor)) or 1.0
        return [v / norma for v in vetor]


class ArmazenamentoEmMemoria:
    """Busca exata por cosseno, com a mesma escala de pontuação do MongoDB."""

    def __init__(self) -> None:
        self._registros: dict[str, tuple[Trecho, list[float]]] = {}

    def preparar(self, dimensoes: int) -> None:
        self.dimensoes = dimensoes

    def substituir_documento(
        self, documento: str, trechos: Sequence[Trecho], vetores: Sequence[Sequence[float]]
    ) -> None:
        ids = {t.id for t in trechos}
        self._registros = {
            chave: valor for chave, valor in self._registros.items() if valor[0].documento != documento or chave in ids
        }
        for trecho, vetor in zip(trechos, vetores, strict=True):
            self._registros[trecho.id] = (trecho, list(vetor))

    def buscar(self, vetor: Sequence[float], k: int, documentos: Sequence[str] | None = None) -> list[TrechoRecuperado]:
        candidatos = [
            TrechoRecuperado(trecho=trecho, pontuacao=(1 + _cosseno(vetor, armazenado)) / 2)
            for trecho, armazenado in self._registros.values()
            if not documentos or trecho.documento in documentos
        ]
        return sorted(candidatos, key=lambda item: item.pontuacao, reverse=True)[:k]

    def contar(self) -> int:
        return len(self._registros)


def _cosseno(a: Sequence[float], b: Sequence[float]) -> float:
    produto = sum(x * y for x, y in zip(a, b, strict=True))
    normas = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return produto / normas if normas else 0.0


class ModeloRoteirizado:
    """Substitui o modelo de chat devolvendo respostas pré-definidas, na ordem.

    Saídas estruturadas e texto livre têm roteiros separados. Toda chamada fica
    registrada em `chamadas`, para os testes conferirem o que foi enviado.
    Uma chamada sem resposta roteirizada falha o teste.
    """

    def __init__(self, estruturadas: Iterable[BaseModel] = (), textos: Iterable[str] = ()) -> None:
        self._estruturadas = list(estruturadas)
        self._textos = list(textos)
        self.chamadas: list[tuple[str, list[BaseMessage]]] = []

    def with_structured_output(self, esquema: type[BaseModel], **_opcoes: Any) -> RunnableLambda[Any, Any]:
        def responder(mensagens: list[BaseMessage]) -> BaseModel:
            self.chamadas.append((esquema.__name__, mensagens))
            if not self._estruturadas:
                raise AssertionError(f"Chamada inesperada ao modelo com saída {esquema.__name__}.")
            resposta = self._estruturadas.pop(0)
            if not isinstance(resposta, esquema):
                raise AssertionError(f"O roteiro previa {type(resposta).__name__}, mas foi pedido {esquema.__name__}.")
            return resposta

        return RunnableLambda(responder)

    def invoke(self, mensagens: list[BaseMessage], *_args: Any, **_kwargs: Any) -> AIMessage:
        self.chamadas.append(("texto", mensagens))
        if not self._textos:
            raise AssertionError("Chamada inesperada ao modelo com saída de texto.")
        return AIMessage(content=self._textos.pop(0))

    def tipos_chamados(self) -> list[str]:
        return [tipo for tipo, _ in self.chamadas]


def trecho(
    id_: str,
    texto: str,
    documento: str = "lgpd",
    artigo: str | None = "1",
    secao: str = "",
) -> Trecho:
    return Trecho(
        id=id_,
        documento=documento,
        titulo_documento="Lei nº 13.709/2018 (Lei Geral de Proteção de Dados Pessoais)",
        artigo=artigo,
        secao=secao,
        texto=texto,
        ordem=0,
    )


def recuperado(id_: str, texto: str, pontuacao: float = 0.9, **campos: Any) -> TrechoRecuperado:
    return TrechoRecuperado(trecho=trecho(id_, texto, **campos), pontuacao=pontuacao)


def verificar_contrato_de_armazenamento(
    armazenamento: ArmazenamentoVetorial,
    aguardar_indexacao: Callable[[int], None] = lambda _quantidade: None,
) -> None:
    """Comportamento que toda implementação de ArmazenamentoVetorial deve ter.

    `aguardar_indexacao(n)` existe para índices com consistência eventual, como o
    Atlas Vector Search, em que um documento gravado demora a aparecer na busca.
    """
    vetorizador = VetorizadorDeTeste()
    armazenamento.preparar(vetorizador.dimensoes)
    lgpd = [
        trecho("lgpd:art-7", "Art. 7º O tratamento de dados pessoais mediante consentimento do titular", artigo="7"),
        trecho("lgpd:art-48", "Art. 48. O controlador deverá comunicar incidente de segurança", artigo="48"),
    ]
    lai = [
        trecho(
            "lai:art-11",
            "Art. 11. O órgão público deverá conceder o acesso imediato à informação",
            documento="lai",
            artigo="11",
        )
    ]
    armazenamento.substituir_documento("lgpd", lgpd, vetorizador.vetorizar_trechos(lgpd))
    armazenamento.substituir_documento("lai", lai, vetorizador.vetorizar_trechos(lai))
    aguardar_indexacao(3)
    assert armazenamento.contar() == 3

    consulta = vetorizador.vetorizar_consulta("comunicar incidente de segurança ao controlador")
    resultados = armazenamento.buscar(consulta, k=3)
    assert resultados[0].trecho.id == "lgpd:art-48"
    assert resultados[0].trecho == lgpd[1]
    assert all(0.0 <= r.pontuacao <= 1.0 for r in resultados)
    assert [r.pontuacao for r in resultados] == sorted((r.pontuacao for r in resultados), reverse=True)

    filtrados = armazenamento.buscar(consulta, k=3, documentos=["lai"])
    assert [r.trecho.id for r in filtrados] == ["lai:art-11"]

    # Reingerir um documento remove os trechos que deixaram de existir nele.
    armazenamento.substituir_documento("lgpd", lgpd[:1], vetorizador.vetorizar_trechos(lgpd[:1]))
    aguardar_indexacao(2)
    assert armazenamento.contar() == 2
    assert "lgpd:art-48" not in {r.trecho.id for r in armazenamento.buscar(consulta, k=3)}


def aguardar(condicao: Callable[[], bool], tempo_limite: float = 60.0, intervalo: float = 0.5) -> None:
    prazo = time.monotonic() + tempo_limite
    while time.monotonic() < prazo:
        if condicao():
            return
        time.sleep(intervalo)
    raise TimeoutError("Condição não satisfeita dentro do tempo limite.")
