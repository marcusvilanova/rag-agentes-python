from __future__ import annotations

from typing import Any

import fakeredis
import pytest
import redis

from rag_agentes.cache import PREFIXO, CacheDeRespostas, normalizar_pergunta
from rag_agentes.modelos import RespostaDaPergunta
from rag_agentes.observabilidade import Rastreamento
from rag_agentes.servico import ServicoDePerguntas


class GrafoFalso:
    """Devolve um estado final fixo e conta quantas vezes foi executado."""

    def __init__(self, situacao: str = "respondida") -> None:
        self.situacao = situacao
        self.execucoes = 0

    def invoke(self, entrada: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.execucoes += 1
        return {"pergunta": entrada["pergunta"], "situacao": self.situacao, "resposta": "texto", "trechos": []}


class RedisForaDoAr:
    def get(self, *_: Any) -> None:
        raise redis.ConnectionError("recusada")

    def set(self, *_: Any, **__: Any) -> None:
        raise redis.ConnectionError("recusada")


@pytest.fixture
def cliente_redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


def _resposta(pergunta: str = "Quando posso tratar dados?") -> RespostaDaPergunta:
    return RespostaDaPergunta(pergunta=pergunta, situacao="respondida", resposta="r [1]", fontes=[], tentativas=1)


@pytest.mark.parametrize(
    "variante",
    ["Quando posso tratar dados?", "  quando   POSSO tratar dados ", "Quando posso tratar dados?!"],
)
def test_normalizacao_ignora_caixa_espacos_e_pontuacao_final(variante: str) -> None:
    assert normalizar_pergunta(variante) == "quando posso tratar dados"


def test_cache_devolve_o_que_foi_gravado_para_pergunta_equivalente(cliente_redis: fakeredis.FakeRedis) -> None:
    cache = CacheDeRespostas(cliente_redis, ttl_segundos=60, assinatura="v1")
    cache.gravar(_resposta())

    assert cache.obter("quando posso tratar dados") == _resposta()
    assert 0 < cliente_redis.ttl(cache.chave("quando posso tratar dados")) <= 60


def test_assinatura_diferente_nao_reaproveita_respostas(cliente_redis: fakeredis.FakeRedis) -> None:
    CacheDeRespostas(cliente_redis, 60, assinatura="modelo-a").gravar(_resposta())

    assert CacheDeRespostas(cliente_redis, 60, assinatura="modelo-b").obter("Quando posso tratar dados?") is None


def test_limpar_remove_apenas_as_respostas(cliente_redis: fakeredis.FakeRedis) -> None:
    cache = CacheDeRespostas(cliente_redis, 60, assinatura="v1")
    cache.gravar(_resposta("a"))
    cache.gravar(_resposta("b"))
    cliente_redis.set("outra:chave", "1")

    assert cache.limpar() == 2
    assert cliente_redis.keys(f"{PREFIXO}*") == []
    assert cliente_redis.get("outra:chave") == "1"


def test_redis_fora_do_ar_nao_impede_a_resposta() -> None:
    cache = CacheDeRespostas(RedisForaDoAr(), 60, assinatura="v1")  # type: ignore[arg-type]
    grafo = GrafoFalso()
    servico = ServicoDePerguntas(grafo, Rastreamento(), cache)  # type: ignore[arg-type]

    resposta = servico.perguntar("Quando posso tratar dados?")

    assert resposta.situacao == "respondida"
    assert grafo.execucoes == 1


def test_servico_usa_o_cache_na_segunda_pergunta(cliente_redis: fakeredis.FakeRedis) -> None:
    grafo = GrafoFalso()
    servico = ServicoDePerguntas(grafo, Rastreamento(), CacheDeRespostas(cliente_redis, 60, "v1"))  # type: ignore[arg-type]

    primeira = servico.perguntar("Quando posso tratar dados?")
    segunda = servico.perguntar("quando posso tratar dados")

    assert grafo.execucoes == 1
    assert not primeira.em_cache
    assert segunda.em_cache
    assert segunda.resposta == primeira.resposta


@pytest.mark.parametrize("situacao", ["fora_do_escopo", "sem_fundamento"])
def test_respostas_nao_aprovadas_nao_entram_no_cache(cliente_redis: fakeredis.FakeRedis, situacao: str) -> None:
    grafo = GrafoFalso(situacao)
    servico = ServicoDePerguntas(grafo, Rastreamento(), CacheDeRespostas(cliente_redis, 60, "v1"))  # type: ignore[arg-type]

    servico.perguntar("Qual a capital da Austrália?")
    servico.perguntar("Qual a capital da Austrália?")

    assert grafo.execucoes == 2


def test_rastreamento_sem_chaves_fica_desligado() -> None:
    rastreamento = Rastreamento(public_key=None, secret_key=None)

    with rastreamento.observar("teste", entrada="x") as observacao:
        observacao.registrar_saida("y")

    assert not rastreamento.ativo
    assert rastreamento.callbacks() == []


def test_resposta_do_cache_traz_a_pergunta_recebida(cliente_redis: fakeredis.FakeRedis) -> None:
    servico = ServicoDePerguntas(GrafoFalso(), Rastreamento(), CacheDeRespostas(cliente_redis, 60, "v1"))  # type: ignore[arg-type]

    servico.perguntar("Quando posso tratar dados?")
    segunda = servico.perguntar("QUANDO posso tratar dados")

    assert segunda.em_cache
    assert segunda.pergunta == "QUANDO posso tratar dados"


def test_entrada_de_cache_em_formato_antigo_e_tratada_como_ausente(cliente_redis: fakeredis.FakeRedis) -> None:
    cache = CacheDeRespostas(cliente_redis, 60, assinatura="v1")
    cliente_redis.set(cache.chave("p"), '{"formato": "antigo"}')

    assert cache.obter("p") is None


def test_limpar_com_redis_fora_do_ar_nao_interrompe() -> None:
    class RedisSemScan(RedisForaDoAr):
        def scan_iter(self, **_: Any) -> list[str]:
            raise redis.ConnectionError("recusada")

    assert CacheDeRespostas(RedisSemScan(), 60, "v1").limpar() == 0  # type: ignore[arg-type]


def test_rastreamento_com_chaves_registra_o_grafo_sem_alterar_o_resultado(monkeypatch: pytest.MonkeyPatch) -> None:
    # Com as chaves definidas o caminho do Langfuse é exercitado; o envio fica desligado
    # para o teste não depender de rede.
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    rastreamento = Rastreamento(public_key="pk-lf-teste", secret_key="sk-lf-teste", base_url="http://127.0.0.1:9")
    grafo = GrafoFalso()

    resposta = ServicoDePerguntas(grafo, rastreamento).perguntar("Quando posso tratar dados?")  # type: ignore[arg-type]
    rastreamento.descarregar()

    assert rastreamento.ativo
    assert [type(c).__name__ for c in rastreamento.callbacks()] == ["LangchainCallbackHandler"]
    assert resposta.situacao == "respondida"
