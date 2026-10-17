"""Rastreamento no Langfuse.

As chamadas de LLM feitas dentro do grafo são registradas pelo CallbackHandler
do Langfuse para LangChain, que já captura modelo, prompt, resposta, tokens e
latência de cada nó. A vetorização da consulta não passa pelo LangChain e é
registrada à parte, como observação do tipo "embedding".

Sem LANGFUSE_PUBLIC_KEY e LANGFUSE_SECRET_KEY o rastreamento fica desligado e
todas as operações abaixo viram no-op.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

TipoDeObservacao = Literal["span", "retriever", "embedding"]


class Observacao:
    def __init__(self, alvo: Any = None) -> None:
        self._alvo = alvo

    def registrar_saida(self, saida: Any) -> None:
        if self._alvo is not None:
            self._alvo.update(output=saida)


class Rastreamento:
    def __init__(self, public_key: str | None = None, secret_key: str | None = None, base_url: str = "") -> None:
        self._public_key = public_key
        self._cliente: Langfuse | None = None
        if public_key and secret_key:
            self._cliente = Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url or None)

    @property
    def ativo(self) -> bool:
        return self._cliente is not None

    def callbacks(self) -> list[BaseCallbackHandler]:
        """Handlers a passar no `config` do grafo; um novo por execução."""
        if self._cliente is None:
            return []
        return [CallbackHandler(public_key=self._public_key)]

    @contextmanager
    def observar(
        self,
        nome: str,
        tipo: TipoDeObservacao = "span",
        entrada: Any = None,
        modelo: str | None = None,
    ) -> Iterator[Observacao]:
        if self._cliente is None:
            yield Observacao()
            return
        atributos: dict[str, Any] = {"name": nome, "as_type": tipo, "input": entrada}
        if modelo is not None:
            atributos["model"] = modelo
        cliente: Any = self._cliente
        with cliente.start_as_current_observation(**atributos) as alvo:
            yield Observacao(alvo)

    def descarregar(self) -> None:
        """Envia o que estiver pendente; necessário em processos curtos como a CLI."""
        if self._cliente is not None:
            self._cliente.flush()
