"""Cache de respostas no Redis.

É um cache exato: a chave é o hash da pergunta normalizada (caixa, espaços e
pontuação final) somado a uma assinatura da configuração que influencia a
resposta (modelos, k, limiar). Mudar qualquer um desses parâmetros muda a
chave, então respostas de configurações antigas nunca são servidas.

Só respostas aprovadas pelo verificador entram no cache. Se o Redis estiver
fora do ar, o serviço continua respondendo sem cache.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata

import redis
from pydantic import ValidationError

from rag_agentes.modelos import RespostaDaPergunta

PREFIXO = "rag:resposta:"
logger = logging.getLogger(__name__)


def normalizar_pergunta(pergunta: str) -> str:
    texto = unicodedata.normalize("NFKC", pergunta).casefold()
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto.rstrip("?!. ")


class CacheDeRespostas:
    def __init__(self, cliente: redis.Redis, ttl_segundos: int, assinatura: str) -> None:
        self._cliente = cliente
        self._ttl = ttl_segundos
        self._assinatura = assinatura

    def chave(self, pergunta: str) -> str:
        conteudo = f"{self._assinatura}\n{normalizar_pergunta(pergunta)}"
        return PREFIXO + hashlib.sha256(conteudo.encode("utf-8")).hexdigest()

    def obter(self, pergunta: str) -> RespostaDaPergunta | None:
        try:
            bruto = self._cliente.get(self.chave(pergunta))
        except redis.RedisError:
            logger.warning("Redis indisponível; seguindo sem cache.", exc_info=True)
            return None
        if bruto is None:
            return None
        try:
            return RespostaDaPergunta.model_validate_json(bruto)
        except ValidationError:
            # Entrada gravada com outro formato de resposta: trata como ausente.
            logger.warning("Entrada de cache em formato antigo ignorada.")
            return None

    def gravar(self, resposta: RespostaDaPergunta) -> None:
        try:
            self._cliente.set(self.chave(resposta.pergunta), resposta.model_dump_json(), ex=self._ttl or None)
        except redis.RedisError:
            logger.warning("Redis indisponível; resposta não armazenada.", exc_info=True)

    def limpar(self) -> int:
        """Remove todas as respostas em cache; usado após reindexar o corpus."""
        try:
            chaves = list(self._cliente.scan_iter(match=f"{PREFIXO}*"))
            return int(self._cliente.delete(*chaves)) if chaves else 0
        except redis.RedisError:
            logger.warning("Redis indisponível; cache não foi limpo.", exc_info=True)
            return 0
