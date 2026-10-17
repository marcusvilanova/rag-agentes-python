"""Tipos de dados compartilhados entre ingestão, recuperação, agentes e API."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_PRIMEIRO_ARTIGO_CARDINAL = 10


class Trecho(BaseModel):
    """Unidade de indexação: um artigo de lei, ou parte dele quando é longo."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description='Identificador estável, ex.: "lgpd:art-7" ou "lgpd:art-7:parte-2".')
    documento: str = Field(description='Identificador do documento de origem, ex.: "lgpd".')
    titulo_documento: str
    artigo: str | None = Field(description='Número do artigo ("7", "55-A"); None no preâmbulo.')
    secao: str = Field(description="Caminho estrutural: título, capítulo e seção do artigo.")
    texto: str
    ordem: int = Field(description="Posição do trecho dentro do documento.")

    @property
    def referencia(self) -> str:
        """Forma legível de citar o trecho, ex.: "Lei nº 13.709/2018, art. 7º"."""
        titulo = re.sub(r"\s*\(.*\)$", "", self.titulo_documento)
        if self.artigo is None:
            return f"{titulo}, preâmbulo"
        return f"{titulo}, {formatar_artigo(self.artigo)}"


class TrechoRecuperado(BaseModel):
    model_config = ConfigDict(frozen=True)

    trecho: Trecho
    pontuacao: float = Field(description="Similaridade de cosseno normalizada para [0, 1].")


class FonteCitada(BaseModel):
    numero: int = Field(description="Número usado na citação [n] da resposta.")
    id: str
    referencia: str
    texto: str
    pontuacao: float


class RespostaDaPergunta(BaseModel):
    pergunta: str
    situacao: Literal["respondida", "fora_do_escopo", "sem_fundamento"] = Field(
        description="respondida: aprovada pelo verificador; fora_do_escopo: rejeitada pelo classificador; "
        "sem_fundamento: nenhum trecho relevante ou nenhuma resposta aprovada."
    )
    resposta: str
    fontes: list[FonteCitada] = Field(description="Trechos citados na resposta, na ordem das citações.")
    tentativas: int = Field(description="Quantas vezes o redator escreveu a resposta.")
    motivo_da_verificacao: str | None = None
    em_cache: bool = False


def formatar_artigo(numero: str) -> str:
    """Aplica a convenção da técnica legislativa: ordinal até o 9, cardinal a partir do 10."""
    base, _, sufixo = numero.partition("-")
    ordinal = f"{base}º" if int(base) < _PRIMEIRO_ARTIGO_CARDINAL else base
    return f"art. {ordinal}-{sufixo}" if sufixo else f"art. {ordinal}"
