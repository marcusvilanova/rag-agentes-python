"""Fragmentação de textos legais por artigo.

Em lei, o artigo é a menor unidade que se sustenta sozinha: parágrafos e
incisos só fazem sentido lidos junto do caput. Por isso o corte segue a
estrutura do texto, e não um número fixo de caracteres. Artigos longos são
divididos entre incisos/parágrafos, repetindo o caput em cada parte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_agentes.modelos import Trecho

TAMANHO_MAXIMO_PADRAO = 1500
_TAMANHO_MAXIMO_NOME_DE_SECAO = 200

# O sufixo de artigo inserido vem colado ao número ("Art. 4º-A", "Art. 55-B"); com
# espaço ("Art. 1º - O ...") o hífen é só pontuação do caput.
_ARTIGO = re.compile(r"^Art\.\s*(\d+)\s*º?(?:-([A-Z]))?(?![\d])")
_TITULO_ESTRUTURAL = re.compile(
    r"^(LIVRO|TÍTULO|CAPÍTULO|SEÇÃO|Seção|SUBSEÇÃO|Subseção)\s+([IVXLC]+(?:-[A-Z])?|\d+|ÚNIC[OA]|Únic[oa])\b"
)
_NIVEIS = ("LIVRO", "TÍTULO", "CAPÍTULO", "SEÇÃO", "SUBSEÇÃO")
_ASSINATURA = re.compile(r"^(Brasília|Rio de Janeiro),\s+\d")
_ANOTACAO = re.compile(r"^\([^()]*\)\.?$")
_PARENTETICO = re.compile(r"\([^()]*\)")
_MARCA_DE_VETO_OU_REVOGACAO = re.compile(r"\(\s*(vetad[oa]|revogad[oa])", re.IGNORECASE)
_SO_PONTUACAO = re.compile(r"^[\s.;,:-]*$")


@dataclass
class _Bloco:
    artigo: str | None
    secao: str
    linhas: list[str] = field(default_factory=list)


def fragmentar(
    documento: str,
    titulo_documento: str,
    texto: str,
    tamanho_maximo: int = TAMANHO_MAXIMO_PADRAO,
) -> list[Trecho]:
    """Divide um ato normativo em trechos: preâmbulo e um ou mais por artigo."""
    blocos = _separar_em_blocos(texto.splitlines())
    trechos: list[Trecho] = []
    ids_usados: set[str] = set()

    for bloco in blocos:
        if not bloco.linhas or _sem_conteudo(bloco):
            continue
        id_base = f"{documento}:art-{bloco.artigo}" if bloco.artigo else f"{documento}:preambulo"
        id_base = _tornar_unico(id_base, ids_usados)
        partes = _dividir(bloco.linhas, tamanho_maximo)
        for indice, linhas in enumerate(partes, start=1):
            trechos.append(
                Trecho(
                    id=id_base if len(partes) == 1 else f"{id_base}:parte-{indice}",
                    documento=documento,
                    titulo_documento=titulo_documento,
                    artigo=bloco.artigo,
                    secao=bloco.secao,
                    texto="\n".join(linhas),
                    ordem=len(trechos),
                )
            )
    return trechos


def _separar_em_blocos(linhas: list[str]) -> list[_Bloco]:
    hierarquia: dict[str, str] = {}
    blocos = [_Bloco(artigo=None, secao="")]
    indice = 0
    while indice < len(linhas):
        linha = linhas[indice].strip()
        indice += 1
        if not linha:
            continue
        if _ASSINATURA.match(linha):
            break

        titulo = _TITULO_ESTRUTURAL.match(linha)
        if titulo:
            # O nome da divisão costuma vir na linha seguinte, às vezes depois de uma
            # anotação como "(Incluído pela Lei ...)". Anotações do título não pertencem
            # ao artigo anterior, então também são consumidas aqui.
            rotulo = linha
            if linha == titulo.group(0):
                indice = _pular_anotacoes(linhas, indice)
                if indice < len(linhas) and _eh_nome_de_secao(linhas[indice]):
                    rotulo = f"{linha} - {linhas[indice].strip()}"
                    indice += 1
            indice = _pular_anotacoes(linhas, indice)
            hierarquia = _atualizar_hierarquia(hierarquia, titulo.group(1).upper(), rotulo)
            continue

        artigo = _ARTIGO.match(linha)
        if artigo:
            numero = artigo.group(1) + (f"-{artigo.group(2)}" if artigo.group(2) else "")
            blocos.append(_Bloco(artigo=numero, secao=" > ".join(hierarquia.values())))
        blocos[-1].linhas.append(linha)
    return blocos


def _pular_anotacoes(linhas: list[str], indice: int) -> int:
    while indice < len(linhas) and _ANOTACAO.match(linhas[indice].strip()):
        indice += 1
    return indice


def _eh_nome_de_secao(linha: str) -> bool:
    linha = linha.strip()
    return (
        bool(linha)
        and len(linha) <= _TAMANHO_MAXIMO_NOME_DE_SECAO
        and not _ARTIGO.match(linha)
        and not _TITULO_ESTRUTURAL.match(linha)
        and not _ANOTACAO.match(linha)
    )


def _atualizar_hierarquia(hierarquia: dict[str, str], nivel: str, rotulo: str) -> dict[str, str]:
    # Um novo capítulo encerra as seções do capítulo anterior, e assim por diante.
    superiores = _NIVEIS[: _NIVEIS.index(nivel)]
    return {**{n: r for n, r in hierarquia.items() if n in superiores}, nivel: rotulo}


def _sem_conteudo(bloco: _Bloco) -> bool:
    """Artigos que só dizem "(VETADO)" ou "(Revogado)" não têm o que recuperar.

    Cobre também as formas com anotação ("(VETADO) (Incluído pela Lei ...)") e as
    de várias linhas, desde que todas sejam apenas parentéticas.
    """
    if bloco.artigo is None:
        return False
    linhas = [_ARTIGO.sub("", bloco.linhas[0]), *bloco.linhas[1:]]
    tem_marca = any(_MARCA_DE_VETO_OU_REVOGACAO.search(linha) for linha in linhas)
    return tem_marca and all(_SO_PONTUACAO.match(_PARENTETICO.sub("", linha)) for linha in linhas)


def _tornar_unico(id_base: str, ids_usados: set[str]) -> str:
    candidato, repeticao = id_base, 1
    while candidato in ids_usados:
        repeticao += 1
        candidato = f"{id_base}~{repeticao}"
    ids_usados.add(candidato)
    return candidato


def _dividir(linhas: list[str], tamanho_maximo: int) -> list[list[str]]:
    """Agrupa as linhas em partes de até `tamanho_maximo` caracteres.

    O caput (primeira linha) abre todas as partes. Uma linha isolada maior que o
    limite não é cortada no meio, para não separar uma frase de lei.
    """
    if _comprimento(linhas) <= tamanho_maximo:
        return [linhas]

    caput, restantes = linhas[0], linhas[1:]
    partes: list[list[str]] = []
    atual = [caput]
    for linha in restantes:
        if len(atual) > 1 and _comprimento([*atual, linha]) > tamanho_maximo:
            partes.append(atual)
            atual = [caput]
        atual.append(linha)
    partes.append(atual)
    return partes


def _comprimento(linhas: list[str]) -> int:
    return sum(len(linha) for linha in linhas) + max(len(linhas) - 1, 0)
