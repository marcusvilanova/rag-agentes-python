"""Coleta dos textos legais do Planalto e conversão de HTML para texto limpo.

As páginas de legislação compilada do Planalto marcam o texto revogado com
<strike>, <s>, <del> ou com `text-decoration: line-through`. Esse texto é
descartado: indexá-lo faria o sistema citar dispositivos que não vigoram mais.
"""

from __future__ import annotations

import re
import tomllib
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

_TAGS_DE_BLOCO = frozenset({"p", "br", "div", "tr", "table", "li", "center", "blockquote", "h1", "h2", "h3", "h4"})
_TAGS_SEMPRE_REVOGADAS = frozenset({"strike", "s", "del"})
_TAGS_COM_ESTILO = frozenset({"span", "font"})
_TAGS_IGNORADAS = frozenset({"script", "style", "head"})

_INICIO_DO_ATO = re.compile(r"^(LEI COMPLEMENTAR|LEI|DECRETO-LEI|DECRETO)\s+N", re.IGNORECASE)
_FIM_DO_ATO = "Este texto não substitui"
_TEMPO_LIMITE_SEGUNDOS = 30
# O servidor do Planalto recusa conexões cujo User-Agent não começa com "Mozilla";
# o formato "compatible" mantém a identificação do cliente.
_USER_AGENT = "Mozilla/5.0 (compatible; rag-agentes/0.1)"


@dataclass(frozen=True)
class Fonte:
    id: str
    titulo: str
    url: str


def carregar_fontes(caminho: Path) -> list[Fonte]:
    with caminho.open("rb") as arquivo:
        dados = tomllib.load(arquivo)
    return [Fonte(**item) for item in dados["fonte"]]


class _ExtratorDeTexto(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._partes: list[str] = []
        # Pilha de (tag, revoga) apenas para as tags que podem marcar revogação.
        self._pilha: list[tuple[str, bool]] = []
        self._ignoradas = 0
        self._em_sobrescrito = 0

    @property
    def _dentro_de_revogado(self) -> bool:
        return any(revoga for _, revoga in self._pilha)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _TAGS_SEMPRE_REVOGADAS:
            self._pilha.append((tag, True))
        elif tag in _TAGS_COM_ESTILO:
            estilo = (dict(attrs).get("style") or "").lower().replace(" ", "")
            self._pilha.append((tag, "line-through" in estilo))
        elif tag in _TAGS_IGNORADAS:
            self._ignoradas += 1
        elif tag == "sup":
            self._em_sobrescrito += 1
        if tag in _TAGS_DE_BLOCO:
            self._partes.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _TAGS_SEMPRE_REVOGADAS or tag in _TAGS_COM_ESTILO:
            for indice in range(len(self._pilha) - 1, -1, -1):
                if self._pilha[indice][0] == tag:
                    del self._pilha[indice]
                    break
        elif tag in _TAGS_IGNORADAS:
            self._ignoradas = max(0, self._ignoradas - 1)
        elif tag == "sup":
            self._em_sobrescrito = max(0, self._em_sobrescrito - 1)
        if tag in _TAGS_DE_BLOCO:
            self._partes.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignoradas or self._dentro_de_revogado:
            return
        # "1<sup>o</sup>" é a forma antiga de escrever o ordinal "1º".
        if self._em_sobrescrito and data.strip() in {"o", "°"}:
            data = "º"
        self._partes.append(re.sub(r"\s+", " ", data))

    def texto(self) -> str:
        linhas = (re.sub(r"\s+", " ", linha).strip() for linha in "".join(self._partes).split("\n"))
        return "\n".join(linha for linha in linhas if linha)


def html_para_texto(html: str) -> str:
    """Extrai o texto vigente do ato, do título até o aviso final do Planalto."""
    extrator = _ExtratorDeTexto()
    extrator.feed(html)
    extrator.close()
    linhas = extrator.texto().split("\n")

    inicio = next((i for i, linha in enumerate(linhas) if _INICIO_DO_ATO.match(linha)), 0)
    fim = next(
        (i for i, linha in enumerate(linhas) if linha.startswith(_FIM_DO_ATO)),
        len(linhas),
    )
    texto = "\n".join(linhas[inicio:fim])
    # O HTML de origem às vezes separa o ordinal ("Art. 1 º") ou parte o número
    # do artigo ("Art. 5 7."); as duas formas são reagrupadas.
    texto = re.sub(r"(\d) º", r"\1º", texto)
    return re.sub(r"^(Art\.\s*\d+) (\d+\.)", r"\1\2", texto, flags=re.MULTILINE)


def decodificar(conteudo: bytes) -> str:
    # As páginas antigas do Planalto não declaram charset e usam Windows-1252.
    try:
        return conteudo.decode("utf-8")
    except UnicodeDecodeError:
        return conteudo.decode("cp1252")


def baixar_html(url: str) -> str:
    if not url.startswith("https://"):
        raise ValueError(f"Apenas URLs https são aceitas: {url!r}")
    requisicao = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310 (esquema validado acima)
    with urllib.request.urlopen(requisicao, timeout=_TEMPO_LIMITE_SEGUNDOS) as resposta:  # noqa: S310
        return decodificar(resposta.read())


def gerar_corpus(arquivo_fontes: Path, pasta_saida: Path) -> list[Path]:
    """Baixa cada fonte e grava o texto limpo em `<pasta_saida>/<id>.txt`."""
    gerados = []
    for fonte in carregar_fontes(arquivo_fontes):
        texto = html_para_texto(baixar_html(fonte.url))
        destino = pasta_saida / f"{fonte.id}.txt"
        # LF fixo: o arquivo gerado é o mesmo em Windows e Linux.
        destino.write_text(texto + "\n", encoding="utf-8", newline="\n")
        gerados.append(destino)
    return gerados
