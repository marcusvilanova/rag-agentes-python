from __future__ import annotations

from pathlib import Path

import pytest

from rag_agentes.corpus import baixar_html, carregar_fontes, decodificar, html_para_texto

HTML = """<html><head><title>L99999</title></head><body>
<p>Presidência da República</p>
<p><a href="#">LEI N<sup>o</sup> 99.999, DE 1º DE JANEIRO DE 2030.</a></p>
<p>Art. 1<sup>o</sup> Texto vigente
   quebrado em várias linhas do HTML.</p>
<p><strike>Art. 2º Texto revogado por strike.</strike></p>
<p>Art. 2º Texto novo do artigo.</p>
<p><span style="text-decoration: line-through">§ 1º Parágrafo revogado por estilo.</span></p>
<p><span style="font-size:10pt">§ 2º Parágrafo vigente dentro de span.</span></p>
<p>Art. 5 7. (VETADO).</p>
<script>var x = 1;</script>
<p>Este texto não substitui o publicado no DOU de 2.1.2030.</p>
<p>Rodapé.</p>
</body></html>"""


def test_html_para_texto_mantem_so_o_texto_vigente_do_ato() -> None:
    assert html_para_texto(HTML).splitlines() == [
        "LEI Nº 99.999, DE 1º DE JANEIRO DE 2030.",
        "Art. 1º Texto vigente quebrado em várias linhas do HTML.",
        "Art. 2º Texto novo do artigo.",
        "§ 2º Parágrafo vigente dentro de span.",
        "Art. 57. (VETADO).",
    ]


def test_decodificar_aceita_utf8_e_windows_1252() -> None:
    assert decodificar("Não".encode()) == "Não"
    assert decodificar("Não".encode("cp1252")) == "Não"


def test_baixar_html_recusa_esquemas_diferentes_de_https() -> None:
    with pytest.raises(ValueError, match="https"):
        baixar_html("file:///etc/passwd")


def test_fontes_tem_ids_unicos_e_arquivos_gerados(pasta_corpus: Path) -> None:
    fontes = carregar_fontes(pasta_corpus / "fontes.toml")

    assert len({f.id for f in fontes}) == len(fontes)
    assert all(f.url.startswith("https://www.planalto.gov.br/") for f in fontes)
    assert all((pasta_corpus / f"{f.id}.txt").is_file() for f in fontes)
