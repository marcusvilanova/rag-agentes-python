from __future__ import annotations

from apoio import ModeloRoteirizado, recuperado

from rag_agentes.agentes.verificador import (
    Julgamento,
    checar_citacoes,
    extrair_citacoes,
    verificar,
)

TRECHOS = [
    recuperado(
        "lgpd:art-7", "Art. 7º O tratamento de dados pessoais somente poderá ser realizado mediante consentimento."
    ),
    recuperado("lgpd:art-48", "Art. 48. O controlador deverá comunicar à autoridade nacional incidente de segurança."),
]


def test_extrair_citacoes_devolve_numeros_unicos_e_ordenados() -> None:
    assert extrair_citacoes("Primeira [2]. Segunda [1][2].") == [1, 2]


def test_resposta_sem_citacao_e_reprovada_sem_chamar_o_llm() -> None:
    modelo = ModeloRoteirizado()

    resultado = verificar(modelo, "pergunta", "O tratamento depende de consentimento.", TRECHOS)  # type: ignore[arg-type]

    assert not resultado.aprovada
    assert "não cita" in resultado.motivo
    assert modelo.chamadas == []


def test_citacao_de_trecho_inexistente_e_reprovada_sem_chamar_o_llm() -> None:
    modelo = ModeloRoteirizado()

    resultado = verificar(modelo, "pergunta", "Depende de consentimento [1] e de aviso [3].", TRECHOS)  # type: ignore[arg-type]

    assert not resultado.aprovada
    assert "[3]" in resultado.motivo
    assert modelo.chamadas == []


def test_checar_citacoes_aceita_citacoes_validas() -> None:
    assert checar_citacoes("Afirmação [1]. Outra [2].", quantidade_de_trechos=2) is None


def test_resposta_sustentada_e_aprovada() -> None:
    modelo = ModeloRoteirizado(
        estruturadas=[
            Julgamento(afirmacoes_sem_suporte=[], justificativa="Tudo consta do trecho 1.", fundamentada=True)
        ]
    )

    resultado = verificar(modelo, "Quando posso tratar dados?", "Mediante consentimento [1].", TRECHOS)  # type: ignore[arg-type]

    assert resultado.aprovada
    assert modelo.tipos_chamados() == ["Julgamento"]
    prompt = modelo.chamadas[0][1][-1].content
    assert "[1] Lei nº 13.709/2018, art. 1º" in prompt
    assert "Resposta a auditar:\nMediante consentimento [1]." in prompt


def test_resposta_com_afirmacao_sem_suporte_e_reprovada() -> None:
    modelo = ModeloRoteirizado(
        estruturadas=[
            Julgamento(
                afirmacoes_sem_suporte=["O prazo de comunicação é de 48 horas."],
                justificativa="O trecho 2 não fixa prazo.",
                fundamentada=False,
            )
        ]
    )

    resultado = verificar(modelo, "pergunta", "Deve comunicar em 48 horas [2].", TRECHOS)  # type: ignore[arg-type]

    assert not resultado.aprovada
    assert resultado.afirmacoes_sem_suporte == ["O prazo de comunicação é de 48 horas."]


def test_veredito_contraditorio_prevalece_a_reprovacao() -> None:
    modelo = ModeloRoteirizado(
        estruturadas=[
            Julgamento(afirmacoes_sem_suporte=["Afirmação inventada."], justificativa="ok", fundamentada=True)
        ]
    )

    resultado = verificar(modelo, "pergunta", "Afirmação inventada [1].", TRECHOS)  # type: ignore[arg-type]

    assert not resultado.aprovada
