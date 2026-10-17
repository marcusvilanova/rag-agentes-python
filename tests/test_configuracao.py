from __future__ import annotations

import pytest

from rag_agentes.configuracao import Configuracao


@pytest.mark.parametrize("valor", [None, ""])
def test_chave_google_ausente_ou_vazia_gera_erro_explicativo(valor: str | None) -> None:
    configuracao = Configuracao(google_api_key=valor, _env_file=None)  # type: ignore[arg-type,call-arg]

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        configuracao.chave_google()


def test_uris_nao_aparecem_na_representacao() -> None:
    configuracao = Configuracao(mongodb_uri="mongodb+srv://usuario:senha@cluster", _env_file=None)  # type: ignore[arg-type,call-arg]

    assert "senha" not in repr(configuracao)
    assert configuracao.mongodb_uri.get_secret_value().endswith("@cluster")
