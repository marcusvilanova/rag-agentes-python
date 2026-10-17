from __future__ import annotations

from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def pasta_corpus() -> Path:
    return RAIZ / "dados" / "corpus"


@pytest.fixture(scope="session")
def arquivo_de_avaliacao() -> Path:
    return RAIZ / "avaliacao" / "perguntas_v1.jsonl"
