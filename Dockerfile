FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Dependências primeiro, em camada própria, para aproveitar o cache do build.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY dados ./dados
COPY avaliacao ./avaliacao
RUN uv sync --locked --no-dev

RUN useradd --create-home --uid 1000 aplicacao
USER aplicacao

EXPOSE 8000
CMD ["rag", "api", "--host", "0.0.0.0", "--porta", "8000"]
