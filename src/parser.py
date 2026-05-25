from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .models import MatrixShape, MatrixTest


TEST_LINE_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s+(\d+)\s*x\s*(\d+)\s*$", re.IGNORECASE)


def parse_test_file(path: str | Path) -> List[MatrixTest]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Arquivo de testes vazio.")

    try:
        count = int(lines[0])
    except ValueError as exc:
        raise ValueError("A primeira linha deve conter o número de testes.") from exc

    if len(lines) - 1 < count:
        raise ValueError(f"Esperado {count} testes, mas encontrei apenas {len(lines) - 1}.")

    tests: List[MatrixTest] = []
    for i in range(count):
        line = lines[i + 1]
        match = TEST_LINE_RE.match(line)
        if not match:
            raise ValueError(
                f"Linha inválida na posição {i + 2}: '{line}'. "
                "Use o formato AxB CxD, por exemplo: 3x5 5x2"
            )
        a_rows, a_cols, b_rows, b_cols = map(int, match.groups())
        test = MatrixTest(a=MatrixShape(a_rows, a_cols), b=MatrixShape(b_rows, b_cols))
        if not test.valid:
            raise ValueError(
                f"Teste inválido na linha {i + 2}: colunas de A ({a_cols}) "
                f"devem ser iguais às linhas de B ({b_rows})."
            )
        tests.append(test)

    return tests
