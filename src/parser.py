from pathlib import Path

from .models import MatrixShape, MatrixTest


def parse_test_file(path):
    lines = Path(path).read_text().splitlines()

    count = int(lines[0])

    tests = []

    for i in range(1, count + 1):
        left, right = lines[i].split()

        a_rows, a_cols = map(int, left.split("x"))
        b_rows, b_cols = map(int, right.split("x"))

        tests.append(
            MatrixTest(
                a=MatrixShape(a_rows, a_cols),
                b=MatrixShape(b_rows, b_cols),
            )
        )

    return tests
