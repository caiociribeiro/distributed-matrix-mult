from dataclasses import dataclass


@dataclass
class MatrixShape:
    rows: int
    cols: int


@dataclass
class MatrixTest:
    a: MatrixShape
    b: MatrixShape


@dataclass
class BenchmarkResult:
    test_index: int
    mode: str
    workers: int
    rows_a: int
    cols_a: int
    rows_b: int
    cols_b: int
    work_units: int
    elapsed_ms: float
    speedup: float
    correct: bool

    @property
    def signature(self):
        return f"{self.rows_a}x{self.cols_a} · {self.rows_b}x{self.cols_b}"
