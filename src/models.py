from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixShape:
    rows: int
    cols: int


@dataclass(frozen=True)
class MatrixTest:
    a: MatrixShape
    b: MatrixShape

    @property
    def valid(self) -> bool:
        return self.a.cols == self.b.rows


@dataclass(frozen=True)
class MatrixDimensions:
    rows_a: int
    cols_a: int
    rows_b: int
    cols_b: int

    @property
    def work_units(self) -> int:
        return self.rows_a * self.cols_a * self.cols_b

    @property
    def result_cells(self) -> int:
        return self.rows_a * self.cols_b


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
    def signature(self) -> str:
        return f"{self.rows_a}x{self.cols_a} · {self.rows_b}x{self.cols_b}"
