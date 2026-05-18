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


@dataclass
class BenchmarkResult:
    test_index: int
    a_shape: str
    b_shape: str
    workers: int
    serial_ms: float
    distributed_ms: float
    speedup: float
    correct: bool
