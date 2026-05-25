# Multiplicação de Matrizes Distribuída

Execução local:
```bash
python main.py tests/test_5.txt
```

Workers distribuídos simulados:
```bash
python main.py tests/test_5.txt --distributed-workers 2,4,6
```

Worker manual:
```bash
python main.py --worker --host 0.0.0.0 --port 5001
```
