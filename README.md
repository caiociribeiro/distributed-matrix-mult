# Multiplicação de Matrizes Distribuída

## Execução local

```bash
python main.py tests/test_5.txt
```

## Execução com número específico de workers

```bash
python main.py tests/test_5.txt --workers 4
```

## Modo socket com workers remotos

Em cada máquina worker:

```bash
python main.py --worker --host 0.0.0.0 --port 5001
```

No cliente:

```bash
python main.py testes.txt --mode socket --hosts 192.168.0.10:5001,192.168.0.11:5001
```
