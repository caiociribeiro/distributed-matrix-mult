# Multiplicação de Matrizes Distribuída

Projeto desenvolvido para comparar diferentes abordagens de multiplicação de matrizes utilizando:

- execução serial;
- paralelismo com threads;
- paralelismo com processos;
- computação distribuída utilizando sockets TCP.

Na abordagem distribuída, múltiplos workers são iniciados localmente em portas diferentes do `localhost`, simulando nós distribuídos responsáveis por processar partes da matriz em paralelo.

O programa:

- gera matrizes aleatórias;
- executa os diferentes métodos de multiplicação;
- mede tempo de execução e speedup;
- salva os resultados em CSV;
- gera gráficos comparativos.

---

# Requisitos

- Python 3.10+
- Bibliotecas:
  - numpy
  - matplotlib

## Ambiente Virtual (Opcional)

Recomendado para evitar instalar bibliotecas globalmente:

```bash
python -m venv env
```

Ativar no Windows:

```bash
.\env\Scripts\activate
```

Ativar no Linux/macOS:

```bash
source ./env/bin/activate
```

Instalação:

```bash
pip install numpy matplotlib
```

---

# Estrutura do Arquivo de Testes

O programa recebe um arquivo `.txt` contendo os testes.

Formato:

```text
linha 1 -> quantidade de testes
linha 2+ -> dimensões das matrizes no formato NxM MxP
```

Exemplo:

```text
5
6x6 6x6
12x12 12x12
48x48 48x48
96x96 96x96
192x192 192x192
```

Cada linha representa:

```text
matriz A = NxM
matriz B = MxP
```

onde:

- o número de colunas da matriz A deve ser igual ao número de linhas da matriz B.

---

# Execução

Execute:

```bash
python main.py tests/test.txt
```

Durante a execução, o programa:

1. inicia os workers distribuídos;
2. gera matrizes aleatórias;
3. executa:
   - serial;
   - threads;
   - processos;
   - distribuída serial;
   - distribuída paralela;
4. mede os tempos de execução;
5. calcula speedup;
6. gera gráficos e arquivos CSV automaticamente.

---

# Workers Distribuídos

Os workers distribuídos:

- são executados localmente;
- utilizam sockets TCP;
- escutam em portas diferentes do `localhost`;
- recebem chunks das matrizes;
- processam os dados;
- retornam os resultados ao processo principal.

A implementação distribuída paralela também utiliza multiprocessing interno dentro de cada worker.

---

# Resultados Gerados

Após a execução, o programa cria automaticamente uma pasta em:

```text
results/
```

contendo:

- resultados em CSV;
- matrizes utilizadas nos testes;
- matrizes resultantes;
- gráficos de tempo;
- gráficos de speedup.
