#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import find_spec
import re
import subprocess
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Tuple


VOCAB_INICIAL: Dict[str, int] = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
}

FRASE_TESTE = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
MODELO_WORDPIECE = "bert-base-multilingual-cased"


def print_section(titulo: str) -> None:
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def get_stats(vocab: Dict[str, int]) -> DefaultDict[Tuple[str, str], int]:
    pairs: DefaultDict[Tuple[str, str], int] = defaultdict(int)

    for word, freq in vocab.items():
        symbols = word.split()
        for left, right in zip(symbols, symbols[1:]):
            pairs[(left, right)] += freq

    return pairs


def merge_vocab(pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
    pattern = re.escape(" ".join(pair))
    regex = re.compile(r"(?<!\S)" + pattern + r"(?!\S)")
    merged = "".join(pair)

    return {regex.sub(merged, word): freq for word, freq in vocab.items()}


def formatar_vocab(vocab: Dict[str, int]) -> Iterable[str]:
    for token, contagem in vocab.items():
        yield f"  {token!r}: {contagem}"


def modulo_disponivel(modulo: str) -> bool:
    try:
        return find_spec(modulo) is not None
    except ModuleNotFoundError:
        return False


def resumir_erro_pip(saida: str) -> str:
    linhas = [linha.strip() for linha in saida.splitlines() if linha.strip()]
    for prefixo in ("ERROR:", "Could not", "No matching", "Failed to establish"):
        for linha in linhas:
            if linha.startswith(prefixo):
                return linha
    return linhas[-1] if linhas else "falha desconhecida do pip"


def treinar_bpe(vocab: Dict[str, int], iteracoes: int = 5) -> Dict[str, int]:
    vocab_atual = vocab.copy()

    print_section("TAREFA 1 - Validação do motor de frequências")
    stats_iniciais = get_stats(vocab_atual)
    if not stats_iniciais:
        raise ValueError("O vocabulário inicial não gerou pares para análise.")

    melhor_par_inicial = max(stats_iniciais, key=stats_iniciais.get)
    print(f"Par mais frequente no início: {melhor_par_inicial} -> {stats_iniciais[melhor_par_inicial]}")
    print(f"Validação pedida: ('e', 's') -> {stats_iniciais.get(('e', 's'), 0)}")

    print_section(f"TAREFA 2 - Loop principal de fusão ({iteracoes} iterações)")

    for rodada in range(1, iteracoes + 1):
        stats = get_stats(vocab_atual)
        if not stats:
            break

        melhor_par = max(stats, key=stats.get)
        freq = stats[melhor_par]
        vocab_atual = merge_vocab(melhor_par, vocab_atual)

        print(f"\nIteração {rodada}")
        print(f"Par mais frequente fundido: {melhor_par} -> {freq}")
        print("Estado do vocab após a fusão:")
        for linha in formatar_vocab(vocab_atual):
            print(linha)

    return vocab_atual


def garantir_pacotes() -> bool:
    dependencias = {
        "transformers": "transformers",
        "google.protobuf": "protobuf",
    }
    faltantes = [pacote for modulo, pacote in dependencias.items() if not modulo_disponivel(modulo)]

    if not faltantes:
        return True

    print(f"\n[INFO] Dependências ausentes: {', '.join(faltantes)}. Tentando instalar...")
    try:
        resultado = subprocess.run(
            [sys.executable, "-m", "pip", "install", *faltantes],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        print(f"[AVISO] Não foi possível instalar as dependências: {exc}")
        return False

    if resultado.returncode != 0:
        detalhe_erro = resumir_erro_pip(resultado.stdout or resultado.stderr)
        detalhe = f": {detalhe_erro}" if detalhe_erro else "."
        print(f"[AVISO] Não foi possível instalar as dependências{detalhe}")
        return False

    for modulo in dependencias:
        if not modulo_disponivel(modulo):
            print(f"[AVISO] O módulo {modulo} continua indisponível após a instalação.")
            return False

    return True


def carregar_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODELO_WORDPIECE, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(MODELO_WORDPIECE)


def executar_wordpiece() -> None:
    print_section("TAREFA 3 - WordPiece com Hugging Face")

    if not garantir_pacotes():
        print("Não foi possível executar a tokenização WordPiece neste ambiente.")
        print("Para rodar localmente, use: pip install transformers protobuf")
        return

    try:
        tokenizer = carregar_tokenizer()
        tokens = tokenizer.tokenize(FRASE_TESTE)
    except Exception as exc:
        print(f"[AVISO] Falha ao carregar o modelo/tokenizer {MODELO_WORDPIECE}: {exc}")
        print("Se estiver sem internet, execute novamente em um ambiente conectado.")
        return

    print(f"Frase de teste: {FRASE_TESTE}")
    print("Tokens gerados:")
    print(tokens)


def main() -> None:
    vocab_final = treinar_bpe(VOCAB_INICIAL, iteracoes=5)

    print_section("Validação final esperada no PDF")
    print('Após 5 iterações, é possível observar a formação de tokens morfológicos como "est</w>".')
    print("Vocabulário final:")
    for linha in formatar_vocab(vocab_final):
        print(linha)

    executar_wordpiece()


if __name__ == "__main__":
    main()
