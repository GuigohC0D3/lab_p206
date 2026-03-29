# Laboratorio 6 - BPE e WordPiece

Este projeto implementa o calculo de frequencias do BPE, executa o loop principal de fusao por 5 iteracoes e tenta realizar a tokenizacao WordPiece com o modelo `bert-base-multilingual-cased`.

No WordPiece, o prefixo `##` indica que o token e uma continuacao da mesma palavra, e nao o comeco de uma nova palavra. Assim, um token como `##mente` representa uma subparte que deve ser anexada ao fragmento anterior. Esse mecanismo evita que o modelo falhe diante de palavras desconhecidas, porque ele nao depende de armazenar cada palavra completa no vocabulario. Em vez disso, ele pode decompor termos raros ou ineditos em subpalavras conhecidas, preservando parte do significado morfologico e reduzindo o problema de vocabulario fora do conjunto conhecido.

## Arquivo do projeto

- `wordpiece.py`: script principal com a implementacao das etapas do laboratorio.

## Como rodar

1. Abra o terminal na pasta do projeto:

```powershell
cd C:\Users\guilh\Documents\Projetos\lab_p206\lab_06
```

2. Ative o ambiente virtual existente:

```powershell
..\venv\Scripts\Activate.ps1
```

3. Execute o script principal:

```powershell
python .\wordpiece.py
```

## Execucao sem ativar o ambiente virtual

Se preferir, rode diretamente com o Python do ambiente virtual:

```powershell
..\venv\Scripts\python.exe .\wordpiece.py
```

## Dependencias

O script usa `transformers` para a etapa de WordPiece. Em alguns ambientes, tambem sera necessario ter `protobuf` instalado.

Se precisar instalar manualmente:

```powershell
..\venv\Scripts\python.exe -m pip install transformers protobuf
```

## Observacao importante

As etapas de BPE executam normalmente no projeto atual. A etapa de WordPiece depende de acesso as dependencias e, em alguns casos, de conexao com a internet para baixar o tokenizer do Hugging Face.
