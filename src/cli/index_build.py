import os, glob, json, typer
from  typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.chunking.splitter import get_splitter
from src.embeddings.embedding import E5Embeddings

app = typer.Typer()

def load_json(folder: str) -> List[Dict]:
    paths = glob.glob(os.path.join(folder, '*.jsonl'))
    records = []

    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
    return records

@app.command()
def build(
    data_dir: str='data/processed',
    persist_dir: str='vectorstore/chroma',
    collection_name:str='support',
    chunk_size:int=900,
    chunk_overlap:int=100,
    model_name:str='intfloat/multilingual-e5-base',
):
    '''
    Читает json
    режет на чанки через RecursiveCharacterTextSplitter
    создаем документы
    Эмбединг E5
    Сохраняет Chroma
    '''
    typer.secho('Шаг 1/5: читаю данные...', fg=typer.colors.CYAN)
    records = load_json(data_dir)
    typer.secho(f'Найдено записей {len(records)}')

    typer.secho('Шаг 2/5: чанкование...', fg=typer.colors.CYAN)
    splitter = get_splitter(chunk_size=chunk_size, overlap=chunk_overlap)

    docs: List[Document] = []
    for r in records:
        text = r.get('text', '') or ''
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    'class': r.get('class'),
                    'source': r.get('source'),
                    'parent_id': r.get('id')
                },
            )
        )

    typer.echo(f'Готово документов {len(docs)}')


    typer.secho('Шаг 3/5: чанкование документов', fg=typer.colors.CYAN)
    chunked_docs: List[Document] = splitter.split_documents(docs)
    typer.secho(f'Получено чанков {len(chunked_docs)}')

    typer.secho('Шаг 4/5: создание эмбеддера...',fg=typer.colors.CYAN)
    embedder = E5Embeddings(model_name=model_name)

    typer.secho('Шаг 5/5: Создание Chroma...')

    db = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedder,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    typer.secho(
        f"Готово! {len(chunked_docs)} чанков в {persist_dir}/{collection_name}", 
        fg=typer.colors.GREEN,
        bold=True)

if __name__ == '__main__':
    app()