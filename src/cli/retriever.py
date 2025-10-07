import typer
from langchain_chroma  import Chroma
from src.embeddings.embedding import E5Embeddings

app = typer.Typer()

@app.command(name='q')
def q(
    query:str,
    persist_dir:str='vectorstore/chroma',
    collection_name:str='support',
    k:int=5,
    model_name:str='intfloat/multilingual-e5-base'
):
    """
    Делает similarity_search по Chroma через LangChain.
    """

    embedder =E5Embeddings(model_name=model_name)

    db = Chroma(
        embedding_function=embedder,
        persist_directory=persist_dir,
        collection_name=collection_name
    )

    docs = db.similarity_search(query=query, k=k)

    typer.secho(f'\nТоп-{k} для {query}\n')

    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        parent = meta.get("parent_id")
        cls = meta.get("class")
        src = meta.get("source")
        preview = (d.page_content[:900] + "...") if len(d.page_content) > 300 else d.page_content
        typer.echo(f"[{i}] parent={parent}\nclass={cls} source={src}\n---\n{preview}\n")

if __name__ == "__main__":
    app()
