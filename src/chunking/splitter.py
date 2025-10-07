from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitter(chunk_size: int = 900, overlap: int = 100):
    '''
    Возвращает готовый сплитер из langchain_splitter
    -chunk_size - размер чанка в символах
    -overlap - перехлест символов между соседними чанками
    '''
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""] # по возможности рвем не в середине слов
    )