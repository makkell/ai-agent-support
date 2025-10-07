from src.rag.pipeline import answer_rag

print(answer_rag("Не приходит письмо для сброса пароля", k=5))