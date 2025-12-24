#!/usr/bin/env python3
"""
Простой консольный интерфейс для RAG-бота
"""

from rag_bot import RAGBot

def main():
    print("RAG Bot запущен. Введите 'quit' для выхода.")
    
    # Инициализируем бота
    try:
        bot = RAGBot()
        print("Бот готов к работе.")
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        return
    
    # Основной цикл
    while True:
        try:
            user_input = input("\nВопрос: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("До свидания!")
                break
            
            if not user_input:
                continue
            
            # Получаем ответ от бота
            result = bot.ask(user_input)
            print(f"Ответ: {result['answer']}")
            
        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()
