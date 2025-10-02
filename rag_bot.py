#!/usr/bin/env python3
"""
RAG-бот с техниками промптинга (Few-shot и Chain-of-Thought)
Использует FAISS индекс и OpenAI API для генерации ответов
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class RAGBot:
    """RAG-бот с техниками промптинга"""
    
    def __init__(self, 
                 index_path: str = "vector_index/faiss_index.index",
                 metadata_path: str = "vector_index/metadata.json",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Инициализация RAG-бота
        
        Args:
            index_path: Путь к FAISS индексу
            metadata_path: Путь к файлу метаданных
            embedding_model: Название модели эмбеддингов
        """
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Загружаем компоненты
        self._load_embedding_model()
        self._load_index()
        self._load_metadata()
        self._init_openai_client()
        
        # Настройки поиска
        self.k_results = 5  # Количество результатов поиска
        
        print("RAG-бот инициализирован")
        print(f"Загружено чанков: {len(self.metadata['chunks'])}")
        print(f"Модель эмбеддингов: {self.embedding_model}")
    
    def _load_embedding_model(self):
        """Загружает модель эмбеддингов"""
        print("Загружаем модель эмбеддингов...")
        self.embedding_model_instance = SentenceTransformer(self.embedding_model)
        print(f"Модель загружена: {self.embedding_model}")
    
    def _load_index(self):
        """Загружает FAISS индекс"""
        print("Загружаем FAISS индекс...")
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Индекс не найден: {self.index_path}")
        
        self.index = faiss.read_index(self.index_path)
        print(f"Индекс загружен: {self.index.ntotal} векторов")
    
    def _load_metadata(self):
        """Загружает метаданные"""
        print("Загружаем метаданные...")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Метаданные не найдены: {self.metadata_path}")
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"Метаданные загружены: {self.metadata['total_chunks']} чанков")
    
    def _init_openai_client(self):
        """Инициализирует клиент OpenAI"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("OPENAI_API_KEY не найден в переменных окружения")
            print("Бот будет работать в режиме поиска без генерации ответов")
            self.openai_client = None
        else:
            self.openai_client = OpenAI(api_key=api_key)
            print("OpenAI клиент инициализирован")
    
    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Поиск релевантных чанков по запросу
        
        Args:
            query: Поисковый запрос
            k: Количество результатов (по умолчанию self.k_results)
            
        Returns:
            Список найденных чанков с метаданными
        """
        if k is None:
            k = self.k_results
        
        # Создаем эмбеддинг для запроса
        query_embedding = self.embedding_model_instance.encode([query])
        
        # Ищем ближайшие векторы
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Формируем результаты
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            chunk_metadata = self.metadata['chunks'][idx]
            results.append({
                'chunk_id': chunk_metadata['chunk_id'],
                'source': chunk_metadata['source'],
                'title': chunk_metadata['title'],
                'content_preview': chunk_metadata['content_preview'],
                'distance': float(distance),
                'relevance_score': 1.0 / (1.0 + distance)  # Преобразуем расстояние в релевантность
            })
        
        return results
    
    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """
        Получает полное содержимое чанка по его ID
        
        Args:
            chunk_id: ID чанка
            
        Returns:
            Полное содержимое чанка или None
        """
        # Находим чанк в метаданных
        for chunk in self.metadata['chunks']:
            if chunk['chunk_id'] == chunk_id:
                # Читаем файл
                source_path = chunk['source']
                if os.path.exists(source_path):
                    with open(source_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Разбиваем на чанки (упрощенная версия)
                    # В реальности нужно использовать тот же splitter, что и при создании индекса
                    chunks = content.split('\n\n')
                    chunk_index = chunk['chunk_index']
                    
                    if chunk_index < len(chunks):
                        return chunks[chunk_index].strip()
        
        return None
    
    def create_few_shot_examples(self) -> str:
        """
        Создает примеры для Few-shot prompting
        
        Returns:
            Строка с примерами
        """
        examples = """
Примеры вопросов и ответов:

Q: Что такое Energo Mech?
A: Energo Mech - это энергетическое оружие, используемое Strazei. Это меч, который излучает энергетическое лезвие и может резать практически любой материал.

Q: Кто такой Dmitri Volkov?
A: Dmitri Volkov - это бывший Strazh, который перешел на темную сторону и стал Tenevoy Voin. Он был учеником Igor Sokolov, но предал Orden Strazei.

Q: Что такое Proekt Alpha?
A: Proekt Alpha - это огромная боевая станция, построенная Galakticheskaya Imperiya. Она способна уничтожать целые планеты одним выстрелом.

"""
        return examples
    
    def create_system_prompt(self) -> str:
        """
        Создает системный промпт с Chain-of-Thought инструкциями
        
        Returns:
            Системный промпт
        """
        system_prompt = """Ты - помощник по базе знаний Zvezdnye Vojny. Твоя задача - отвечать на вопросы, используя только информацию из предоставленных документов.

ВАЖНЫЕ ПРАВИЛА:
1. Всегда объясняй свои шаги рассуждения (Chain-of-Thought)
2. Используй только информацию из предоставленных документов
3. Если информации нет в документах, честно скажи "Я не знаю"
4. Никогда не выдумывай факты
5. Всегда указывай источник информации

СТРУКТУРА ОТВЕТА:
1. Сначала найди релевантную информацию в документах
2. Проанализируй найденную информацию
3. Сформулируй ответ на основе найденных данных
4. Укажи источник информации

ПРИМЕР РАССУЖДЕНИЯ:
"Давайте разберем этот вопрос пошагово:
1. Сначала я ищу информацию о [теме] в предоставленных документах
2. В документе [название] я нашел следующую информацию: [цитата]
3. На основе этой информации могу ответить: [ответ]
4. Источник: [название документа]"

Если в документах нет информации для ответа на вопрос, отвечай: "Я не знаю. В предоставленных документах нет информации по этому вопросу."
"""
        return system_prompt
    
    def generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Генерирует ответ с использованием LLM
        
        Args:
            query: Пользовательский запрос
            search_results: Результаты поиска
            
        Returns:
            Сгенерированный ответ
        """
        if not self.openai_client:
            return "❌ OpenAI API не настроен. Проверьте переменную OPENAI_API_KEY."
        
        # Формируем контекст из найденных чанков
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):  # Берем топ-3 результата
            content = self.get_chunk_content(result['chunk_id'])
            if content:
                context_parts.append(f"Документ {i} ({result['title']}):\n{content}\n")
        
        if not context_parts:
            return "Я не знаю. В базе знаний нет релевантной информации по этому вопросу."
        
        context = "\n".join(context_parts)
        
        # Создаем промпт
        system_prompt = self.create_system_prompt()
        few_shot_examples = self.create_few_shot_examples()
        
        user_prompt = f"""{few_shot_examples}

Контекст из базы знаний:
{context}

Вопрос: {query}

Ответ (обязательно объясни свои шаги):"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"❌ Ошибка при генерации ответа: {str(e)}"
    
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Основной метод для получения ответа на вопрос
        
        Args:
            query: Пользовательский вопрос
            
        Returns:
            Словарь с ответом и метаданными
        """
        print(f"Обрабатываем запрос: '{query}'")
        
        # 1. Поиск релевантных чанков
        search_results = self.search(query)
        
        if not search_results:
            return {
                "query": query,
                "answer": "Я не знаю. В базе знаний нет информации по этому вопросу.",
                "search_results": [],
                "reasoning": "Поиск не вернул результатов"
            }
        
        print(f"Найдено результатов: {len(search_results)}")
        for i, result in enumerate(search_results[:3], 1):
            print(f"  {i}. {result['title']} (релевантность: {result['relevance_score']:.3f})")
        
        # 2. Генерация ответа
        answer = self.generate_answer(query, search_results)
        
        return {
            "query": query,
            "answer": answer,
            "search_results": search_results,
            "reasoning": f"Найдено {len(search_results)} релевантных документов"
        }
    


def main():
    """Основная функция для тестирования бота"""
    print("🚀 Запуск RAG-бота")
    print("=" * 50)
    
    try:
        # Инициализируем бота
        bot = RAGBot()
        
        # Тестовые запросы
        test_queries = [
            "Что такое Energo Mech?",
            "Кто такой Dmitri Volkov?",
            "Расскажи про Pustynya",
            "Что такое Proekt Alpha?",
            "Кто такой Master Boris?"
        ]
        
        print("\n📝 Тестируем бота:")
        for query in test_queries:
            result = bot.ask(query)
            print(f"\n❓ Вопрос: {result['query']}")
            print(f"💡 Ответ: {result['answer']}")
            print(f"🔍 {result['reasoning']}")
            print("-" * 50)
        
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
