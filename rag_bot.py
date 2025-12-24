#!/usr/bin/env python3
"""
Простой RAG-бот
Использует FAISS индекс для поиска релевантной информации
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

import faiss
from sentence_transformers import SentenceTransformer

class RAGBot:
    """Простой RAG-бот для поиска информации в базе знаний"""
    
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
                'distance': float(distance),
                'relevance_score': 1.0 / (1.0 + distance)
            })
        
        return results
    
    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """Получает полное содержимое чанка по его ID"""
        for chunk in self.metadata['chunks']:
            if chunk['chunk_id'] == chunk_id:
                return chunk.get('content', None)
        return None
    
    
    def generate_answer(self, search_results: List[Dict[str, Any]]) -> str:
        """Формирует ответ из найденных документов"""
        if not search_results:
            return "Я не знаю. В базе знаний нет информации по этому вопросу."
        
        # Берем первый самый релевантный чанк
        best_result = search_results[0]
        content = self.get_chunk_content(best_result['chunk_id'])
        
        if not content:
            return "Я не знаю. В базе знаний нет информации по этому вопросу."
        
        # Очищаем от технического мусора
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # Пропускаем пустые строки и технические метки
            if not line or line.startswith('[') or line.startswith('(') or len(line) < 3:
                continue
            # Пропускаем строки с языками и категориями
            if any(word in line.lower() for word in ['in other languages', 'related categories', 'main article']):
                break
            clean_lines.append(line)
        
        # Берем первые несколько предложений
        clean_text = ' '.join(clean_lines[:5])
        
        # Ограничиваем длину
        if len(clean_text) > 300:
            clean_text = clean_text[:300] + '...'
        
        if not clean_text:
            clean_text = content[:200].strip()
        
        # Добавляем источник
        answer = f"{clean_text}\n\nИсточник: {best_result['title']}"
        
        return answer
    
    def ask(self, query: str) -> Dict[str, Any]:
        """Основной метод для получения ответа на вопрос"""
        # Поиск релевантных чанков
        search_results = self.search(query)
        
        # Проверяем релевантность - если расстояние слишком большое, нет информации
        if not search_results or search_results[0]['distance'] > 1.0:
            return {
                "query": query,
                "answer": "Я не знаю. В базе знаний нет информации по этому вопросу.",
                "search_results": []
            }
        
        # Формируем ответ
        answer = self.generate_answer(search_results)
        
        return {
            "query": query,
            "answer": answer,
            "search_results": search_results
        }
    


def main():
    """Тестирование бота"""
    print("Запуск RAG-бота")
    
    bot = RAGBot()
    
    test_queries = [
        "Что такое Energo Mech?",
        "Кто такой Dmitri Volkov?",
        "Как приготовить борщ?"
    ]
    
    for query in test_queries:
        result = bot.ask(query)
        print(f"\nВопрос: {result['query']}")
        print(f"Ответ: {result['answer']}")


if __name__ == "__main__":
    main()
