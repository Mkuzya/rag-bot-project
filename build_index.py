#!/usr/bin/env python3
"""
Скрипт для создания векторного индекса базы знаний RAG-бота.
Создает FAISS индекс из обработанных текстов с использованием sentence-transformers.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import faiss

# Конфигурация
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
KNOWLEDGE_BASE_DIR = "knowledge_base/processed"
OUTPUT_DIR = "vector_index"
INDEX_FILE = "faiss_index.index"
METADATA_FILE = "metadata.json"

def load_documents() -> List[Document]:
    """Загружает все документы из папки knowledge_base/processed"""
    documents = []
    
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"❌ Папка {KNOWLEDGE_BASE_DIR} не найдена!")
        return documents
    
    for file_path in Path(KNOWLEDGE_BASE_DIR).glob("*.txt"):
        print(f"📄 Загружаем: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if content:
                # Создаем документ с метаданными
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "title": file_path.stem,
                        "file_type": "processed_text"
                    }
                )
                documents.append(doc)
                print(f"  ✅ Загружено: {len(content)} символов")
            else:
                print(f"  ⚠️  Пустой файл: {file_path.name}")
                
        except Exception as e:
            print(f"  ❌ Ошибка загрузки {file_path.name}: {e}")
    
    print(f"\n📊 Всего загружено документов: {len(documents)}")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Разбивает документы на чанки"""
    print(f"\n🔪 Разбиваем документы на чанки...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for i, doc in enumerate(documents):
        doc_chunks = text_splitter.split_documents([doc])
        
        # Добавляем chunk_id к метаданным
        for j, chunk in enumerate(doc_chunks):
            chunk.metadata.update({
                "chunk_id": f"{doc.metadata['title']}_chunk_{j}",
                "chunk_index": j,
                "total_chunks": len(doc_chunks)
            })
        
        chunks.extend(doc_chunks)
        print(f"  📄 {doc.metadata['title']}: {len(doc_chunks)} чанков")
    
    print(f"📊 Всего создано чанков: {len(chunks)}")
    return chunks

def create_embeddings(chunks: List[Document]) -> np.ndarray:
    """Создает эмбеддинги для всех чанков"""
    print(f"\n🤖 Создаем эмбеддинги с помощью {EMBEDDING_MODEL}...")
    
    # Загружаем модель
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✅ Модель загружена: {EMBEDDING_MODEL}")
    
    # Извлекаем тексты
    texts = [chunk.page_content for chunk in chunks]
    
    # Создаем эмбеддинги
    print(f"  🔄 Обрабатываем {len(texts)} чанков...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"  ✅ Создано эмбеддингов: {embeddings.shape}")
    print(f"  📏 Размерность: {embeddings.shape[1]}")
    
    return embeddings

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Создает FAISS индекс"""
    print(f"\n🗂️  Создаем FAISS индекс...")
    
    # Создаем индекс L2 (Euclidean distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Добавляем векторы в индекс
    index.add(embeddings.astype('float32'))
    
    print(f"  ✅ Индекс создан: {index.ntotal} векторов")
    return index

def save_metadata(chunks: List[Document], embeddings: np.ndarray) -> Dict[str, Any]:
    """Сохраняет метаданные"""
    metadata = {
        "model": EMBEDDING_MODEL,
        "embedding_dimension": embeddings.shape[1],
        "total_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks": []
    }
    
    for i, chunk in enumerate(chunks):
        chunk_metadata = {
            "chunk_id": chunk.metadata["chunk_id"],
            "source": chunk.metadata["source"],
            "title": chunk.metadata["title"],
            "chunk_index": chunk.metadata["chunk_index"],
            "content_preview": chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
        }
        metadata["chunks"].append(chunk_metadata)
    
    return metadata

def test_search(index: faiss.Index, chunks: List[Document], model: SentenceTransformer):
    """Тестирует поиск в индексе"""
    print(f"\n🔍 Тестируем поиск...")
    
    test_queries = [
        "Что такое Energo Mech?",
        "Кто такой Dmitri Volkov?",
        "Расскажи про Pustynya",
        "Что такое Proekt Alpha?"
    ]
    
    for query in test_queries:
        print(f"\n  🔍 Запрос: '{query}'")
        
        # Создаем эмбеддинг для запроса
        query_embedding = model.encode([query])
        
        # Ищем ближайшие векторы
        k = 3  # Топ-3 результата
        distances, indices = index.search(query_embedding.astype('float32'), k)
        
        print(f"    📋 Найдено результатов: {len(indices[0])}")
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            chunk = chunks[idx]
            print(f"    {i+1}. Расстояние: {distance:.4f}")
            print(f"       Источник: {chunk.metadata['title']}")
            print(f"       Превью: {chunk.page_content[:150]}...")
            print()

def main():
    """Основная функция"""
    print("🚀 Создание векторного индекса для RAG-бота")
    print("=" * 50)
    
    # Создаем выходную папку
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Загружаем документы
    documents = load_documents()
    if not documents:
        print("❌ Нет документов для обработки!")
        return
    
    # 2. Разбиваем на чанки
    chunks = split_documents(documents)
    
    # 3. Создаем эмбеддинги
    embeddings = create_embeddings(chunks)
    
    # 4. Создаем FAISS индекс
    index = create_faiss_index(embeddings)
    
    # 5. Сохраняем индекс
    index_path = os.path.join(OUTPUT_DIR, INDEX_FILE)
    faiss.write_index(index, index_path)
    print(f"💾 Индекс сохранен: {index_path}")
    
    # 6. Сохраняем метаданные
    metadata = save_metadata(chunks, embeddings)
    metadata_path = os.path.join(OUTPUT_DIR, METADATA_FILE)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Метаданные сохранены: {metadata_path}")
    
    # 7. Тестируем поиск
    model = SentenceTransformer(EMBEDDING_MODEL)
    test_search(index, chunks, model)
    
    print("\n🎉 Векторный индекс успешно создан!")
    print(f"📁 Результаты в папке: {OUTPUT_DIR}/")
    print(f"📄 Файлы:")
    print(f"  - {INDEX_FILE} (FAISS индекс)")
    print(f"  - {METADATA_FILE} (метаданные)")

if __name__ == "__main__":
    main()
