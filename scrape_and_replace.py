#!/usr/bin/env python3
"""
Скрипт для скачивания страниц Star Wars Fandom и замены терминов
"""

import requests
import json
import os
import re
from bs4 import BeautifulSoup
import time

def load_terms_map():
    """Загружает словарь замен терминов"""
    with open('terms_map.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def scrape_page(url, title):
    """Скачивает и очищает HTML страницу"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Удаляем ненужные элементы
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Извлекаем основной контент
        main_content = soup.find('div', {'id': 'mw-content-text'})
        if main_content:
            # Удаляем навигационные элементы
            for nav in main_content.find_all(['nav', 'div'], class_=re.compile(r'nav|toc')):
                nav.decompose()
            
            # Получаем чистый текст
            text = main_content.get_text(separator='\n', strip=True)
            
            # Очищаем от лишних пробелов и переносов
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r' +', ' ', text)
            
            return text.strip()
        else:
            return "Не удалось извлечь контент"
            
    except Exception as e:
        return f"Ошибка при скачивании: {str(e)}"

def replace_terms(text, terms_map):
    """Заменяет термины в тексте согласно словарю"""
    # Собираем все термины в один словарь
    all_terms = {}
    for category, terms in terms_map.items():
        all_terms.update(terms)
    
    # Сортируем по длине (длинные термины первыми)
    sorted_terms = sorted(all_terms.items(), key=lambda x: len(x[0]), reverse=True)
    
    # Заменяем термины
    for original, replacement in sorted_terms:
        # Используем регулярные выражения для точной замены
        pattern = r'\b' + re.escape(original) + r'\b'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def process_pages():
    """Основная функция обработки страниц"""
    
    # Создаем папки
    os.makedirs('knowledge_base/raw', exist_ok=True)
    os.makedirs('knowledge_base/processed', exist_ok=True)
    
    # Загружаем словарь замен
    terms_map = load_terms_map()
    
    # Читаем список страниц
    pages = []
    with open('knowledge_base/pages_to_scrape.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() and 'https://' in line:
                # Парсим строку: "1. Darth Vader - https://starwars.fandom.com/wiki/Darth_Vader"
                parts = line.split(' - ')
                if len(parts) == 2:
                    title = parts[0].split('. ', 1)[1] if '. ' in parts[0] else parts[0].strip()
                    url = parts[1].strip()
                    pages.append((title, url))
    
    print(f"Найдено {len(pages)} страниц для обработки")
    
    # Обрабатываем каждую страницу
    for i, (title, url) in enumerate(pages, 1):
        print(f"Обрабатываем {i}/{len(pages)}: {title}")
        
        # Скачиваем и очищаем страницу
        content = scrape_page(url, title)
        
        if content.startswith("Ошибка"):
            print(f"  ❌ {content}")
            continue
        
        # Сохраняем исходный текст
        raw_filename = f"knowledge_base/raw/{title.replace(' ', '_').replace('/', '_')}.txt"
        with open(raw_filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Заменяем термины
        processed_content = replace_terms(content, terms_map)
        
        # Сохраняем обработанный текст
        processed_filename = f"knowledge_base/processed/{title.replace(' ', '_').replace('/', '_')}.txt"
        with open(processed_filename, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        print(f"  ✅ Сохранено: {len(content)} -> {len(processed_content)} символов")
        
        # Пауза между запросами
        time.sleep(1)
    
    print("\n🎉 Обработка завершена!")
    print(f"Исходные файлы: knowledge_base/raw/")
    print(f"Обработанные файлы: knowledge_base/processed/")

if __name__ == "__main__":
    process_pages()
