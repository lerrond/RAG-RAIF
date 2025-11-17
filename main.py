import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
import json
# from tqdm import tqdm
from threading import Lock
from collections import Counter
import pandas as pd
# import re
import sys
import re

import numpy as np
import faiss
# import getpass
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph
import requests
# import os.path
# import langchain_text_splitters.html
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
# import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langgraph.graph import END
from openai import OpenAI

#from syte_lera import age_group

LOG_PATH = "rag_logs.log"
logger.remove()
logger.add(sys.stderr, level="DEBUG", backtrace=False, diagnose=False)
# сохраняем логи дял разрабов
logger.add(
    LOG_PATH,
    rotation="10 MB",
    retention="30 days",
    encoding="utf-8",
    level="DEBUG",
    enqueue=True,
    backtrace=True,
    diagnose=True
)

# Логи в консоль
# logger.add(
#     sys.stdout,
#     level="DEBUG",    # показывать DEBUG и выше
# )
# ПРОМПТЫ:
# -------------------------------------------
router_instructions = """
    Ты являеешься умнейшим экономистом, помогающим отвечать на вопросы подростков, относяшихся к экономике или финансовой грамотности.
    Ты можешь направить вопрос пользователя в vectorstore или в autoanswer, а также твоя задача - анализировать вопросы пользователей и определять для них
    релевантные тэги из строго заданного списка, который называется 'СПИСОК ДОСТУПНЫХ ТЭГОВ', он предоставлен тебе ниже и содержит 17 уникальных тэгов, перечисленных через
    запятую.

    СПИСОК ДОСТУПНЫХ ТЭГОВ: 'Экономика', 'Подработка', 'Банковские карты', 'Банковские вклады и кредиты', 'Инвестиции', 'Покупки', 'Жилье',
    'Учебные материалы', 'Карманные деньги', 'Мошенники', 'Экономия', 'Сбережения', 'Налоги', 'Права', 'Документы', 'Финансовые цели', 'Иное'

    В vectorstore собраны статьи из интернет-блогов для подростков. В них содержится различная информация - об экономике, финансовой грамотности, безопасности,
    защите от мошенничества, инвестициях, страховании, пенсиях и многом другом. 

    Для вопросов по схожим темам надо использовать vectorstore.

    Если вопрос пользователя содержит в себе непристойную лексику, агрессивные высказывания, неккоректные с точки зрения морали высказывания,
    острополитические вопросы, то есть все, на что не должен отвечать наставник подростка согласно моральным устоям - то нужно использовать autoanswer.
    Если подросток говорит, что кто-то просит от него денег - то нужно ОБЯЗАТЕЛЬНО пойти в vectorstore. 
    Если подросток говорит про накопления - о нужно ОБЯЗАТЕЛЬНО пойти в vectorstore. 
    Если подросток спрашивает это из соображений своей безопасности, например, он делится ситуацией из жизни, в которой он мог столкнуться с мошенником, то нужно ОБЯЗАТЕЛЬНО пойти в vectorstore.
    Так никто не потеряет репутацию, а подросток останется доволен ответом.

    Если ты решил использовать vectorstore для вопроса, то к данному вопросу тебе также нужно определить тэги по следующей интсрукции:
    1. Проанализируй вопрос пользователя и определи 3 наиболее релевантных тэга из списка выше
    2. Тэги должны максимально точно отражать суть вопроса и быть уникальными (то есть среди трех выбранных тэгов не может быть двух или трех одинаковых)
    3. Если вопрос касается нескольких тем, подбери три тэга так, чтобы они максимально полно охватывали все темы вопроса
    4. Всегда возвращай ровно 3 тэга
    5. Используй ТОЛЬКО ТЭГИ из предоставленного СПИСКА ДОСТУПНЫХ ТЭГОВ

    ФОРМАТ ОТВЕТА - ТОЛЬКО JSON:
{
    "datasource": "vectorstore",
    "tags": ["тэг1", "тэг2", "тэг3"]
}

ИЛИ:
{
    "datasource": "autoanswer",
    "tags": []
}
"""

rag_prompt = """ Ты - российский чат-бот для подростков, помогающий молодежи познавать финансовую грамотность. Ты ассистент, помогающий отвечать на вопросы ПОЛЬЗОВАТЕЛЯ . Ты можешь отвечать на вопросы только на русском.
    Помни - отвечаешь на вопросы ПОДРОСТКА. Не пиши слишком сложно и нудно, рассказывай все простыми словами и главное - интересно. Иногда можешь использовать ассоциации.
    Или иные способы, с помощью которых сложную информацию можно легко понять. 
    Возраст пользователя: 
    {age_group}
    
    Это контекст, который ты можешь использовать для ответа на поставленный вопрос:
    {context} 
    
    Тщательно подумай над предоставленным контекстом. Однако, тебе не нужно в ответе пользователю указывать на то, что ты опираешься на какой-то контекст.
    
    Теперь ответь на вопрос пользователя:
    
    {question}
    
    Используя только заданный контекст, напиши ответ на вопрос. Ответ на вопрос должен быть максимально обширным, и давать исчерпывающую информацию по теме,
    которой интересуется пользователь. Ответ не должен включать слишком много сложноподчиненных предложений. Если есть вариант упростить предложение - упрости, но ответь полно. Иногда используй и сложные предложения.
    Всегда отвечай исключительно на русском языке. Если хочешь ответить на вопрос, используя перечисление пунктов (например, как в случае рекомендаци), то сделай это.
    Старайся дать ответ в той форме, которая соотвествет возрасту твоего пользователя. Если возраст пользователя 10-14 не пиши много про подработку/работу/зарплату и другие сложные для ребнка вещи, старайся обьяснять все на понятных примерах. 
    Если возраст пользователя 14-18 - можешь писать про подработку/работу и прочие взрослые вещи, старайся не давать простых примеров для объяснения.
    
    Твой ответ должен состоять в среднем из более чем 6 предложений. Если ты думашеь, что нужно больше пояснений - используй больше слов. Не пиши их количество пользователю.
    Еще раз: твой ответ должен МАКСИМАЛЬНО ТОЧНО и ШИРОКО отвечать на поставленный вопрос.
    
    Можешь начать свой ответ с: "Привет! Давай разберемся". Или как-то по-другому, на твой вкус и цвет. 
    Если  возраст пользователя 10-14, то  добавь "Пока дружок, будь осторожен в интернете!" в конце своего ответа. Если возраст пользователя 14-18,  добавь в конце своего ответа "Пока приятель, не теряй своего интереса к финансам!"

    Ответ:"""

tags_for_docs_instructions = """
    Ты являеешься умнейшим экономистом, помогающим классифицировать документы, относяшиеся к экономике или финансовой грамотности, по более узким тэгам.
    Твоя задача для документа подобрать соответсвующие ему 3 тэга из "СПИСОК ДОСТУПНЫХ ТЭГОВ", тэги должны максимально полно и точно отражать основную суть и тему документа

    СПИСОК ДОСТУПНЫХ ТЭГОВ: 'Экономика', 'Подработка', 'Банковские карты', 'Банковские вклады и кредиты', 'Инвестиции', 'Покупки', 'Жилье',
    'Учебные материалы', 'Карманные деньги', 'Мошенники', 'Экономия', 'Сбережения', 'Налоги', 'Права', 'Документы', 'Финансовые цели', 'Иное'



    Тэги нужно определять по соответсвующей инструкции:
    1. Проанализируй документ  и определи 3 наиболее релевантных тэга из списка выше
    2. Тэги должны максимально точно отражать суть документа и быть уникальными (то есть среди трех выбранных тэгов не может быть двух или трех одинаковых)
    3. Если документ касается нескольких тем, подбери три тэга так, чтобы они максимально полно охватывали все темы вопроса
    4. Всегда возвращай ровно 3 тэга
    5. Используй ТОЛЬКО ТЭГИ из предоставленного СПИСКА ДОСТУПНЫХ ТЭГОВ

    ФОРМАТ ОТВЕТА - ТОЛЬКО JSON:
{
    "tags": ["тэг1", "тэг2", "тэг3"]
}"""



hallucination_grader_prompt = """ФАКТЫ: \n\n {documents} \n\n ОТВЕТ СТУДЕНТА: {generation}. 
Ответь в формате JSON (не текстом, а именно json-файлом) с помощью двух ключей, первый это  binary_score - это оценка "yes" или "no", чтобы указать, основан ли ОТВЕТ СТУДЕНТА на ФАКТАХ. 
Второй ключ это explanation - пояснение, которое содержит объяснение поставленного  binary_score.
"""

hallucination_grader_instructions = """

Ты являешься преподавателем, оценивающим ответ учащегося на основе предоставленного фактического материала.
На вход ты получишь два артефакта:
- ФАКТЫ: ключевая фактическая информация, извлеченная из документов.
- ОТВЕТ СТУДЕНТА: сгенерированный ответ на вопрос пользователя.

Твоя цель - определить, подтверждается ли ОТВЕТ СТУДЕНТА ФАКТАМИ. То есть, отражает ли ОТВЕТ СТУДЕНТА информацию из ФАКТОВ. И была ли сохранена логика.

Используй следующие критерии:

1. Ответ не обязательно должен в точности повторять ФАКТЫ, но он должен отражать то же значение или не противоречить им.
2. Допустимо, чтобы ОТВЕТ СТУДЕНТА включал легкое перефразирование, обобщение или разумные выводы, если они
основаны на предоставленном содержании.
3. Не помечай ответ как неправильный, если только он явно не содержит выдуманной, неподтвержденной или противоречивой информации
 
"""

answer_grader_prompt = """ВОПРОС: \n\n {question} \n\n ОТВЕТ УЧЕНИКА: {generation}. 

Верни JSON с двумя ключами: binary_score это 'yes' или 'no'  - скор, который показывает, подходит ли ОТВЕТ УЧЕНИКА под критерии. 
И explanation, объяснение, которое поясняет выставленный binary_score."""

answer_grader_instructions = """
    
    Ты - преподаватель, оценивающий результаты теста. Тебе будет показан ВОПРОС, заданный УЧЕНИКУ, и ОТВЕТ УЧЕНИКА. 
    Ниже приведены критерии выставления оценки, которым тебе необходимо следовать:

    (1) ОТВЕТ УЧЕНИКА помогает ответить на поставленный ВОПРОС. 
    (2) УЧЕНИК не отвечает на острополитический, аморальный ВОПРОС. УЧЕНИКУ категорически нельзя отвечать на ВОПРОСЫ, в которых затрагивается
    расовая принадлежность с аморальной точки зрения, черный юмор, непристойные темы.  
    
    Правила выставления оценки УЧЕНИКУ:

    Оценка "yes" означает, что ответ учащегося соответствует всем критериям. Это наивысший балл. 

    Учащийся может получить оценку "yes", если ответ содержит дополнительную информацию, которая явно не указана в вопросе. Но информация не должна
    перечить общим жизненным закономерностям. 

    Оценка "no" означает, что ответ учащегося не соответствует всем критериям. Это максимально низкий балл, который ты можешь выставить.

    Объясни свои рассуждения поэтапно, чтобы убедиться в правильности своих рассуждений и выводов. 

    Избегай простого указания правильного ответа в начале.
    
    """

# -------------------------------------------
# ПАЙПЛАЙН:

load_dotenv()


OPENROUTER_API_KEY='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'
# EMBEDDER_API_KEY='sk-UCKjC2dshq8VKM2YVdNxkw'
# #
# LLM_API_KEY = os.getenv("LLM_API_KEY")
# # Подключаем ключ для EMBEDDER-модели
# EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


llm_client = OpenAI(
        # Базовый url - сохранять без изменения
        base_url="https://openrouter.ai/api/v1/",
        # Указываем наш ключ, полученный ранее
        api_key=OPENROUTER_API_KEY,
    )

embed_client = OpenAI(
    base_url="https://openrouter.ai/api/v1/",
    api_key=OPENROUTER_API_KEY,
)


# штука для тегов
TAG_HISTORY = []
TAG_HISTORY_LOCK = Lock()

def add_tags_to_session(tags):
    if not tags:
        return
    if isinstance(tags, str):
        tags = [tags]
    with TAG_HISTORY_LOCK:
        for t in tags:
            if isinstance(t, str) and t.strip():
                TAG_HISTORY.append(t.strip())

def reset_tag_history():
    with TAG_HISTORY_LOCK:
        TAG_HISTORY.clear()

def get_tag_stats():
    with TAG_HISTORY_LOCK:
        return dict(Counter(TAG_HISTORY))

# ------------------------


def llm_generate(prompt_text: str):

    response = llm_client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}]
            }
        ]
    )
    return response.choices[0].message.content


def llm_json(prompt_text: str, system_text: str = ""):

    messages = []
    if system_text:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_text}]
        })
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": prompt_text}]
    })

    response = llm_client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=messages,
        temperature=0.1
    )
    return response.choices[0].message.content



def rerank_docs_local(query, docs):

    pairs = [(query, doc) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    return ranked_indices, scores


def get_embedding(text):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1/",
        api_key=OPENROUTER_API_KEY,
    )
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            print("Ошибка парсинга JSON:", e)
            return None
    else:
        print("JSON не найден в тексте")
        return None

def autoanswer(state):

    logger.debug("---АВТООТВЕТ НА НЕКОРРЕКТНЫЙ ВОПРОС---")
    return {"generation": "Кажется,я еще не научился отвечать на подобные вопросы!:("}



def load_train_csv(file_path: str = "./train_data_child.csv"):

    dff = pd.read_csv(file_path)

    documents = []
    for _, row in dff.iterrows():

        combined_text = f"{row.iloc[1]} {row.iloc[2]}"

        tags_for_doc = llm_json(
            prompt_text=combined_text,
            system_text=tags_for_docs_instructions
        )
        tags_for_doc = tags_for_doc.replace("```json", "").replace("```", "").strip()
        doc_tags = json.loads(tags_for_doc)["tags"]
        doc_tags=[tag.strip().strip("'\"") for tag in doc_tags]
        metadata = {
            "source": file_path,

            "tags": doc_tags
        }

        documents.append(Document(page_content=combined_text, metadata=metadata))

    return documents

def split_documents(documents: list, chunk_size: int =7500, chunk_overlap: int = 1500) -> list:

    logger.debug(f"Разбиение на чанки: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=['!','?','.'])
    return splitter.split_documents(documents)


class EmbeddingWrapper:
    def embed_documents(self, texts):
        return [get_embedding(t) for t in texts]

    def embed_query(self, query):
        return get_embedding(query)


    def __call__(self, text):
        return self.embed_query(text)

# embedding_wrapper = EmbeddingWrapper()

def indexed_df():
    embedding_wrapper = EmbeddingWrapper()
    index_path = "faiss_index"

    if os.path.isdir(index_path):
        try:
            logger.debug(f" Попытка загрузки FAISS из {index_path}")
            db = FAISS.load_local(
                index_path,
                embeddings=embedding_wrapper,
                allow_dangerous_deserialization=True
            )
            logger.debug("✅ FAISS успешно загружен.")
            return db
        except Exception as e:
            logger.warning(f"Не удалось загрузить FAISS ({index_path}). Пересоздаём. Ошибка: {e}")


    logger.debug("Создаём FAISS с нуля...")

    documents = load_train_csv("train_data_child.csv")
    if not documents:
        raise ValueError("Нет данных для индексации.")


    print("Документы:", documents[:5])


    split_docs = split_documents(documents)
    texts = [d.page_content for d in split_docs]
    metadatas = [d.metadata for d in split_docs]


    print("texts:", texts[:5])

    logger.debug(f"Чанков для индексации: {len(texts)}")


    db = FAISS.from_texts(
        texts=texts,
        embedding=embedding_wrapper,
        metadatas=metadatas  # Добавляем метаданные
    )


    os.makedirs(index_path, exist_ok=True)
    db.save_local(index_path)
    logger.debug(f"FAISS сохранён в {index_path}")

    return db


class GraphState(TypedDict):


    question: str     # Вопрос пользователя
    generation: str   # LLM генерация
    autoanswer: str   # Двоичное решение об ответе (можем ли ответить в прицнипе)
    max_retries: int  # Максимальное количество повторных попыток генерации
    answers: int      # Количество сгенерированных ответов
    #loop_step: Annotated[int, operator.add]
    loop_step: int
    documents: List[str]  # Список найденных документов
    relevant_tags: str
    choice: str
    age_group: str #возрастная групап

#вспомогательная функция для retrieve
def retriever_tag(db, allowed_tags=None, k=5):

    if allowed_tags:
        def custom_tag_filter(metadata):
            doc_tags = metadata.get('tags', [])
            # Приводим всё к списку
            if not isinstance(doc_tags, (list, tuple)):
                doc_tags = [doc_tags] if isinstance(doc_tags, str) else []
            return any(tag in doc_tags for tag in allowed_tags)

        retriever = db.as_retriever(
            search_kwargs={
                "k": k,
                "filter": custom_tag_filter
            }
        )
    else:
        retriever = db.as_retriever(search_kwargs={"k": k})

    return retriever

def retrieve(state: GraphState):

    logger.debug("---ПОИСК С УЧЕТОМ МЕТАДАННЫХ---")
    # logger.debug(f"ВСЕ СОСТОЯНИЕ В RETRIEVE: {state}")
    logger.debug(f"Ключи состояния: {list(state.keys())}")

    relevant_tags = state.get("relevant_tags", []) #получаем тэги вопроса из состояния
    logger.debug(f"Полученные тэги: {relevant_tags}")

    question = state["question"]



    retriever_tags = retriever_tag(
         df,
         allowed_tags=relevant_tags,
         k=5 #берем k документов с фильтрацией по общим тэгам и семантике
     )

    documents_tag = retriever_tags.invoke(question)
    logger.debug(f'documents_tag = {documents_tag}')


    return {"documents": documents_tag}

def reranke(state):
    logger.debug('---РАНЖИРОВАНИЕ ДОКУМЕНТОВ (локально)---')

    documents = state.get('documents', [])
    if not documents:
        logger.debug("Нет документов для реранкинга, возвращаем автоответ.")
        return {"autoanswer": "Yes"}

    text_docs = []
    for d in documents:
        if hasattr(d, "page_content"):
            text_docs.append(str(d.page_content))
        elif isinstance(d, dict) and "page_content" in d:
            text_docs.append(str(d["page_content"]))
        else:
            text_docs.append(str(d))

    question = state.get("question", "")
    logger.debug(f"Вопрос: {question[:100]}")

    pairs = [(question, doc_text) for doc_text in text_docs]
    scores = cross_encoder.predict(pairs)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    reranked_documents = [documents[i] for i in ranked_indices]

    logger.debug("Результаты реранкинга:")
    for rank, idx in enumerate(ranked_indices, start=1):
        score = scores[idx]
        snippet = text_docs[idx][:100]
        logger.debug(f"Rank {rank} | Score: {score:.4f} | Text snippet: {snippet}")

    return {"documents": reranked_documents}



def generate(state):

    logger.debug("---СГЕНЕРИРОВАТЬ---")

    loop_step = state.get("loop_step", 0)

    # logger.debug('RAG generation')

    feedback = state.get("feedback", "")
    rag_prompt_formatted = rag_prompt.format(
        context=state["documents"],
        question=state["question"] + (
            "\nТвой прошлый ответ мне не понравился. И вот почему: " + feedback if feedback else "") ,
        age_group = state['age_group']
    )
    generation = llm_generate(rag_prompt_formatted)
    logger.debug(f'generation={generation}')
    # logger.debug('----ФИЛЬТРАЦИЯ ИНОСТРАННЫХ СИМВОЛОВ----')
    filtered_generation=generation
    # filtered_generation = re.sub(r'[^a-zA-Zа-яА-Я0-9,.?\/ёЁ%#*-—:№\n]', ' ', generation)
    # filtered_generation = re.sub(r'\s+', ' ', filtered_generation).strip()
    return {"generation": filtered_generation, "loop_step": loop_step + 1}




def route_question(state):
    route_response = llm_json(prompt_text=state["question"], system_text=router_instructions)
    route_response = route_response.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(route_response)
    except Exception as e:
        logger.error("Не удалось распарсить JSON от роутера: %s | raw: %s", e, route_response)

        return {"choice": "autoanswer", "relevant_tags": []}

    source = parsed.get("datasource")
    tags = parsed.get("tags", [])


    if tags:
        add_tags_to_session(tags)

    if source == "autoanswer":
        return {"choice": "autoanswer", "relevant_tags": []}
    elif source == "vectorstore":
        return {"choice": "vectorstore", "relevant_tags": tags}
    else:
        logger.warning("router returned unknown datasource: %s", source)
        return {"choice": "autoanswer", "relevant_tags": []}

#функция для связи маршрутизатора и ретривера
def router_to_retriever(state):
    choice = state.get("choice")
    print(f"Роутер видит choice: {choice}")

    if choice == "vectorstore":
        return "vectorstore"
    elif choice == "autoanswer":
        return "autoanswer"


def decide_to_generate(state):


    logger.debug("---РЕЛЕВАНТНЫ ЛИ ДОКУМЕНТЫ?---")
    question = state.get("question", "")
    autoanswer = state.get("autoanswer", "No")
    filtered_documents = state.get("documents", [])


    if autoanswer == "Yes":

        logger.debug(
            "---БЫЛ ВЫБРАН АВТОМАТИЧЕСКИЙ ОТВЕТ---"
        )
        return "autoanswer"
    else:

        logger.debug("---РЕШЕНИЕ: ГЕНЕРИРОВАТЬ---")
        return "generate"


def judge_model(instructions: str, prompt_text: str):


    response = llm_client.chat.completions.create(
        model="mistralai/mistral-small-3.2-24b-instruct",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": instructions  # твои системные инструкции
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text  # конкретная проверка
                    }
                ]
            }
        ],
        temperature=0.2
    )


    return response.choices[0].message.content

def grade_generation_v_documents_and_question(state):


    logger.debug("---ПРОВЕРИТЬ ГАЛЛЮЦИНАЦИИ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)

    max_retries = int(max_retries)
    loop_step = int(state.get("loop_step", 0))

    hallucination_prompt = hallucination_grader_prompt.format(
        documents=documents, generation=generation
    )
    result_text = judge_model(
        instructions=hallucination_grader_instructions,
        prompt_text=hallucination_prompt
    )


    result_text = result_text.replace("```json", "").replace("```", "").strip()

    try:
        if not result_text:
            raise ValueError("Пустой ответ судьи")
        result_json = json.loads(result_text)
        grade = result_json.get("binary_score", "no")
        explanation = result_json.get("explanation", "")
    except Exception as e:
        grade = "no"
        explanation = f"Ошибка парсинга ответа судьи: {result_text}. Exception: {e}"
        logger.warning(explanation)

    logger.debug(f"---ОЦЕНКА ГАЛЛЮЦИНАЦИИ: {grade} --- ОБЪЯСНЕНИЕ: {explanation}")

    if grade.lower() == "yes":
        logger.debug("---РЕШЕНИЕ: ГЕНЕРАЦИЯ ОСНОВАНА НА ДОКУМЕНТАХ---")
        logger.debug("---Оценка: ГЕНЕРАЦИЯ против ВОПРОСА---")

        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation
        )
        result_text = llm_json(answer_grader_prompt_formatted, answer_grader_instructions)
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        # lean_text = result_text.replace("```json", "").replace("```", "").strip()


        json_start = result_text.find("{")
        if json_start != -1:
            json_text = result_text[json_start:]
            try:
                if not result_text:
                    raise ValueError("Пустой результат оценки ответа")
                result_json = json.loads(json_text)
                grade_answer = result_json.get("binary_score", "no")
            except Exception as e:
                grade_answer = "no"
                logger.warning(f"Ошибка парсинга JSON при оценке ответа: {result_text}. Exception: {e}")

        if grade_answer.lower() == "yes":
            logger.debug("---РЕШЕНИЕ: GENERATION ОБРАЩАЕТСЯ К ВОПРОСУ---")
            return "useful"
        elif loop_step <= max_retries:
            logger.debug("---РЕШЕНИЕ: GENERATION НЕ ОТВЕЧАЕТ НА ВОПРОС---")
            return "not useful"
        else:
            logger.debug("---РЕШЕНИЕ: МАКСИМАЛЬНОЕ КОЛИЧЕСТВО ПОВТОРНЫХ ПОПЫТОК ДОСТИГНУТО---")
            return "max retries"

    elif loop_step <= max_retries:
        logger.debug("---РЕШЕНИЕ: ГЕНЕРАЦИЯ НЕ ОСНОВАНА НА ДОКУМЕНТАХ, ПОВТОРИТЕ ПОПЫТКУ---")
        return "not supported"
    else:
        logger.debug("---РЕШЕНИЕ: МАКСИМАЛЬНОЕ КОЛИЧЕСТВО ПОВТОРНЫХ ПОПЫТОК ДОСТИГНУТО---")
        return "max retries"



workflow = StateGraph(GraphState)

# Определение узлов
workflow.add_node("autoanswer", autoanswer)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("reranker", reranke)# reranke
# workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("route", route_question)

workflow.set_entry_point("route")

#для нормальной связи ретривера и роутера
workflow.add_conditional_edges(
    "route",
    router_to_retriever,
    {
        "vectorstore": "retrieve",
        "autoanswer": "autoanswer"
    }
)
workflow.add_edge("autoanswer", END)
# workflow.add_edge("retrieve", "grade_documents")
# workflow.add_edge("grade_documents", "reranker")
workflow.add_edge("retrieve", "reranker")
workflow.add_conditional_edges(
    "reranker",
    decide_to_generate,
    {
        "autoanswer": "autoanswer",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "autoanswer",
        "max retries": END,
    },
)


graph = workflow.compile()

global df
df = indexed_df()
retriever = df.as_retriever(k=4)


def answer_question(question: str, age_group: str = None, max_retries: int = 3) -> str:

    logger.debug(f"Получен вопрос: {question}")

    # Загрузка индекс FAISS
    df = indexed_df()
    logger.debug("FAISS индекс загружен")
    retriever = df.as_retriever(k=4)
    logger.debug("Retriever создан")


    inputs = {
        "question": question,
        "max_retries": max_retries,
        "age_group": age_group
    }

    final_answer = "Ошибка: ответ не сгенерирован"

    # Прогоняем вопрос через граф
    for event in graph.stream(inputs, stream_mode="values"):
        if "generation" in event and hasattr(event["generation"], "content"):
            final_answer = event["generation"].content
            break

    logger.debug(f"Сгенерированный ответ: {final_answer}")
    return final_answer


#if __name__ == "__main__":
#    test_question = "Что делать, если шантажируют интимными фото и видео?"
#    df = indexed_df()
#    logger.debug('Create retriever')

#    retriever = df.as_retriever(k=4)
#    inputs = {"question": test_question, "max_retries": 2}
#    print(f"\n🔹 Вопрос: {test_question}\n")


#    for event in graph.stream(inputs, stream_mode="values"):
#        pass  # просто дожидаемся завершения пайплайна

#
#    final_answer = event.get("generation", "Ошибка: ответ не сгенерирован")

#    print("\nФинальный ответ модели:")
#    print(final_answer)


#import anyio



# async def process_question(question_text, graph):
#
#     inputs = {"question": question_text, "max_retries": 2}
#
#     for attempt in range(1, 3 + 1):
#         try:
#             # logger.debug(f" Обработка вопроса (попытка {attempt}): {question_text}")
#             final_answer = None
#
#             async for event in graph.astream(inputs, stream_mode="values"):
#                 final_answer = event.get("generation")
#
#             if final_answer and final_answer.strip():
#                 # logger.debug(f" Успешно получен ответ на вопрос: {question_text}")
#                 return final_answer
#
#             raise ValueError("Пустой ответ от модели")
#
#         except Exception as e:
#             # logger.error(f"Ошибка при обработке '{question_text}' (попытка {attempt}): {e}")
#             if attempt < 3:
#                 # logger.debug(f" Повтор через 10 секунд...")
#                 await anyio.sleep(10)
#             else:
#                 # logger.error(f" Все {3} попытки исчерпаны для вопроса: {question_text}")
#                 return "Кажется,я еще не научился отвечать на подобные вопросы!:("
#
#
# async def main():
#     csv_path = "./questions.csv"
#     output_path = "submission.csv"
#
#     # logger.debug(" Запуск асинхронной генерации ответов...")
#
#
#     questions = pd.read_csv(csv_path)
#     questions_list = questions['Вопрос'].tolist()
#
#
#     global df
#     df = indexed_df()
#
#     #global retriever
#     #retriever = df.as_retriever(search_kwargs={"k":3})
#
#     # logger.debug(f"Всего вопросов для обработки: {len(questions_list)}")
#
#     results = []
#
#
#     async with anyio.create_task_group() as tg:
#         async def handle_question(q):
#             answer = await process_question(q, graph)
#             results.append((q, answer))
#
#         for q in questions_list:
#             tg.start_soon(handle_question, q)
#
#
#     question_to_answer = dict(results)
#     questions["Ответы на вопрос"] = questions["Вопрос"].map(question_to_answer)
#
#
#     # questions["Ответы на вопрос"] = questions["Ответы на вопрос"].apply(clean_generation)
#
#     # Сохраняем в CSV
#     questions.to_csv(output_path, index=False)
#     # logger.debug(f" Ответы сохранены в файл {output_path}")
#     print(f" Ответы сохранены в {output_path}")
#
#
# if __name__ == "__main__":
#    anyio.run(main)