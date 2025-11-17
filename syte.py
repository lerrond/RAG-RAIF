import threading
from collections import Counter
from threading import Lock
import streamlit as st
from loguru import logger
import main as sr
import ast
import pandas as pd
import time
import os
import datetime
import plotly.express as px

# ------------------------------
# История тегов
# ------------------------------
# TAG_HISTORY = []
# TAG_HISTORY_LOCK = Lock()

if "tag_history" not in st.session_state:
    st.session_state.tag_history = []

def add_tags_to_session(tags):
    if not tags:
        return
    if isinstance(tags, str):
        tags = [tags]
    for t in tags:
        if isinstance(t, str) and t.strip():
            st.session_state.tag_history.append(t.strip())

def get_tag_stats():
    return dict(Counter(st.session_state.tag_history))

def reset_tag_history():
    st.session_state.tag_history = []



## #############
# Настройка теста
#############
def calculate_result(answers,answ=None):

    counts = {"A": 0, "B": 0, "C": 0}

    for answer in answers:
        counts[answer] += 1

    max_count = max(counts.values())
    max_chars = [char for char, count in counts.items() if count == max_count]


    if len(max_chars) > 1 or (max_count - min(counts.values())) <= 1:
        return "D"

    return answ[answ['character_results']==max_chars[0]]['name'].values[0]

def run_quiz(tag=None,demo_test=None,answ=None):

    if tag is None:
        demo_test=demo_test[demo_test['tag']=='Общий '].reset_index(drop=True).copy()
        answ=answ[answ['tag']=='Общий '].reset_index(drop=True).copy()
    else:
        demo_test = demo_test[demo_test['tag'] == tag].reset_index(drop=True).copy()
        answ = answ[answ['tag'] == tag].reset_index(drop=True).copy()


    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    if "answers" not in st.session_state:
        st.session_state.answers = []
    if "quiz_completed" not in st.session_state:
        st.session_state.quiz_completed = False


    if not st.session_state.quiz_started and not st.session_state.quiz_completed:
        st.markdown(f"### {demo_test['title'].values[0]}")# #QUIZ_DATA['title']
        st.markdown(demo_test['description'].values[0]) #QUIZ_DATA['description']
        st.markdown("**Ответь на 5 простых вопросов и открой своего внутреннего финансиста!**")

        if st.button("Начать тест!", type="primary", key="start_quiz"):
            st.session_state.quiz_started = True
            st.rerun()
        return


    elif st.session_state.quiz_started and not st.session_state.quiz_completed:

        progress = st.session_state.current_question /5
        st.progress(progress)
        st.write(f"Вопрос {st.session_state.current_question + 1} из {5}")


        question_data = demo_test.loc[st.session_state.current_question,['questions','options','letters']] # QUIZ_DATA["questions"][st.session_state.current_question]
        st.subheader(question_data["questions"])

        selected_option = st.radio(
            "Выбери ответ:",
            question_data["options"],
            key=f"question{st.session_state.current_question}"
        )

        col1, col2 = st.columns([1, 4])

        with col1:
            if st.button("Далее →", type="primary", key=f"next_{st.session_state.current_question}"):

                option_index = question_data["options"].index(selected_option)
                st.session_state.answers.append(question_data["letters"][option_index])


                if st.session_state.current_question < 4:
                    st.session_state.current_question += 1
                    st.rerun()
                else:
                    st.session_state.quiz_completed = True
                    st.rerun()

        with col2:
            if st.session_state.current_question > 0:
                if st.button("← Назад", key=f"back_{st.session_state.current_question}"):
                    st.session_state.current_question -= 1
                    st.session_state.answers.pop()
                    st.rerun()


    elif st.session_state.quiz_completed:
        result_char = calculate_result(st.session_state.answers,answ=answ)


        st.success("🎉 Тест завершён! 🎉")

        if result_char == "D":
            st.write(answ[answ['character_results']=='D']['description'].values[0])
        else:
            char_data = answ[answ['name']==result_char]
            st.header(f"Ты — {result_char}!")
            st.write(char_data["description"].values[0])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Пройти этот тест снова", type="primary"):

                st.session_state.quiz_started = False
                st.session_state.current_question = 0
                st.session_state.answers = []
                st.session_state.quiz_completed = False
                st.rerun()

        with col2:
            if st.button("📋 Выбрать другой тест", type="secondary"):
                keys_to_delete = ["quiz_started", "current_question", "answers", "quiz_completed"]
                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]
                st.query_params.clear()
                st.rerun()

#скачиваем датасет с тестом
demo_tests=pd.read_csv('test_demo.csv')
answer_test=pd.read_csv('answers.csv')
demo_tests['options'] = demo_tests['options'].apply(
    lambda options_str: ast.literal_eval(options_str)
)
demo_tests['letters'] = demo_tests['letters'].apply(
    lambda options_str: ast.literal_eval(options_str)
)

# ------------------------------
# Настройки страницы
# ------------------------------
st.set_page_config(page_title="Bobik", page_icon="🐶")
page = st.sidebar.selectbox("Навигация", ["Bobik", "Аналитика", "Тесты"])



# page = st.session_state.get("page", "Bobik")
logger.add("log/st.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")


st.sidebar.image("images/dog.png")
st.sidebar.markdown("## Найди ответ на любой вопрос")
st.sidebar.markdown("Бобик - твой личный ассистент в мире финансов. Он поможет тебе найти первую работу, накопить на мечту, научиться управлять карманными деньгами и защищаться от мошенников. ")
st.sidebar.markdown("Скорее задавай свой вопрос! Он точно знает на него ответ! 🤓 ")
# ------------------------------
# Кеширование индекса FAISS
# ------------------------------
@st.cache_data
def load_index():
    return sr.indexed_df()

df = load_index()

# ------------------------------
# Универсальная функция ответа
# ------------------------------
def answer_question(question: str, age_group: str = None, max_retries: int = 2):
    inputs = {
        "question": question,
        "max_retries": max_retries,
        "age_group": age_group,
        "df": df
    }
    final_answer = "Ошибка: ответ не сгенерирован"
    tags = []
    for event in sr.graph.stream(inputs, stream_mode="values"):
        if "generation" in event:
            gen = event["generation"]
            if hasattr(gen, "content"):
                final_answer = gen.content
            elif isinstance(gen, str):
                final_answer = gen
            else:
                final_answer = str(gen)
        if "relevant_tags" in event:
            tags = event["relevant_tags"]
    return final_answer, tags

# ------------------------------
# 1. Чат
# ------------------------------



if page == "Bobik":
    st.title(" 🐶 Бобик - помощник в твоих самых важных делах ")

    TAGS = [
        'Экономика', 'Подработка', 'Банковские карты', 'Банковские вклады и кредиты',
        'Инвестиции', 'Покупки', 'Жилье', 'Учебные материалы', 'Карманные деньги',
        'Мошенники', 'Экономия', 'Сбережения', 'Налоги', 'Права', 'Документы',
        'Финансовые цели', 'Иное'
    ]


    TAGS = sorted(TAGS)


    if "chosen_tags" not in st.session_state:
        st.session_state.chosen_tags = []

    with st.container():
        col_age, col_tags = st.columns([1, 2])

        with col_age:
            age_group = st.segmented_control(
                "Возраст:",
                options=["10–14", "14–18"],
                default="14–18",
                key="age_control"
            )

        with col_tags:
            selected = st.multiselect(
                "Теги (до 3):",
                TAGS,
                default=st.session_state.chosen_tags,
                max_selections=3,
                placeholder="Выбери до трёх тегов",
                key="tag_control"
            )


            if len(selected) > 3:
                st.warning("Можно выбрать не более трёх тегов.")
                selected = selected[:3]


            st.session_state.chosen_tags = selected

    if "questions_history" not in st.session_state:
        st.session_state.questions_history = []
    question_input = st.chat_input("Задай свой вопрос - постараюсь найти на него ответ:", key="input_text_field")

    if question_input:

        st.session_state.questions_history.append({
            "question": question_input,
            "tags": st.session_state.chosen_tags.copy(),
            "timestamp": datetime.datetime.now()
        })

        st.chat_message("user", avatar="images/club-penguin.gif").markdown(question_input)
        with st.chat_message("assistant", avatar="images/dog.png"):
            response_placeholder = st.empty()
            response_placeholder.markdown("_Так, дай мне немного подумать..._")


            answer, tags = answer_question(question_input,age_group)
            add_tags_to_session(tags)
            response_placeholder.markdown(f"Bobik: {answer}")

            stats = get_tag_stats()


            if tags:
                for t in tags:
                    if stats.get(t, 0) >= 3:
                        st.warning(
                            f"О, ты уже 3 раза спросил про **{t}**! Скорее переходи на вкладку 'Тесты' и проверяй свои знания!?"
                        )



# 2. Аналитика
# ------------------------------
elif page == "Аналитика":
    st.title("📊 Аналитика твоих вопросов")


    if st.button("Сбросить историю вопросов"):
        reset_tag_history()
        st.session_state.questions_history = []
        st.session_state.chosen_tags = []
        st.stop()

    stats = get_tag_stats()


    if stats:
        df_stats = [{"tag": k, "count": v} for k, v in stats.items()]
        custom_blue_scale = [

            [0, "#e0e1dd"],
            [0.25, "#778da9"],
            [0.5, "#415a77"],
            [0.75, "#1b263b"],
            [1, "#0d1b2a"]
        ]

        fig = px.bar(
            df_stats,
            x="tag",
            y="count",
            text="count",
            color="count",
            color_continuous_scale=custom_blue_scale,
            title="📊 На какие темы мы чаще всего общались? ",
        )
        fig.update_layout(
            yaxis=dict(title="Количество вопросов"),
            xaxis=dict(title="Темы", tickangle=-45),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=14)
        )
        st.plotly_chart(fig,width='content')
    else:
        st.info("Пока нет истории тегов.")


    st.subheader("📝 История твоих вопросов")
    if st.session_state.questions_history:
        import pandas as pd

        df_questions = pd.DataFrame(st.session_state.questions_history)
        st.dataframe(df_questions[['question','timestamp']].sort_values("timestamp", ascending=False),width='content')
    else:
        st.info("Пока нет истории вопросов.")


# ------------------------------
# 3. Тесты
# ------------------------------
elif page == "Тесты":
    st.title("📝 Тесты по финансовой грамотности")


    stats = get_tag_stats()
    most_common_tag = None
    if stats:
        most_common_tag = max(stats.items(), key=lambda x: x[1])[0]
    tag = most_common_tag



    # if tag:
    #     st.subheader(f"Тест по теме: {tag}")
    #     st.info(f"Здесь будет тест по теме '{tag}' (в разработке)")


    st.subheader("🎯 Выбери тест")

    test_option = st.radio(
        "Доступные тесты:",
        ["Тест сюрприз!", "Тест на основе твоих интересов."],
        key="test_selection"
    )

    if test_option in ["Тест сюрприз!", "Тест на основе твоих интересов."]:
        if test_option == "Тест сюрприз!":
            run_quiz(tag=None, demo_test=demo_tests, answ=answer_test)
        else:  # "Тест на основе твоих интересов"
            if tag is None or tag not in ['Сбережения']:
                st.info("🚧 Упс... Тебе пока, что недоступен данный тест. Продолжай общаться со своим помощником - Бобиком и ты обязательно узнаешь, что это за тест!")
            else:
                run_quiz(tag=tag, demo_test=demo_tests, answ=answer_test)
    else:
        st.info("🚧 Этот тест скоро появится! Сейчас пройди один из тестов, что тебе доступен - они очень интересные!")


        if st.button("Перейти к тесту сюрприз.", type="primary"):
            # Сбрасываем состояние теста
            if "quiz_started" in st.session_state:
                del st.session_state.quiz_started
            if "current_question" in st.session_state:
                del st.session_state.current_question
            if "answers" in st.session_state:
                del st.session_state.answers
            if "quiz_completed" in st.session_state:
                del st.session_state.quiz_completed
            st.rerun()