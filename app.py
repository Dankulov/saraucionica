import json
import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image


# =========================================================
# Configuration
# =========================================================

st.set_page_config(
    page_title="Moj tutor",
    page_icon="📚",
    layout="wide"
)

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

VISION_MODEL = st.secrets.get(
    "VISION_MODEL",
    "google/gemini-2.5-flash"
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# =========================================================
# App title
# =========================================================

st.title("📚 Moj tutor")

st.write(
    "Učitaj fotografije lekcije. "
    "Aplikacija će napraviti kratke beleške i test za proveru znanja."
)


# =========================================================
# Helpers
# =========================================================

def image_to_data_url(uploaded_file):
    """
    Convert uploaded image into a base64 data URL
    suitable for OpenRouter vision models.
    """

    image = Image.open(uploaded_file)

    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def call_openrouter(messages, temperature=0.2):
    """
    Generic OpenRouter API call.
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": VISION_MODEL,
        "messages": messages,
        "temperature": temperature
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=180
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter error {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    return data["choices"][0]["message"]["content"]


def extract_json(text):
    """
    Extract JSON from model response.
    Handles responses wrapped in markdown fences.
    """

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Model nije vratio validan JSON."
        )

    return json.loads(
        text[start:end + 1]
    )


# =========================================================
# Lesson analysis
# =========================================================

def analyse_lesson(uploaded_files):

    prompt = """
Ti si pomoćnik za učenje učeniku osnovne škole.

Na slikama se nalazi jedna lekcija iz udžbenika.

Pažljivo pročitaj sve stranice kao jednu celinu.

VAŽNO:
- koristi samo informacije koje postoje na slikama;
- ne dodaj činjenice iz sopstvenog znanja;
- ignoriši rukom napisane oznake;
- zadrži važne definicije i činjenice;
- piši jasno i jednostavno;
- koristi isti jezik kao u udžbeniku;
- beleške treba da budu kratke i pogodne za učenje.

Vrati SAMO validan JSON u sledećem obliku:

{
  "title": "naslov lekcije",
  "summary": "kratak pregled lekcije u 2 do 4 rečenice",
  "notes": [
    "kratka beleška 1",
    "kratka beleška 2"
  ],
  "concepts": [
    {
      "term": "pojam",
      "definition": "jednostavna definicija"
    }
  ],
  "facts": [
    "važna činjenica 1",
    "važna činjenica 2"
  ]
}
"""

    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    for uploaded_file in uploaded_files:

        data_url = image_to_data_url(
            uploaded_file
        )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            }
        )

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    response = call_openrouter(
        messages,
        temperature=0.1
    )

    return extract_json(response)


# =========================================================
# Quiz generation
# =========================================================

def generate_quiz(
    lesson,
    number_of_questions=5
):

    prompt = f"""
Ti si nastavnik koji pravi test za učenika osnovne škole.

Koristi ISKLJUČIVO sadržaj lekcije koji je dat ispod.

Ne koristi dodatno znanje.

Sadržaj lekcije:

{json.dumps(
    lesson,
    ensure_ascii=False,
    indent=2
)}

Napravi tačno {number_of_questions} pitanja.

Koristi kombinaciju sledećih tipova:

1. multiple_choice
2. true_false
3. short_answer

Pravila:

- pitanja treba da proveravaju najvažnije delove lekcije;
- neka pitanja proveravaju pamćenje;
- neka pitanja proveravaju razumevanje;
- izbegavaj trik pitanja;
- odgovor mora moći da se pronađe u sadržaju lekcije;
- za multiple_choice koristi tačno 4 ponuđena odgovora;
- samo jedan odgovor sme biti tačan.

Vrati SAMO validan JSON:

{{
  "questions": [
    {{
      "type": "multiple_choice",
      "question": "tekst pitanja",
      "options": [
        "odgovor A",
        "odgovor B",
        "odgovor C",
        "odgovor D"
      ],
      "correct_answer": "tačan odgovor",
      "explanation": "kratko objašnjenje"
    }},
    {{
      "type": "true_false",
      "question": "tvrdnja",
      "options": [
        "Tačno",
        "Netačno"
      ],
      "correct_answer": "Tačno",
      "explanation": "kratko objašnjenje"
    }},
    {{
      "type": "short_answer",
      "question": "pitanje",
      "options": [],
      "correct_answer": "očekivani kratak odgovor",
      "explanation": "kratko objašnjenje"
    }}
  ]
}}
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = call_openrouter(
        messages,
        temperature=0.4
    )

    return extract_json(response)


# =========================================================
# Session state
# =========================================================

if "lesson" not in st.session_state:
    st.session_state.lesson = None

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "current_answers" not in st.session_state:
    st.session_state.current_answers = {}


# =========================================================
# Upload section
# =========================================================

st.header("1. Učitaj lekciju")

uploaded_files = st.file_uploader(
    "Izaberi fotografije stranica",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    accept_multiple_files=True
)

if uploaded_files:

    st.write(
        f"Učitano stranica: "
        f"**{len(uploaded_files)}**"
    )

    number_of_columns = min(
        len(uploaded_files),
        4
    )

    columns = st.columns(
        number_of_columns
    )

    for i, file in enumerate(
        uploaded_files
    ):

        with columns[
            i % number_of_columns
        ]:

            st.image(
                file,
                caption=f"Strana {i + 1}",
                use_container_width=True
            )


# =========================================================
# Analyze lesson
# =========================================================

if uploaded_files:

    if st.button(
        "✨ Napravi beleške",
        type="primary"
    ):

        with st.spinner(
            "Čitam lekciju..."
        ):

            try:

                lesson = analyse_lesson(
                    uploaded_files
                )

                st.session_state.lesson = lesson
                st.session_state.quiz = None
                st.session_state.quiz_submitted = False
                st.session_state.current_answers = {}

            except Exception as e:

                st.error(
                    f"Greška: {e}"
                )


# =========================================================
# Display lesson
# =========================================================

lesson = st.session_state.lesson

if lesson:

    st.divider()

    st.header(
        f"2. {lesson.get('title', 'Lekcija')}"
    )

    # Summary

    st.subheader("Ukratko")

    st.write(
        lesson.get(
            "summary",
            ""
        )
    )

    # Notes

    notes = lesson.get(
        "notes",
        []
    )

    if notes:

        st.subheader(
            "📝 Kratke beleške"
        )

        for note in notes:

            st.write(
                f"• {note}"
            )

    # Concepts

    concepts = lesson.get(
        "concepts",
        []
    )

    if concepts:

        st.subheader(
            "🧠 Važni pojmovi"
        )

        for concept in concepts:

            term = concept.get(
                "term",
                ""
            )

            definition = concept.get(
                "definition",
                ""
            )

            st.markdown(
                f"**{term}:** {definition}"
            )

    # Facts

    facts = lesson.get(
        "facts",
        []
    )

    if facts:

        st.subheader(
            "⭐ Zapamti"
        )

        for fact in facts:

            st.write(
                f"• {fact}"
            )


# =========================================================
# Generate quiz
# =========================================================

if lesson:

    st.divider()

    st.header(
        "3. Proveri znanje"
    )

    number_of_questions = st.slider(
        "Broj pitanja",
        min_value=3,
        max_value=10,
        value=5
    )

    if st.button(
        "🎯 Napravi test"
    ):

        with st.spinner(
            "Pravim test..."
        ):

            try:

                quiz = generate_quiz(
                    lesson,
                    number_of_questions
                )

                st.session_state.quiz = quiz
                st.session_state.quiz_submitted = False
                st.session_state.current_answers = {}

            except Exception as e:

                st.error(
                    f"Greška: {e}"
                )


# =========================================================
# Display quiz
# =========================================================

quiz = st.session_state.quiz

if quiz:

    st.subheader(
        "Test"
    )

    answers = {}

    questions = quiz.get(
        "questions",
        []
    )

    for i, q in enumerate(
        questions
    ):

        st.markdown(
            f"### {i + 1}. "
            f"{q.get('question', '')}"
        )

        qtype = q.get(
            "type"
        )

        key = f"question_{i}"

        if qtype in [
            "multiple_choice",
            "true_false"
        ]:

            answers[i] = st.radio(
                "Izaberi odgovor:",
                q.get(
                    "options",
                    []
                ),
                key=key,
                index=None
            )

        elif qtype == "short_answer":

            answers[i] = st.text_input(
                "Tvoj odgovor:",
                key=key
            )

        st.write("")

    if st.button(
        "✅ Proveri test",
        type="primary"
    ):

        st.session_state.quiz_submitted = True
        st.session_state.current_answers = answers

        st.rerun()


# =========================================================
# Results
# =========================================================

if (
    quiz
    and st.session_state.quiz_submitted
):

    st.divider()

    st.header(
        "📊 Rezultat"
    )

    answers = st.session_state.current_answers

    questions = quiz.get(
        "questions",
        []
    )

    score = 0

    for i, q in enumerate(
        questions
    ):

        student_answer = answers.get(
            i
        )

        correct_answer = q.get(
            "correct_answer",
            ""
        )

        qtype = q.get(
            "type"
        )

        # -----------------------------------------
        # Simple grading for MVP
        # -----------------------------------------

        if not student_answer:

            is_correct = False

        elif qtype == "short_answer":

            is_correct = (
                student_answer
                .strip()
                .lower()
                ==
                correct_answer
                .strip()
                .lower()
            )

        else:

            is_correct = (
                student_answer
                ==
                correct_answer
            )

        # -----------------------------------------
        # Show result
        # -----------------------------------------

        st.markdown(
            f"**{i + 1}. "
            f"{q.get('question', '')}**"
        )

        if is_correct:

            score += 1

            st.success(
                "Tačno ✅"
            )

        else:

            st.error(
                "Netačno"
            )

            if student_answer:

                st.write(
                    f"Tvoj odgovor: "
                    f"**{student_answer}**"
                )

            else:

                st.write(
                    "Nisi unela odgovor."
                )

            st.write(
                f"Tačan odgovor: "
                f"**{correct_answer}**"
            )

        explanation = q.get(
            "explanation"
        )

        if explanation:

            st.caption(
                explanation
            )

        st.write("")

    # ---------------------------------------------
    # Overall score
    # ---------------------------------------------

    total = len(
        questions
    )

    percentage = (
        score / total * 100
        if total
        else 0
    )

    st.metric(
        "Rezultat",
        f"{score} / {total}"
    )

    st.progress(
        percentage / 100
    )

    if percentage == 100:

        st.success(
            "Odlično! Sve je tačno! 🎉"
        )

    elif percentage >= 80:

        st.success(
            "Odlično znaš lekciju! 🌟"
        )

    elif percentage >= 60:

        st.info(
            "Dobro ide. "
            "Još malo vežbanja. 🙂"
        )

    else:

        st.warning(
            "Vredi još jednom "
            "proći kroz beleške."
        )
