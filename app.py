import base64
import json
from io import BytesIO

import requests
import streamlit as st
from PIL import Image


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Moj tutor",
    page_icon="📚",
    layout="wide",
)

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

VISION_MODEL = st.secrets.get(
    "VISION_MODEL",
    "google/gemini-2.5-flash",
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# =========================================================
# APP HEADER
# =========================================================

st.title("📚 Moj tutor")

st.write(
    "Učitaj fotografije lekcije, napravi beleške, "
    "vežbaj pomoću flashcards, odgovaraj na pitanja "
    "i proveri znanje kroz test."
)


# =========================================================
# HELPERS
# =========================================================

def image_to_data_url(uploaded_file):
    """
    Convert Streamlit uploaded image to base64 data URL.
    """

    image = Image.open(uploaded_file)

    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90,
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
        "Content-Type": "application/json",
    }

    payload = {
        "model": VISION_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=180,
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
    Extract JSON from an LLM response.
    """

    if not isinstance(text, str):
        raise ValueError("Model nije vratio tekstualni odgovor.")

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```JSON", "")
        text = text.replace("```", "")
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Model nije vratio validan JSON."
        )

    json_text = text[start:end + 1]

    return json.loads(json_text)


# =========================================================
# LESSON ANALYSIS
# =========================================================

def analyse_lesson(uploaded_files):

    prompt = """
Ti si pomoćnik za učenje učeniku osnovne škole.

Na slikama se nalazi jedna lekcija iz udžbenika.

Pažljivo pročitaj SVE stranice i posmatraj ih kao jednu lekciju.

VAŽNA PRAVILA:

- koristi samo informacije koje postoje na slikama;
- nemoj dodavati činjenice iz svog opšteg znanja;
- ignoriši rukom napisane oznake;
- sačuvaj važne definicije;
- sačuvaj važne činjenice;
- obrati pažnju na tekst ispod slika;
- obrati pažnju na tekst u okvirima;
- obrati pažnju na pitanja na kraju lekcije;
- piši jednostavno, jasno i prilagođeno učeniku osnovne škole;
- koristi isti jezik kao lekcija;
- beleške treba da budu kratke i pogodne za učenje.

Vrati SAMO validan JSON.

Format:

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
            "text": prompt,
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
                },
            }
        )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    response = call_openrouter(
        messages,
        temperature=0.1,
    )

    return extract_json(response)


# =========================================================
# FLASHCARDS
# =========================================================

def generate_flashcards(
    lesson,
    number_of_cards=8,
):

    prompt = f"""
Ti si nastavnik koji pravi flashcards za učenika osnovne škole.

Koristi ISKLJUČIVO sadržaj lekcije koji je dat ispod.

Nemoj koristiti dodatno znanje.

LEKCIJA:

{json.dumps(
    lesson,
    ensure_ascii=False,
    indent=2
)}

Napravi tačno {number_of_cards} kartica.

PRAVILA:

- prednja strana je kratko pitanje;
- zadnja strana je kratak i jasan odgovor;
- jedna kartica proverava jednu ideju;
- obuhvati najvažnije pojmove;
- obuhvati najvažnije činjenice;
- ne pravi trik pitanja;
- ne pravi dva gotovo ista pitanja;
- odgovor mora postojati u sadržaju lekcije;
- odgovor treba da bude razumljiv detetu.

Vrati SAMO validan JSON:

{{
  "flashcards": [
    {{
      "question": "pitanje",
      "answer": "kratak odgovor"
    }}
  ]
}}
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = call_openrouter(
        messages,
        temperature=0.3,
    )

    return extract_json(response)


# =========================================================
# OPEN QUESTION
# =========================================================

def generate_open_question(
    lesson,
    previous_questions=None,
):

    previous_questions = previous_questions or []

    previous_text = json.dumps(
        previous_questions,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
Ti si nastavnik koji usmeno ispituje učenika osnovne škole.

Koristi ISKLJUČIVO sadržaj ove lekcije.

LEKCIJA:

{json.dumps(
    lesson,
    ensure_ascii=False,
    indent=2
)}

Pitanja koja su već postavljena:

{previous_text}

Postavi JEDNO NOVO otvoreno pitanje.

PRAVILA:

- pitanje treba da proverava važan deo lekcije;
- pitanje mora imati jasan odgovor;
- učenik treba da može da odgovori svojim rečima;
- odgovor može biti jedna ili nekoliko kratkih rečenica;
- nemoj postavljati trik pitanje;
- nemoj ponavljati prethodna pitanja;
- nemoj koristiti informacije kojih nema u lekciji.

Napravi i kratko objašnjenje koje ćemo pokazati učeniku
ako kaže da ne zna odgovor.

Objašnjenje treba:
- da bude jednostavno;
- da objasni ideju, a ne samo da navede odgovor;
- da bude prilagođeno detetu.

Vrati SAMO validan JSON:

{{
  "question": "otvoreno pitanje",
  "ideal_answer": "kratak očekivani odgovor",
  "explanation": "jednostavno objašnjenje"
}}
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = call_openrouter(
        messages,
        temperature=0.5,
    )

    return extract_json(response)


# =========================================================
# GRADE OPEN ANSWER
# =========================================================

def grade_open_answer(
    lesson,
    question,
    ideal_answer,
    student_answer,
):

    prompt = f"""
Ti si nastavnik osnovne škole.

Treba da proceniš odgovor učenika.

Koristi ISKLJUČIVO sadržaj lekcije.

LEKCIJA:

{json.dumps(
    lesson,
    ensure_ascii=False,
    indent=2
)}

PITANJE:

{question}

OČEKIVANI ODGOVOR:

{ideal_answer}

ODGOVOR UČENIKA:

{student_answer}

Proceni ZNAČENJE odgovora.

VAŽNO:

- učenik ne mora koristiti iste reči kao očekivani odgovor;
- prihvati drugačiju formulaciju ako je značenje tačno;
- prihvati kratak odgovor ako sadrži suštinu;
- sitne gramatičke i pravopisne greške ignoriši;
- ne kažnjavaj učenika zato što nije napisao nešto što pitanje nije tražilo;
- ako je glavna ideja tačna ali nešto važno nedostaje, odgovor je partial;
- ako odgovor pokazuje pogrešno razumevanje, odgovor je incorrect;
- feedback treba da bude kratak, prijatan i konkretan.

Status mora biti tačno jedan od:

correct
partial
incorrect

Vrati SAMO validan JSON:

{{
  "status": "correct",
  "feedback": "kratka povratna informacija učeniku",
  "explanation": "jednostavno objašnjenje tačnog odgovora"
}}
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = call_openrouter(
        messages,
        temperature=0.1,
    )

    return extract_json(response)


# =========================================================
# QUIZ
# =========================================================

def generate_quiz(
    lesson,
    number_of_questions=5,
):

    prompt = f"""
Ti si nastavnik koji pravi test za učenika osnovne škole.

Koristi ISKLJUČIVO sadržaj lekcije.

Nemoj koristiti dodatno znanje.

LEKCIJA:

{json.dumps(
    lesson,
    ensure_ascii=False,
    indent=2
)}

Napravi tačno {number_of_questions} pitanja.

Koristi kombinaciju:

- multiple_choice
- true_false
- short_answer

PRAVILA:

- proveri različite delove lekcije;
- obuhvati najvažnije pojmove;
- obuhvati najvažnije činjenice;
- neka neka pitanja proveravaju pamćenje;
- neka neka pitanja proveravaju razumevanje;
- ne pravi trik pitanja;
- ne ponavljaj isto pitanje različitim rečima;
- odgovor mora biti zasnovan na lekciji;
- za multiple_choice koristi tačno četiri ponuđena odgovora;
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
      "correct_answer": "očekivani odgovor",
      "explanation": "kratko objašnjenje"
    }}
  ]
}}
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = call_openrouter(
        messages,
        temperature=0.4,
    )

    return extract_json(response)


# =========================================================
# SESSION STATE
# =========================================================

default_state = {
    "lesson": None,

    "flashcards": None,
    "current_flashcard": 0,
    "show_flashcard_answer": False,
    "known_flashcards": [],
    "repeat_flashcards": [],

    "open_question": None,
    "open_question_result": None,
    "dont_know_explanation": False,
    "previous_open_questions": [],
    "open_question_version": 0,

    "quiz": None,
    "quiz_submitted": False,
    "current_answers": {},
    "quiz_version": 0,
}

for key, value in default_state.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# RESET LESSON DATA
# =========================================================

def reset_learning_state():

    st.session_state.flashcards = None
    st.session_state.current_flashcard = 0
    st.session_state.show_flashcard_answer = False
    st.session_state.known_flashcards = []
    st.session_state.repeat_flashcards = []

    st.session_state.open_question = None
    st.session_state.open_question_result = None
    st.session_state.dont_know_explanation = False
    st.session_state.previous_open_questions = []
    st.session_state.open_question_version += 1

    st.session_state.quiz = None
    st.session_state.quiz_submitted = False
    st.session_state.current_answers = {}
    st.session_state.quiz_version += 1


# =========================================================
# UPLOAD
# =========================================================

st.header("1. Učitaj lekciju")

uploaded_files = st.file_uploader(
    "Izaberi fotografije stranica",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
    accept_multiple_files=True,
)

if uploaded_files:

    st.write(
        f"Učitano stranica: "
        f"**{len(uploaded_files)}**"
    )

    number_of_columns = min(
        len(uploaded_files),
        4,
    )

    columns = st.columns(
        number_of_columns
    )

    for i, file in enumerate(uploaded_files):

        with columns[
            i % number_of_columns
        ]:

            st.image(
                file,
                caption=f"Strana {i + 1}",
                use_container_width=True,
            )


# =========================================================
# ANALYSE LESSON
# =========================================================

if uploaded_files:

    if st.button(
        "✨ Pročitaj lekciju",
        type="primary",
    ):

        with st.spinner(
            "Čitam lekciju..."
        ):

            try:

                lesson = analyse_lesson(
                    uploaded_files
                )

                st.session_state.lesson = lesson

                reset_learning_state()

                st.rerun()

            except Exception as e:

                st.error(
                    f"Greška: {e}"
                )


# =========================================================
# LESSON
# =========================================================

lesson = st.session_state.lesson

if lesson:

    st.divider()

    st.header(
        lesson.get(
            "title",
            "Lekcija",
        )
    )

    (
        tab_notes,
        tab_flashcards,
        tab_practice,
        tab_quiz,
    ) = st.tabs(
        [
            "📝 Beleške",
            "🧠 Flashcards",
            "💬 Pitaj me",
            "🎯 Test",
        ]
    )


    # =====================================================
    # TAB 1 — NOTES
    # =====================================================

    with tab_notes:

        st.subheader("Ukratko")

        st.write(
            lesson.get(
                "summary",
                ""
            )
        )

        notes = lesson.get(
            "notes",
            [],
        )

        if notes:

            st.subheader(
                "📝 Kratke beleške"
            )

            for note in notes:

                st.write(
                    f"• {note}"
                )

        concepts = lesson.get(
            "concepts",
            [],
        )

        if concepts:

            st.subheader(
                "🧠 Važni pojmovi"
            )

            for concept in concepts:

                term = concept.get(
                    "term",
                    "",
                )

                definition = concept.get(
                    "definition",
                    "",
                )

                st.markdown(
                    f"**{term}:** {definition}"
                )

        facts = lesson.get(
            "facts",
            [],
        )

        if facts:

            st.subheader(
                "⭐ Zapamti"
            )

            for fact in facts:

                st.write(
                    f"• {fact}"
                )


    # =====================================================
    # TAB 2 — FLASHCARDS
    # =====================================================

    with tab_flashcards:

        st.subheader(
            "🧠 Flashcards"
        )

        if st.session_state.flashcards is None:

            st.write(
                "Vežbaj najvažnije pojmove i činjenice "
                "iz lekcije."
            )

            number_of_cards = st.slider(
                "Broj kartica",
                min_value=5,
                max_value=15,
                value=8,
                key="flashcard_count",
            )

            if st.button(
                "✨ Napravi flashcards",
                key="generate_flashcards",
            ):

                with st.spinner(
                    "Pravim kartice..."
                ):

                    try:

                        flashcards = generate_flashcards(
                            lesson,
                            number_of_cards,
                        )

                        st.session_state.flashcards = flashcards
                        st.session_state.current_flashcard = 0
                        st.session_state.show_flashcard_answer = False
                        st.session_state.known_flashcards = []
                        st.session_state.repeat_flashcards = []

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Greška: {e}"
                        )

        else:

            flashcards = (
                st.session_state.flashcards.get(
                    "flashcards",
                    [],
                )
            )

            total_cards = len(
                flashcards
            )

            if total_cards == 0:

                st.warning(
                    "Nisu generisane kartice."
                )

            else:

                index = (
                    st.session_state.current_flashcard
                )

                index = max(
                    0,
                    min(
                        index,
                        total_cards - 1,
                    ),
                )

                st.session_state.current_flashcard = index

                card = flashcards[index]

                st.caption(
                    f"Kartica {index + 1} "
                    f"od {total_cards}"
                )

                st.progress(
                    (index + 1)
                    / total_cards
                )

                st.markdown("---")

                st.markdown(
                    f"### ❓ "
                    f"{card.get('question', '')}"
                )

                st.write("")

                if not st.session_state.show_flashcard_answer:

                    if st.button(
                        "👀 Prikaži odgovor",
                        key=f"show_card_{index}",
                    ):

                        st.session_state.show_flashcard_answer = True

                        st.rerun()

                else:

                    st.success(
                        card.get(
                            "answer",
                            "",
                        )
                    )

                    know_col, repeat_col = st.columns(2)

                    with know_col:

                        if st.button(
                            "✅ Znam",
                            use_container_width=True,
                            key=f"know_card_{index}",
                        ):

                            if index not in st.session_state.known_flashcards:

                                st.session_state.known_flashcards.append(
                                    index
                                )

                            if index in st.session_state.repeat_flashcards:

                                st.session_state.repeat_flashcards.remove(
                                    index
                                )

                            if index < total_cards - 1:

                                st.session_state.current_flashcard += 1

                            st.session_state.show_flashcard_answer = False

                            st.rerun()

                    with repeat_col:

                        if st.button(
                            "🔁 Ponovi",
                            use_container_width=True,
                            key=f"repeat_card_{index}",
                        ):

                            if index not in st.session_state.repeat_flashcards:

                                st.session_state.repeat_flashcards.append(
                                    index
                                )

                            if index in st.session_state.known_flashcards:

                                st.session_state.known_flashcards.remove(
                                    index
                                )

                            if index < total_cards - 1:

                                st.session_state.current_flashcard += 1

                            st.session_state.show_flashcard_answer = False

                            st.rerun()

                st.markdown("---")

                previous_col, next_col = st.columns(2)

                with previous_col:

                    if st.button(
                        "⬅️ Prethodna",
                        disabled=index == 0,
                        use_container_width=True,
                        key="previous_flashcard",
                    ):

                        st.session_state.current_flashcard -= 1
                        st.session_state.show_flashcard_answer = False

                        st.rerun()

                with next_col:

                    if st.button(
                        "Sledeća ➡️",
                        disabled=index == total_cards - 1,
                        use_container_width=True,
                        key="next_flashcard",
                    ):

                        st.session_state.current_flashcard += 1
                        st.session_state.show_flashcard_answer = False

                        st.rerun()

                known_count = len(
                    st.session_state.known_flashcards
                )

                repeat_count = len(
                    st.session_state.repeat_flashcards
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Kartica",
                    f"{index + 1}/{total_cards}",
                )

                col2.metric(
                    "✅ Znam",
                    known_count,
                )

                col3.metric(
                    "🔁 Ponoviti",
                    repeat_count,
                )

                if repeat_count:

                    st.info(
                        f"Imaš {repeat_count} "
                        f"kartica za ponavljanje."
                    )

                st.write("")

                if st.button(
                    "🔄 Napravi nove kartice",
                    key="new_flashcards",
                ):

                    st.session_state.flashcards = None
                    st.session_state.current_flashcard = 0
                    st.session_state.show_flashcard_answer = False
                    st.session_state.known_flashcards = []
                    st.session_state.repeat_flashcards = []

                    st.rerun()


    # =====================================================
    # TAB 3 — ASK ME
    # =====================================================

    with tab_practice:

        st.subheader(
            "💬 Pitaj me"
        )

        st.write(
            "Odgovori svojim rečima. "
            "Ako ne znaš odgovor, klikni na **Ne znam** "
            "i dobićeš objašnjenje."
        )

        if st.session_state.open_question is None:

            if st.button(
                "🎓 Postavi mi pitanje",
                type="primary",
                key="start_open_question",
            ):

                with st.spinner(
                    "Smisliću pitanje..."
                ):

                    try:

                        question = generate_open_question(
                            lesson,
                            st.session_state.previous_open_questions,
                        )

                        st.session_state.open_question = question
                        st.session_state.open_question_result = None
                        st.session_state.dont_know_explanation = False
                        st.session_state.open_question_version += 1

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Greška: {e}"
                        )

        else:

            current_question = (
                st.session_state.open_question
            )

            st.markdown("---")

            st.markdown(
                f"### ❓ "
                f"{current_question.get('question', '')}"
            )

            answer_key = (
                f"open_answer_"
                f"{st.session_state.open_question_version}"
            )

            student_answer = st.text_area(
                "Tvoj odgovor:",
                placeholder=(
                    "Napiši odgovor svojim rečima..."
                ),
                key=answer_key,
                height=120,
            )

            check_col, dont_know_col = st.columns(2)

            with check_col:

                check_answer = st.button(
                    "✅ Proveri odgovor",
                    type="primary",
                    use_container_width=True,
                    disabled=not student_answer.strip(),
                    key=(
                        f"check_open_"
                        f"{st.session_state.open_question_version}"
                    ),
                )

            with dont_know_col:

                dont_know = st.button(
                    "🤷 Ne znam",
                    use_container_width=True,
                    key=(
                        f"dont_know_"
                        f"{st.session_state.open_question_version}"
                    ),
                )

            if check_answer:

                with st.spinner(
                    "Proveravam odgovor..."
                ):

                    try:

                        result = grade_open_answer(
                            lesson=lesson,
                            question=current_question.get(
                                "question",
                                "",
                            ),
                            ideal_answer=current_question.get(
                                "ideal_answer",
                                "",
                            ),
                            student_answer=student_answer,
                        )

                        st.session_state.open_question_result = result
                        st.session_state.dont_know_explanation = False

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Greška: {e}"
                        )

            if dont_know:

                st.session_state.dont_know_explanation = True
                st.session_state.open_question_result = None

                st.rerun()

            # ---------------------------------------------
            # DON'T KNOW
            # ---------------------------------------------

            if st.session_state.dont_know_explanation:

                st.warning(
                    "Nema problema — hajde da objasnimo. 🙂"
                )

                st.info(
                    current_question.get(
                        "explanation",
                        "",
                    )
                )

                st.markdown(
                    "**Kratak odgovor:**"
                )

                st.success(
                    current_question.get(
                        "ideal_answer",
                        "",
                    )
                )

            # ---------------------------------------------
            # GRADED ANSWER
            # ---------------------------------------------

            result = (
                st.session_state.open_question_result
            )

            if result:

                status = result.get(
                    "status",
                    "",
                )

                if status == "correct":

                    st.success(
                        "Odlično! Tačno. ✅"
                    )

                elif status == "partial":

                    st.warning(
                        "Skoro tačno. 🟡"
                    )

                else:

                    st.error(
                        "Ovaj odgovor još nije sasvim tačan."
                    )

                feedback = result.get(
                    "feedback",
                    "",
                )

                if feedback:

                    st.write(
                        feedback
                    )

                if status != "correct":

                    explanation = result.get(
                        "explanation",
                        "",
                    )

                    if explanation:

                        st.info(
                            explanation
                        )

            # ---------------------------------------------
            # NEXT QUESTION
            # ---------------------------------------------

            if (
                st.session_state.dont_know_explanation
                or st.session_state.open_question_result
            ):

                st.write("")

                if st.button(
                    "➡️ Sledeće pitanje",
                    type="primary",
                    key=(
                        f"next_open_"
                        f"{st.session_state.open_question_version}"
                    ),
                ):

                    with st.spinner(
                        "Smisliću novo pitanje..."
                    ):

                        try:

                            old_question = (
                                current_question.get(
                                    "question",
                                    "",
                                )
                            )

                            if old_question:

                                st.session_state.previous_open_questions.append(
                                    old_question
                                )

                            new_question = generate_open_question(
                                lesson,
                                st.session_state.previous_open_questions,
                            )

                            st.session_state.open_question = new_question
                            st.session_state.open_question_result = None
                            st.session_state.dont_know_explanation = False
                            st.session_state.open_question_version += 1

                            st.rerun()

                        except Exception as e:

                            st.error(
                                f"Greška: {e}"
                            )


    # =====================================================
    # TAB 4 — QUIZ
    # =====================================================

    with tab_quiz:

        st.subheader(
            "🎯 Test"
        )

        st.write(
            "Napravi test za proveru znanja iz cele lekcije."
        )

        if st.session_state.quiz is None:

            number_of_questions = st.slider(
                "Broj pitanja",
                min_value=3,
                max_value=10,
                value=5,
                key="quiz_question_count",
            )

            if st.button(
                "🎯 Napravi test",
                type="primary",
                key="generate_quiz",
            ):

                with st.spinner(
                    "Pravim test..."
                ):

                    try:

                        quiz = generate_quiz(
                            lesson,
                            number_of_questions,
                        )

                        st.session_state.quiz = quiz
                        st.session_state.quiz_submitted = False
                        st.session_state.current_answers = {}
                        st.session_state.quiz_version += 1

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Greška: {e}"
                        )

        else:

            quiz = st.session_state.quiz

            questions = quiz.get(
                "questions",
                [],
            )

            answers = {}

            st.markdown("---")

            for i, q in enumerate(
                questions
            ):

                st.markdown(
                    f"### {i + 1}. "
                    f"{q.get('question', '')}"
                )

                qtype = q.get(
                    "type",
                    "",
                )

                key = (
                    f"quiz_"
                    f"{st.session_state.quiz_version}_"
                    f"{i}"
                )

                if qtype in [
                    "multiple_choice",
                    "true_false",
                ]:

                    answers[i] = st.radio(
                        "Izaberi odgovor:",
                        q.get(
                            "options",
                            [],
                        ),
                        index=None,
                        key=key,
                    )

                elif qtype == "short_answer":

                    answers[i] = st.text_input(
                        "Tvoj odgovor:",
                        key=key,
                    )

                st.write("")

            if not st.session_state.quiz_submitted:

                if st.button(
                    "✅ Proveri test",
                    type="primary",
                    key="submit_quiz",
                ):

                    st.session_state.current_answers = answers
                    st.session_state.quiz_submitted = True

                    st.rerun()

            # =============================================
            # QUIZ RESULTS
            # =============================================

            if st.session_state.quiz_submitted:

                st.divider()

                st.subheader(
                    "📊 Rezultat"
                )

                saved_answers = (
                    st.session_state.current_answers
                )

                score = 0

                for i, q in enumerate(
                    questions
                ):

                    student_answer = saved_answers.get(
                        i
                    )

                    correct_answer = q.get(
                        "correct_answer",
                        "",
                    )

                    qtype = q.get(
                        "type",
                        "",
                    )

                    if not student_answer:

                        is_correct = False

                    elif qtype == "short_answer":

                        # Simple matching for MVP.
                        # Later we can replace this
                        # with semantic LLM grading.
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
                                "Tvoj odgovor: "
                                f"**{student_answer}**"
                            )

                        else:

                            st.write(
                                "Nisi unela odgovor."
                            )

                        st.write(
                            "Tačan odgovor: "
                            f"**{correct_answer}**"
                        )

                    explanation = q.get(
                        "explanation",
                        "",
                    )

                    if explanation:

                        st.caption(
                            explanation
                        )

                    st.write("")

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
                    f"{score} / {total}",
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
                        "proći kroz lekciju."
                    )

                st.write("")

                if st.button(
                    "🔄 Napravi novi test",
                    key="new_quiz",
                ):

                    st.session_state.quiz = None
                    st.session_state.quiz_submitted = False
                    st.session_state.current_answers = {}
                    st.session_state.quiz_version += 1

                    st.rerun()
