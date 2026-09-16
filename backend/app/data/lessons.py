"""Single source of truth for lessons (Kannada text, romanization, syllables)."""

LESSONS: dict[str, dict] = {
    "w01": {
        "text": "ನಮಸ್ತೆ",
        "roman": "Namaste",
        "english": "Hello / Greetings",
        "syllables": ["na", "mas", "te"],
    },
    "w02": {
        "text": "ನಿಮ್ಮ ಹೆಸರೇನು",
        "roman": "Nimma hesarenu",
        "english": "What is your name?",
        "syllables": ["nim", "ma", "he", "sa", "re", "nu"],
    },
    "w03": {
        "text": "ನೀವು ಹೇಗಿದ್ದೀರ",
        "roman": "Neevu hengiddira",
        "english": "How are you?",
        "syllables": ["nee", "vu", "heng", "id", "di", "ra"],
    },
    "w04": {
        "text": "ನಾನು ಚೆನ್ನಾಗಿದ್ದೀನಿ",
        "roman": "Naanu chennagiddini",
        "english": "I am fine.",
        "syllables": ["naa", "nu", "chen", "na", "gi", "di", "ni"],
    },
    "w05": {
        "text": "ಅದು ಚೆನ್ನಾಗಿದೆ",
        "roman": "Adu chennagide",
        "english": "That is good.",
        "syllables": ["a", "du", "chen", "na", "gi", "de"],
    },
    "w06": {
        "text": "ಧನ್ಯವಾದ",
        "roman": "Dhanyavaada",
        "english": "Thank you.",
        "syllables": ["dha", "nya", "vaa", "da"],
    },
    "w07": {
        "text": "ಇದು ಎಷ್ಟು",
        "roman": "Idu eshtu",
        "english": "How much is this?",
        "syllables": ["i", "du", "es", "htu"],
    },
    "w08": {
        "text": "ಕಮ್ಮಿ ಮಾಡಿ",
        "roman": "Kammi maadi",
        "english": "Reduce it (make it less).",
        "syllables": ["kam", "mi", "maa", "di"],
    },
    "w09": {
        "text": "ದಯವಿಟ್ಟು ಸಹಾಯ ಮಾಡಿ",
        "roman": "Dayavittu sahaya maadi",
        "english": "Please help.",
        "syllables": ["da", "ya", "vit", "tu", "sa", "ha", "ya", "maa", "di"],
    },
    "w10": {
        "text": "ಅಲ್ಲಿಗೆ ಹೇಗೆ ಹೋಗೋದು",
        "roman": "Alige hege hogodu",
        "english": "How do I go there?",
        "syllables": ["a", "li", "ge", "he", "ge", "ho", "go", "du"],
    },
    "w11": {
        "text": "ನಾನು ಕರ್ನಾಟಕದಲ್ಲಿ ಇದ್ದೀನಿ",
        "roman": "Naanu Karnataka dalli iddini",
        "english": "I am in Karnataka.",
        "syllables": ["naa", "nu", "kar", "na", "ta", "ka", "dal", "li", "id", "di", "ni"],
    },
    "w12": {
        "text": "ನಾನು ಕನ್ನಡ ಕಲಿತ ಇದ್ದೀನಿ",
        "roman": "Naanu kannada kalita iddini",
        "english": "I am learning Kannada.",
        "syllables": ["naa", "nu", "kan", "na", "da", "ka", "li", "ta", "id", "di", "ni"],
    },
    "w13": {
        "text": "ಊಟ ಆಯತ",
        "roman": "Oota ayata",
        "english": "Had your meal?",
        "syllables": ["oo", "ta", "ya", "ta"],
    },
    "w14": {
        "text": "ನನಗೆ ಗೊತ್ತಿಲ್ಲ",
        "roman": "Nanage gothilla",
        "english": "I don't know.",
        "syllables": ["na", "na", "ge", "go", "thi", "la"],
    },
    "w15": {
        "text": "ಮನೆಗೆ ಬನ್ನಿ",
        "roman": "Manege banni",
        "english": "Come home.",
        "syllables": ["ma", "ne", "ge", "ban", "ni"],
    },
}


def get_lesson(word_id: str) -> dict:
    if word_id not in LESSONS:
        raise KeyError(word_id)
    return LESSONS[word_id]


def lesson_order(word_id: str) -> int:
    try:
        return int(word_id[1:])
    except (ValueError, IndexError):
        return 0


def difficulty_for(word_id: str) -> str:
    order = lesson_order(word_id)
    if order <= 5:
        return "beginner"
    if order <= 10:
        return "intermediate"
    return "advanced"


def serialize_lesson(word_id: str) -> dict:
    data = LESSONS[word_id]
    return {
        "id": word_id,
        "order": lesson_order(word_id),
        "title": data["text"],
        "kannada_text": data["text"],
        "transliteration": data["roman"],
        "english_translation": data["english"],
        "syllables": list(data["syllables"]),
        "difficulty": difficulty_for(word_id),
    }
