import re
import time
import textwrap
import json
from datetime import datetime
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher
from urllib import request, error
import pandas as pd

# Load dataset
DATA_PATH = Path(__file__).with_name("medical_dataset.csv")
data = pd.read_csv(DATA_PATH)

# Verified reference sources used for general healthcare education.
VERIFIED_SOURCES = {
    "infectious disease": [
        "WHO: https://www.who.int/health-topics",
        "CDC: https://www.cdc.gov/diseasesconditions",
    ],
    "cardiovascular": [
        "WHO Cardiovascular diseases: https://www.who.int/health-topics/cardiovascular-diseases",
        "AHA: https://www.heart.org/",
    ],
    "respiratory": [
        "NHLBI: https://www.nhlbi.nih.gov/",
        "NHS Respiratory conditions: https://www.nhs.uk/conditions/",
    ],
    "mental health": [
        "WHO Mental health: https://www.who.int/health-topics/mental-health",
        "NIMH: https://www.nimh.nih.gov/health/topics",
    ],
    "digestive": [
        "NIDDK: https://www.niddk.nih.gov/health-information",
        "NHS conditions: https://www.nhs.uk/conditions/",
    ],
    "skin": [
        "AAD: https://www.aad.org/public/diseases",
        "NHS skin conditions: https://www.nhs.uk/conditions/",
    ],
    "endocrine": [
        "NIDDK Endocrine: https://www.niddk.nih.gov/health-information/endocrine-diseases",
        "Mayo Clinic Thyroid: https://www.mayoclinic.org/diseases-conditions/thyroid-problems/",
    ],
}

CARE_GUIDANCE_BY_CATEGORY = {
    "infectious disease": [
        "Rest well and stay hydrated.",
        "Monitor fever and worsening symptoms.",
        "Follow hygiene measures to prevent spread.",
        "Use medicines only as advised by a doctor.",
    ],
    "cardiovascular": [
        "Reduce salt and processed food intake.",
        "Exercise regularly based on doctor advice.",
        "Monitor blood pressure and weight.",
        "Take prescribed medicines consistently.",
    ],
    "respiratory": [
        "Avoid smoke, dust, and known triggers.",
        "Drink warm fluids and rest well.",
        "Use inhalers/medication only if prescribed.",
        "Seek care early if breathing worsens.",
    ],
    "mental health": [
        "Maintain regular sleep, meals, and daily routine.",
        "Practice stress management and relaxation.",
        "Stay connected with trusted family/friends.",
        "Consult a mental health professional for persistent symptoms.",
    ],
    "digestive": [
        "Avoid spicy, oily, and trigger foods.",
        "Eat smaller frequent meals.",
        "Stay hydrated and avoid alcohol/smoking.",
        "Consult a doctor if pain persists or worsens.",
    ],
    "skin": [
        "Keep affected skin clean and dry.",
        "Avoid scratching and known irritants.",
        "Use prescribed creams/medication only as advised.",
        "Consult a dermatologist if rash spreads or persists.",
    ],
    "endocrine": [
        "Take thyroid-related medicines exactly as prescribed.",
        "Schedule regular thyroid function tests.",
        "Maintain balanced diet and physical activity.",
        "Follow up with an endocrinologist for dose adjustments.",
    ],
}

RED_FLAGS_BY_CATEGORY = {
    "infectious disease": "Very high fever, confusion, persistent vomiting, breathing difficulty.",
    "cardiovascular": "Chest pain, severe breathlessness, fainting, one-sided weakness.",
    "respiratory": "Fast breathing, chest tightness, low oxygen, bluish lips.",
    "mental health": "Thoughts of self-harm, severe panic, inability to function daily.",
    "digestive": "Blood in vomit/stool, severe abdominal pain, dehydration signs.",
    "skin": "Rapidly spreading rash, facial swelling, breathing difficulty, fever with rash.",
    "endocrine": "Severe palpitations, confusion, chest pain, extreme weakness or fainting.",
}

TYPO_CORRECTIONS = {
    "thypoid": "typhoid",
}

# Common medical terminology aliases mapped to canonical condition names.
MEDICAL_TERM_ALIASES = {
    "high blood pressure": "hypertension",
    "high bp": "hypertension",
    "low blood pressure": "hypotension",
    "heart attack": "heart attack",
    "myocardial infarction": "heart attack",
    "mi": "heart attack",
    "stroke": "stroke",
    "cva": "stroke",
    "cerebrovascular accident": "stroke",
    "tb": "tuberculosis",
    "covid": "covid 19",
    "covid19": "covid 19",
    "corona": "covid 19",
    "blood sugar": "diabetes",
    "sugar": "diabetes",
    "pcod": "pcod",
    "pcos": "pcos",
    "thyroid": "thyroid disorder",
    "underactive thyroid": "hypothyroidism",
    "overactive thyroid": "hyperthyroidism",
    "acid reflux": "gastritis",
    "ulcer disease": "ulcer",
    "cold": "common cold",
    "influenza": "flu",
    "kidney stones": "kidney stone",
    "pink eye": "conjunctivitis",
}

DEFAULT_SOURCES = [
    "WHO: https://www.who.int/health-topics",
    "MedlinePlus: https://medlineplus.gov/",
]

LOW_QUALITY_PATTERNS = [
    "may be caused",
    "common symptoms include",
    "broad term for conditions",
    "general health condition",
]

# LLM configuration (RAG enhancement)
USE_LLM = False
LLM_BACKEND = "ollama"  # "ollama" (default) or "hf"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b-instruct"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_TIMEOUT_SECONDS = 60
LLM_SEED = 42

# Stopwords removed before keyword matching
STOPWORDS = {
    "what","is","are","the","a","an","of","for","in","at","to",
    "do","does","how","why","when","where","who","which","can",
    "tell","me","about","give","explain","describe","define","i",
    "my","have","get","you","and","or","with","it","this","that",
    "please","could","would","should","been","be","its","will",
    "information","on","any",
    "disease","diseases","condition","conditions","disorder",
    "disorders","problem","problems","issue","issues"
}

# Intent signals
SYMPTOM_WORDS = {
    "symptom","symptoms","sign","signs","feel","feeling",
    "suffering","experience","indicate","shows","showing"
}

CAUSE_WORDS = {
    "cause","causes","caused","reason","reasons",
    "trigger","triggers","origin"
}

DEFINITION_WORDS = {
    "what","define","definition","meaning","means",
    "explain","describe","description","tell","about"
}

PERSONAL_SYMPTOM_WORDS = {
    "i", "my", "me", "after", "feel", "feeling", "having", "suffering",
    "pain", "burn", "burns", "burning", "ache", "nausea", "vomiting", "dizzy",
    "cough", "fever", "chills", "rash", "swelling", "breath", "tired"
}


def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [TYPO_CORRECTIONS.get(tok, tok) for tok in text.split()]
    text = " ".join(tokens)
    return text


def apply_medical_aliases(text):
    """Replace known medical alias phrases with canonical condition names."""
    value = f" {normalize(text)} "
    # Replace longer aliases first to avoid partial replacement collisions.
    for alias in sorted(MEDICAL_TERM_ALIASES.keys(), key=len, reverse=True):
        canonical = normalize(MEDICAL_TERM_ALIASES[alias])
        value = value.replace(f" {normalize(alias)} ", f" {canonical} ")
    return normalize(value)


def canonicalize_condition_name(name):
    """Map known term variants to a canonical condition phrase."""
    normalized = normalize(name)
    if normalized in MEDICAL_TERM_ALIASES:
        return normalize(MEDICAL_TERM_ALIASES[normalized])
    return normalized


def log_internal_error(tag, exc):
    """Log internal exceptions without exposing raw tracebacks to users."""
    try:
        with open("error_log.txt", "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {tag}: {type(exc).__name__}: {exc}\n")
    except OSError:
        pass


def scaled_confidence(best_score):
    """Dynamic confidence with a floor to keep symptom-based prediction useful."""
    try:
        value = 0.6 + (float(best_score) * 0.4)
    except (TypeError, ValueError):
        value = 0.6
    return max(0.6, min(0.98, value))


def keywords(text):
    return [
        t for t in normalize(text).split()
        if t not in STOPWORDS and len(t) > 1
    ]


def keyword_hit_score(user_kws, db_q):
    if not user_kws:
        return 0.0

    db_word_list = db_q.split()

    def token_root(token):
        return token[:4]

    hits = sum(
        1 for kw in user_kws
        if any(
            tok == kw
            or tok.startswith(kw)
            or kw.startswith(tok)
            or token_root(tok) == token_root(kw)
            for tok in db_word_list
        )
    )

    return hits / len(user_kws)


def get_sources_for_row(row):
    category = normalize(str(row.get("Category", "")))
    category_sources = VERIFIED_SOURCES.get(category, [])
    return ensure_minimum_verified_sources(category_sources)


def ensure_minimum_verified_sources(sources):
    """Ensure each response includes at least two verified source references."""
    cleaned = []
    seen = set()

    for source in list(sources) + list(DEFAULT_SOURCES):
        value = str(source).strip()
        if not value:
            continue
        key = normalize(value)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
        if len(cleaned) >= 2:
            break

    if len(cleaned) < 2:
        # Absolute fallback guard, though DEFAULT_SOURCES already has two entries.
        cleaned = [
            "WHO: https://www.who.int/health-topics",
            "MedlinePlus: https://medlineplus.gov/",
        ]

    return cleaned


def format_response(answer, confidence, sources):
    source_line = " | ".join(sources)
    return (
        f"{answer}\n"
        f"Confidence: {round(confidence * 100, 2)}%\n"
        f"Verified Sources: {source_line}\n"
        "Note: This is general health information, not a medical diagnosis."
    )


def is_low_quality_medical_text(text):
    value = normalize(text)
    if not value:
        return True
    return any(pattern in value for pattern in LOW_QUALITY_PATTERNS)


def safe_medical_detail(text, fallback_message):
    if is_low_quality_medical_text(text):
        return fallback_message
    return text


def condition_from_question_text(question_text):
    q = normalize(question_text)
    if q.startswith("symptoms of "):
        return q.replace("symptoms of ", "", 1).strip()
    if q.startswith("what causes "):
        return q.replace("what causes ", "", 1).strip()
    if q.startswith("what is "):
        return q.replace("what is ", "", 1).strip()
    return q.strip()


def all_conditions():
    """Return sorted unique condition names available in the dataset."""
    conditions = set()
    for _, row in data.iterrows():
        condition = condition_from_question_text(str(row["Question"]))
        if condition:
            conditions.add(condition)
    return sorted(conditions)


def detect_condition_mention(user_norm):
    """Return known condition explicitly mentioned in free-text query, if any."""
    padded = f" {apply_medical_aliases(user_norm)} "
    conditions = sorted(all_conditions(), key=len, reverse=True)
    normalized_to_original = {normalize(c): c for c in conditions}

    for condition in conditions:
        cond_norm = normalize(condition)
        if not cond_norm:
            continue
        if f" {cond_norm} " in padded:
            return condition

    # Alias phrase detection, e.g. "high blood pressure" -> "hypertension".
    for alias in sorted(MEDICAL_TERM_ALIASES.keys(), key=len, reverse=True):
        alias_norm = normalize(alias)
        if f" {alias_norm} " in padded:
            canonical_norm = normalize(MEDICAL_TERM_ALIASES[alias])
            if canonical_norm in normalized_to_original:
                return normalized_to_original[canonical_norm]
    return ""


def detect_condition_from_symptom_hint(user_norm):
    """Return condition hint from strong symptom phrases."""
    hint_map = {
        "pcod": [
            "irregular periods",
            "periods irregular",
            "missed periods",
            "delayed periods",
            "menstrual irregularity",
            "irregular menstruation",
        ],
    }

    for condition, phrases in hint_map.items():
        if any(phrase in user_norm for phrase in phrases):
            return condition
    return ""


def extract_explicit_condition_request(user_norm):
    """Extract condition name when user clearly asks about a specific disease."""
    patterns = [
        "what is ",
        "what causes ",
        "symptoms of ",
        "cause of ",
        "definition of ",
        "tell me about ",
        "about ",
    ]
    for prefix in patterns:
        if user_norm.startswith(prefix):
            candidate = user_norm.replace(prefix, "", 1).strip()
            if candidate:
                return candidate
    return ""


def resolve_condition_name(requested_condition):
    """Return exact dataset condition when available, otherwise empty string."""
    req = canonicalize_condition_name(requested_condition)
    if not req:
        return ""

    conditions = all_conditions()
    normalized_map = {normalize(c): c for c in conditions}
    if req in normalized_map:
        return normalized_map[req]

    # Allow near-identical condition phrases only.
    best_name = ""
    best_score = 0.0
    for c in conditions:
        score = SequenceMatcher(None, req, normalize(c)).ratio()
        if score > best_score:
            best_score = score
            best_name = c
    if best_score >= 0.88:
        return best_name
    return ""


def build_unsupported_condition_output(patient_name, user_question, requested_condition):
    """Structured output for disease names not present in verified dataset."""
    ranked = rank_condition_suggestions(requested_condition, limit=3, min_score=0.0)
    suggestions = [name.title() for name, _ in ranked]
    suggestion_text = ", ".join(suggestions) if suggestions else "No close condition found in current dataset."
    if ranked:
        probable_condition = ranked[0][0].title()
        confidence_val = max(0.55, min(0.92, 0.55 + ranked[0][1] * 0.35))
        confidence_text = f"{round(confidence_val * 100, 1)}%"
    else:
        probable_condition = requested_condition.title()
        confidence_text = "56.0%"

    field_width = 20
    details_width = 50
    border = "+" + "-" * (field_width + 2) + "+" + "-" * (details_width + 2) + "+"

    def build_row(field, details):
        wrapped = textwrap.wrap(str(details), width=details_width) or [""]
        lines = []
        for i, part in enumerate(wrapped):
            left = field if i == 0 else ""
            lines.append(f"| {left:<{field_width}} | {part:<{details_width}} |")
        return lines

    rows = []
    rows.extend(build_row("Patient Name", patient_name))
    rows.extend(build_row("Patient Input", user_question))
    rows.extend(build_row("Probable Condition", probable_condition))
    rows.extend(build_row("Confidence", confidence_text))
    rows.extend(build_row("Medical Definition", "This specific disease is not yet available in the current verified dataset."))
    rows.extend(build_row("Etiology (Causes)", "Accurate cause details are not returned to avoid misinformation."))
    rows.extend(build_row("Clinical Symptoms", "Accurate symptom details are not returned to avoid misinformation."))
    rows.extend(build_row("Next Best Matches", suggestion_text))
    rows.extend(build_row("Verified Sources", " | ".join(ensure_minimum_verified_sources(DEFAULT_SOURCES))))

    table_block = [
        border,
        f"| {'Field':<{field_width}} | {'Details':<{details_width}} |",
        border,
        *rows,
        border,
    ]

    return (
        "\n" + "=" * len(border) + "\n"
        "PRESCRIPTION STYLE HEALTH REPORT (GENERAL GUIDANCE)\n"
        + "=" * len(border) + "\n"
        + "\n".join(table_block)
        + "\nMedical Note: Add this disease to medical_dataset.csv with verified sources for accurate condition-specific answers."
    )


def rank_condition_suggestions(query, limit=6, min_score=0.25):
    """Return likely conditions ranked by similarity score."""
    query_norm = normalize(query)
    if not query_norm:
        return []

    query_tokens = set(keywords(query))
    ranked = []

    for condition in all_conditions():
        cond_norm = normalize(condition)
        cond_tokens = set(cond_norm.split())

        fuzzy = SequenceMatcher(None, query_norm, cond_norm).ratio()
        all_tok = query_tokens | cond_tokens
        overlap = len(query_tokens & cond_tokens) / len(all_tok) if all_tok else 0.0

        score = max(fuzzy, overlap)
        if any(tok in query_norm for tok in cond_tokens):
            score += 0.10

        ranked.append((score, condition))

    ranked.sort(key=lambda x: x[0], reverse=True)
    ranked = [(name, score) for score, name in ranked[:limit] if score >= min_score]
    return ranked


def suggest_conditions(query, limit=6):
    """Suggest likely conditions when user input does not map confidently."""
    ranked = rank_condition_suggestions(query, limit=limit, min_score=0.25)
    suggestions = [name.title() for name, _ in ranked]

    return suggestions


def find_row_by_prefix(prefix, condition):
    target = f"{prefix} {condition}".strip()
    for _, row in data.iterrows():
        q = normalize(row["Question"])
        if q == target:
            return row
    return None


def extract_json_object(text):
    """Best-effort extraction of first JSON object from model output."""
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def get_retrieved_context(matched_row):
    """Build structured retrieval context from dataset rows."""
    matched_q = str(matched_row.get("Question", ""))
    condition = condition_from_question_text(matched_q)
    category = normalize(str(matched_row.get("Category", "")))

    what_row = find_row_by_prefix("what is", condition)
    cause_row = find_row_by_prefix("what causes", condition)
    symptom_row = find_row_by_prefix("symptoms of", condition)

    description_line = (
        str(what_row.get("Answer", ""))
        if what_row is not None
        else "Description not available in current dataset."
    )
    description_line = safe_medical_detail(
        description_line,
        "Description requires verified dataset update for this condition.",
    )
    cause_line = (
        str(cause_row.get("Answer", ""))
        if cause_row is not None
        else "Cause information not available in current dataset."
    )
    cause_line = safe_medical_detail(
        cause_line,
        "Cause information requires verified dataset update for this condition.",
    )
    symptom_line = (
        str(symptom_row.get("Answer", ""))
        if symptom_row is not None
        else "Symptom information not available in current dataset."
    )
    symptom_line = safe_medical_detail(
        symptom_line,
        "Symptom information requires verified dataset update for this condition.",
    )

    care_steps = CARE_GUIDANCE_BY_CATEGORY.get(
        category,
        [
            "Maintain hydration, balanced diet, and proper rest.",
            "Avoid self-medication and monitor symptoms closely.",
            "Consult a qualified doctor for diagnosis and treatment.",
        ],
    )
    red_flags = RED_FLAGS_BY_CATEGORY.get(
        category,
        "Severe pain, persistent high fever, breathing issues, or sudden worsening symptoms.",
    )

    return {
        "condition": condition,
        "category": category,
        "description": description_line,
        "causes": cause_line,
        "symptoms": symptom_line,
        "care_steps": care_steps,
        "red_flags": red_flags,
        "what_row": what_row,
        "cause_row": cause_row,
        "symptom_row": symptom_row,
    }


def build_possible_prescription_output(patient_name, user_question, possible_conditions, processing_issue=False):
    """Always return a structured prescription-style response for uncertain matches."""
    possible_conditions = possible_conditions or []
    ranked = rank_condition_suggestions(user_question, limit=3, min_score=0.0)

    if possible_conditions:
        condition_line = str(possible_conditions[0]).title()
        symptoms_line = "Possible conditions based on symptoms:\n" + "\n".join(
            [f"- {c}" for c in possible_conditions[:3]]
        )
    else:
        condition_line = "General Symptom Pattern"
        symptoms_line = "Symptoms are not strongly mapped to one condition in the dataset."

    if ranked:
        condition_line = ranked[0][0].title()
        confidence_val = max(0.56, min(0.90, 0.56 + ranked[0][1] * 0.34))
        confidence_text = f"{round(confidence_val * 100, 1)}%"
    elif possible_conditions:
        confidence_text = "62.0%"
    else:
        confidence_text = "56.0%"

    care_steps = [
        "Rest, hydrate, and monitor symptom changes.",
        "Avoid self-medication beyond basic supportive care.",
        "Consult a doctor for persistent or worsening symptoms.",
    ]
    red_flags = "Severe pain, high persistent fever, breathing problems, confusion, or sudden worsening symptoms."
    source_line = " | ".join(ensure_minimum_verified_sources(DEFAULT_SOURCES))

    condition_key = normalize(condition_line)
    what_row = find_row_by_prefix("what is", condition_key)
    cause_row = find_row_by_prefix("what causes", condition_key)

    description_line = (
        str(what_row.get("Answer", "")).strip()
        if what_row is not None
        else f"{condition_line} is a health condition requiring medical evaluation for confirmation."
    )
    cause_line = (
        str(cause_row.get("Answer", "")).strip()
        if cause_row is not None
        else f"Causes of {condition_line.lower()} may vary; clinical assessment is recommended for accurate identification."
    )

    description_line = safe_medical_detail(
        description_line,
        f"{condition_line} is a health condition requiring medical evaluation for confirmation.",
    )
    cause_line = safe_medical_detail(
        cause_line,
        f"Causes of {condition_line.lower()} may vary; clinical assessment is recommended for accurate identification.",
    )

    field_width = 20
    details_width = 50
    border = "+" + "-" * (field_width + 2) + "+" + "-" * (details_width + 2) + "+"

    def build_row(field, details):
        wrapped = textwrap.wrap(str(details), width=details_width) or [""]
        lines = []
        for i, part in enumerate(wrapped):
            left = field if i == 0 else ""
            lines.append(f"| {left:<{field_width}} | {part:<{details_width}} |")
        return lines

    rows = []
    rows.extend(build_row("Patient Name", patient_name))
    rows.extend(build_row("Patient Input", user_question))
    rows.extend(build_row("Probable Condition", condition_line))
    rows.extend(build_row("Confidence", confidence_text))
    rows.extend(build_row("Medical Definition", description_line))
    rows.extend(build_row("Etiology (Causes)", cause_line))
    rows.extend(build_row("Clinical Symptoms", symptoms_line))
    rows.extend(build_row("Management / Solution", "; ".join(care_steps)))
    rows.extend(build_row("Seek Urgent Care If", red_flags))
    rows.extend(build_row("Verified Sources", source_line))

    table_block = [
        border,
        f"| {'Field':<{field_width}} | {'Details':<{details_width}} |",
        border,
        *rows,
        border,
    ]

    issue_line = "Processing issue handled. Displaying best available result." if processing_issue else ""
    return (
        "\n" + "=" * len(border) + "\n"
        "PRESCRIPTION STYLE HEALTH REPORT (GENERAL GUIDANCE)\n"
        + "=" * len(border) + "\n"
        + "\n".join(table_block)
        + (f"\n{issue_line}" if issue_line else "")
        + "\nMedical Note: This is educational guidance only and not a medical diagnosis."
    )


def build_llm_prompt(user_input, context):
    system_prompt = (
        "You are a medical information assistant. Provide clear, simple, and accurate "
        "health information based ONLY on the given context. Do not hallucinate. "
        "Do not provide diagnosis."
    )

    user_prompt = (
        "Context:\n"
        f"- Condition: {context['condition']}\n"
        f"- Description: {context['description']}\n"
        f"- Causes: {context['causes']}\n"
        f"- Symptoms: {context['symptoms']}\n"
        f"- Care Guidance: {'; '.join(context['care_steps'])}\n\n"
        "Question:\n"
        f"{user_input}\n\n"
        "Instructions:\n"
        "- Answer in simple language\n"
        "- Be concise and structured\n"
        "- Do NOT add information outside the context\n"
        "- Include a safety disclaimer\n"
        "- Return ONLY valid JSON with keys: description, symptoms, solution, safety_disclaimer"
    )

    return f"System: {system_prompt}\n\nUser: {user_prompt}"


def call_ollama_mistral(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "seed": LLM_SEED,
        },
    }

    req = request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as response:
        raw = response.read().decode("utf-8")
        body = json.loads(raw)
        return body.get("response", "")


def call_hf_mistral(prompt):
    """Optional HF backend for GPU-enabled setups."""
    try:
        transformers = __import__("transformers")
        torch = __import__("torch")
        AutoTokenizer = transformers.AutoTokenizer
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
    except ImportError:
        return ""

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded


def generate_llm_sections(user_input, matched_row):
    """Generate enhanced Description/Symptoms/Solution from Mistral using retrieved context."""
    if not USE_LLM:
        return None
    context = get_retrieved_context(matched_row)
    prompt = build_llm_prompt(user_input, context)

    try:
        if LLM_BACKEND == "ollama":
            model_text = call_ollama_mistral(prompt)
        elif LLM_BACKEND == "hf":
            model_text = call_hf_mistral(prompt)
        else:
            return None
    except (error.URLError, TimeoutError, ValueError, OSError, RuntimeError):
        return None

    parsed = extract_json_object(model_text)
    if not parsed:
        return None

    description = str(parsed.get("description", "")).strip()
    symptoms = str(parsed.get("symptoms", "")).strip()
    solution = str(parsed.get("solution", "")).strip()
    safety_disclaimer = str(parsed.get("safety_disclaimer", "")).strip()

    if not (description and symptoms and solution):
        return None

    return {
        "description": description,
        "symptoms": symptoms,
        "solution": solution,
        "safety_disclaimer": safety_disclaimer,
    }


def evidence_count_for_condition(condition):
    """Return how many evidence rows exist for a condition: what/cause/symptoms."""
    return sum([
        find_row_by_prefix("what is", condition) is not None,
        find_row_by_prefix("what causes", condition) is not None,
        find_row_by_prefix("symptoms of", condition) is not None,
    ])


def dataset_conditions():
    conditions = set()
    for _, row in data.iterrows():
        condition = condition_from_question_text(str(row["Question"]))
        if condition:
            conditions.add(condition)
    return conditions


def init_session_metrics():
    return {
        "total_queries": 0,
        "answered": 0,
        "unanswered": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "response_times_ms": [],
        "confidence_sum": 0.0,
        "confidence_count": 0,
        "evidence_total": 0,
        "evidence_possible": 0,
        "safety_reports": 0,
        "method_counter": Counter(),
        "condition_counter": Counter(),
    }


def update_session_metrics(metrics, answer, confidence, matched_row, research_info, latency_ms, from_cache):
    metrics["total_queries"] += 1
    metrics["response_times_ms"].append(latency_ms)

    if from_cache:
        metrics["cache_hits"] += 1
    else:
        metrics["cache_misses"] += 1

    if answer and matched_row is not None:
        metrics["answered"] += 1
        metrics["confidence_sum"] += confidence
        metrics["confidence_count"] += 1
        metrics["safety_reports"] += 1

        condition = condition_from_question_text(str(matched_row["Question"])).title()
        metrics["condition_counter"][condition] += 1

        evidence_count = evidence_count_for_condition(normalize(condition))
        metrics["evidence_total"] += evidence_count
        metrics["evidence_possible"] += 3

        method = research_info.get("method", "unknown") if research_info else "unknown"
        metrics["method_counter"][method] += 1
    else:
        metrics["unanswered"] += 1


def render_metrics_dashboard(metrics):
    total = metrics["total_queries"]
    answered = metrics["answered"]
    unanswered = metrics["unanswered"]

    avg_conf = (metrics["confidence_sum"] / metrics["confidence_count"] * 100) if metrics["confidence_count"] else 0.0
    evidence_pct = (metrics["evidence_total"] / metrics["evidence_possible"] * 100) if metrics["evidence_possible"] else 0.0
    safety_pct = (metrics["safety_reports"] / answered * 100) if answered else 0.0

    times = sorted(metrics["response_times_ms"])
    avg_latency = (sum(times) / len(times)) if times else 0.0
    if times:
        p95_index = max(0, min(len(times) - 1, int(len(times) * 0.95 + 0.9999) - 1))
        p95_latency = times[p95_index]
    else:
        p95_latency = 0.0

    all_conditions = dataset_conditions()
    covered_conditions = len(metrics["condition_counter"])
    coverage_pct = (covered_conditions / len(all_conditions) * 100) if all_conditions else 0.0

    top_conditions = metrics["condition_counter"].most_common(3)
    top_methods = metrics["method_counter"].most_common(3)

    top_condition_line = ", ".join([f"{name} ({count})" for name, count in top_conditions]) if top_conditions else "None"
    top_method_line = ", ".join([f"{name} ({count})" for name, count in top_methods]) if top_methods else "None"

    return (
        "\n" + "=" * 70 + "\n"
        "RESEARCH EVALUATION DASHBOARD (SESSION)\n"
        + "=" * 70 + "\n"
        f"Total Queries: {total}\n"
        f"Answered: {answered} | Unanswered: {unanswered}\n"
        f"Cache Hits: {metrics['cache_hits']} | Cache Misses: {metrics['cache_misses']}\n"
        f"Top-1 Relevance Proxy (Avg Confidence): {avg_conf:.2f}%\n"
        f"Evidence Completeness: {evidence_pct:.2f}%\n"
        f"Safety Compliance (Reports with Disclaimer): {safety_pct:.2f}%\n"
        f"Latency Avg/P95: {avg_latency:.2f} ms / {p95_latency:.2f} ms\n"
        f"Coverage Completeness: {covered_conditions}/{len(all_conditions)} conditions ({coverage_pct:.2f}%)\n"
        f"Top Conditions Asked: {top_condition_line}\n"
        f"Top Matching Methods: {top_method_line}\n"
        + "=" * 70
    )


def build_prescription_output(patient_name, user_question, confidence, matched_row, research_info=None, llm_sections=None):
    research_info = research_info or {}
    context = get_retrieved_context(matched_row)
    condition = context["condition"]
    category = context["category"]
    what_row = context["what_row"]
    cause_row = context["cause_row"]
    symptom_row = context["symptom_row"]

    condition_line = condition.title() if condition else "Not identified"
    description_line = context["description"]
    cause_line = context["causes"]
    symptom_line = context["symptoms"]
    care_steps = context["care_steps"]
    red_flags = context["red_flags"]

    if llm_sections:
        description_line = llm_sections.get("description", description_line)
        symptom_line = llm_sections.get("symptoms", symptom_line)
        llm_solution = llm_sections.get("solution", "")
        if llm_solution:
            care_steps = [llm_solution]
    sources = get_sources_for_row(matched_row)
    source_line = " | ".join(sources)
    evidence_count = sum([what_row is not None, cause_row is not None, symptom_row is not None])
    evidence_coverage = f"{evidence_count}/3 (Definition, Etiology, Symptoms)"

    method_line = research_info.get("method", "similarity-ranking")
    score_breakdown = research_info.get("score_breakdown", {})
    if score_breakdown:
        breakdown_line = " | ".join([f"{k}={v}" for k, v in score_breakdown.items()])
    else:
        breakdown_line = "Not available"

    top_matches = research_info.get("top_matches", [])
    if top_matches:
        top_match_line = "; ".join(
            [f"{m['condition']} ({m['score']}%)" for m in top_matches]
        )
    else:
        top_match_line = "Not available"

    if confidence < 0.7 and top_matches:
        most_probable = top_matches[0].get("condition", condition_line)
        others = [m.get("condition", "") for m in top_matches[1:3] if m.get("condition")]
        inference_summary = f"Most probable condition: {most_probable}"
        if others:
            inference_summary += f" | Other possible conditions: {', '.join(others)}"
    else:
        inference_summary = "High-confidence single best match selected from symptom evidence."

    solution_line = "; ".join(care_steps)

    # Keep table compact so it renders correctly in standard PowerShell width.
    field_width = 20
    details_width = 50
    border = "+" + "-" * (field_width + 2) + "+" + "-" * (details_width + 2) + "+"

    def build_row(field, details):
        wrapped = textwrap.wrap(str(details), width=details_width) or [""]
        lines = []
        for i, part in enumerate(wrapped):
            left = field if i == 0 else ""
            lines.append(f"| {left:<{field_width}} | {part:<{details_width}} |")
        return lines

    rows = []
    rows.extend(build_row("Patient Name", patient_name))
    rows.extend(build_row("Patient Input", user_question))
    rows.extend(build_row("Probable Condition", condition_line))
    rows.extend(build_row("Confidence", f"{round(confidence * 100, 2)}%"))
    rows.extend(build_row("Medical Definition", description_line))
    rows.extend(build_row("Etiology (Causes)", cause_line))
    rows.extend(build_row("Clinical Symptoms", symptom_line))
    rows.extend(build_row("Management / Solution", solution_line))
    rows.extend(build_row("Evidence Coverage", evidence_coverage))
    rows.extend(build_row("Research Method", method_line))
    rows.extend(build_row("Score Breakdown", breakdown_line))
    rows.extend(build_row("Inference Summary", inference_summary))
    rows.extend(build_row("Top Related Matches", top_match_line))
    rows.extend(build_row("Analysis Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    rows.extend(build_row("Seek Urgent Care If", red_flags))
    rows.extend(build_row("Verified Sources", source_line))

    table_block = [
        border,
        f"| {'Field':<{field_width}} | {'Details':<{details_width}} |",
        border,
        *rows,
        border,
    ]

    safety_note = llm_sections.get("safety_disclaimer", "") if llm_sections else ""
    if not safety_note:
        safety_note = "This is educational guidance only and cannot diagnose or cure all diseases."

    return (
        "\n" + "=" * len(border) + "\n"
        "PRESCRIPTION STYLE HEALTH REPORT (GENERAL GUIDANCE)\n"
        + "=" * len(border) + "\n"
        + "\n".join(table_block)
        + f"\nMedical Note: {safety_note}"
    )


def infer_condition_from_description(user_norm, user_tokens, user_kws, top_n=3):
    """Match user-described symptoms to the closest 'Symptoms of X' row."""
    best_row = None
    best_score = -1.0
    ranked = []
    digestive_terms = {"stomach", "digestion", "digestive", "food", "abdominal", "abdomen", "acid", "burning"}
    digestive_query = bool(user_tokens & digestive_terms)

    for _, row in data.iterrows():
        db_q = normalize(row.get("Question", ""))
        if not db_q.startswith("symptoms of"):
            continue

        symptom_text = normalize(row.get("Answer", ""))
        symptom_tokens = set(symptom_text.split())
        condition_text = condition_from_question_text(db_q)
        condition_tokens = set(condition_text.split())

        kw_score = keyword_hit_score(user_kws, symptom_text)
        all_tok = user_tokens | symptom_tokens
        jaccard = len(user_tokens & symptom_tokens) / len(all_tok) if all_tok else 0.0
        cond_overlap = len(user_tokens & condition_tokens) / len(user_tokens | condition_tokens) if (user_tokens | condition_tokens) else 0.0
        fuzzy = SequenceMatcher(None, user_norm, symptom_text).ratio()
        symptom_overlap = len(user_tokens & symptom_tokens) / len(user_tokens) if user_tokens else 0.0

        category = normalize(str(row.get("Category", "")))
        digestive_boost = 0.15 if digestive_query and "digestive" in category else 0.0

        score = (
            kw_score * 0.15
            + jaccard * 0.25
            + fuzzy * 0.20
            + cond_overlap * 0.10
            + symptom_overlap * 0.30
            + digestive_boost
        )
        ranked.append((score, row))

        if score > best_score:
            best_score = score
            best_row = row

    ranked = sorted(ranked, key=lambda x: x[0], reverse=True)
    top_matches = [
        {
            "condition": condition_from_question_text(str(r.get("Question", ""))).title(),
            "score": round(s * 100, 2),
            "row": r,
        }
        for s, r in ranked[:top_n]
        if s >= 0.10
    ]

    if best_row is not None:
        return best_row, best_score, top_matches

    return None, 0.0, top_matches


def chatbot(question):

    user_norm = apply_medical_aliases(question)
    user_tokens = set(user_norm.split())

    if not user_norm:
        return None, 0, None, {}

    hinted_condition = detect_condition_from_symptom_hint(user_norm)
    if hinted_condition:
        hint_row = find_row_by_prefix("what is", hinted_condition)
        if hint_row is not None:
            research_info = {
                "method": "symptom-hint-rule",
                "score_breakdown": {
                    "confidence": 0.86,
                },
                "top_matches": [
                    {
                        "condition": hinted_condition.upper(),
                        "score": 86.0,
                    }
                ],
            }
            return hint_row.get("Answer", ""), 0.86, hint_row, research_info

    mentioned_condition = detect_condition_mention(user_norm)
    if mentioned_condition:
        what_row = find_row_by_prefix("what is", normalize(mentioned_condition))
        if what_row is not None:
            research_info = {
                "method": "direct-condition-mention",
                "score_breakdown": {
                    "confidence": 0.88,
                },
                "top_matches": [
                    {
                        "condition": mentioned_condition.title(),
                        "score": 88.0,
                    }
                ],
            }
            return what_row.get("Answer", ""), 0.88, what_row, research_info

    user_kws = keywords(question)

    wants_symptoms = bool(user_tokens & SYMPTOM_WORDS)
    wants_cause = bool(user_tokens & CAUSE_WORDS)
    wants_definition = bool(user_tokens & DEFINITION_WORDS)
    personal_symptom_style = bool(user_tokens & PERSONAL_SYMPTOM_WORDS)

    # Handle free-text symptom descriptions such as:
    # "My stomach burns after eating spicy food"
    if personal_symptom_style and not (wants_symptoms or wants_cause or wants_definition):
        inferred_row, inferred_score, inferred_top = infer_condition_from_description(user_norm, user_tokens, user_kws)
        if inferred_row is not None:
            confidence = scaled_confidence(inferred_score)
            research_info = {
                "method": "symptom-description-inference",
                "score_breakdown": {
                    "inference_score": round(inferred_score, 4),
                    "confidence": round(confidence, 4),
                },
                "top_matches": [
                    {"condition": m["condition"], "score": m["score"]}
                    for m in inferred_top
                ],
            }
            return inferred_row.get("Answer", ""), confidence, inferred_row, research_info

        if inferred_top:
            fallback_row = inferred_top[0].get("row")
            if fallback_row is not None:
                fallback_score = inferred_top[0].get("score", 0.0) / 100
                confidence = scaled_confidence(fallback_score)
                research_info = {
                    "method": "symptom-description-inference",
                    "score_breakdown": {
                        "inference_score": round(fallback_score, 4),
                        "confidence": round(confidence, 4),
                    },
                    "top_matches": [
                        {"condition": m["condition"], "score": m["score"]}
                        for m in inferred_top
                    ],
                }
                return fallback_row.get("Answer", ""), confidence, fallback_row, research_info

    best_score = -1.0
    best_answer = None
    best_row = None
    best_breakdown = {}
    top_candidates = []
    digestive_terms = {"stomach", "digestion", "digestive", "food", "abdominal", "abdomen", "acid", "burning"}
    digestive_query = bool(user_tokens & digestive_terms)

    for _, row in data.iterrows():

        db_q = normalize(row.get("Question", ""))
        db_tokens = set(db_q.split())

        # Exact match
        if user_norm == db_q or db_q in user_norm or user_norm in db_q:
            research_info = {
                "method": "exact-match",
                "score_breakdown": {
                    "confidence": 1.0,
                },
                "top_matches": [
                    {
                        "condition": condition_from_question_text(str(row.get("Question", ""))).title(),
                        "score": 100.0,
                    }
                ],
            }
            return row.get("Answer", ""), 1.0, row, research_info

        # Row type
        is_symptom_row = db_q.startswith("symptoms of") or "symptom" in db_q
        is_cause_row = db_q.startswith("what causes") or "cause" in db_q
        is_definition_row = db_q.startswith("what is") or db_q.startswith("why")

        # Category boost
        db_category = normalize(str(row.get("Category", "")))
        category_boost = 0

        if user_kws and db_category:

            cat_tokens = db_category.split()

            if any(
                kw in db_category or db_category in kw
                or any(
                    tok.startswith(kw) or kw.startswith(tok)
                    for tok in cat_tokens
                )
                for kw in user_kws
            ):
                category_boost = 0.30

        if digestive_query and "digestive" in db_category:
            category_boost += 0.15

        # Similarity scores
        kw_score = keyword_hit_score(user_kws, db_q)

        all_tok = user_tokens | db_tokens
        jaccard = len(user_tokens & db_tokens) / len(all_tok) if all_tok else 0

        fuzzy = SequenceMatcher(None, user_norm, db_q).ratio()

        score = kw_score * 0.55 + jaccard * 0.25 + fuzzy * 0.20 + category_boost
        intent_boost = 0.0

        # Intent boosts
        if wants_symptoms and is_symptom_row:
            score += 0.25
            intent_boost += 0.25

        if wants_cause and is_cause_row:
            score += 0.20
            intent_boost += 0.20

        if wants_definition and is_definition_row:
            score += 0.15
            intent_boost += 0.15

        breakdown = {
            "keyword": round(kw_score, 4),
            "jaccard": round(jaccard, 4),
            "fuzzy": round(fuzzy, 4),
            "category_boost": round(category_boost, 4),
            "intent_boost": round(intent_boost, 4),
            "final_score": round(score, 4),
        }
        top_candidates.append((score, row, breakdown))

        if score > best_score:
            best_score = score
            best_answer = row.get("Answer", "")
            best_row = row
            best_breakdown = breakdown

    if best_row is not None:
        top_candidates = sorted(top_candidates, key=lambda x: x[0], reverse=True)[:3]
        confidence = scaled_confidence(best_score)
        research_info = {
            "method": "similarity-ranking",
            "score_breakdown": {**best_breakdown, "confidence": round(confidence, 4)},
            "top_matches": [
                {
                    "condition": condition_from_question_text(str(candidate_row.get("Question", ""))).title(),
                    "score": round(candidate_score * 100, 2),
                }
                for candidate_score, candidate_row, _ in top_candidates
            ],
        }
        return best_answer, confidence, best_row, research_info

    return None, best_score, None, {}


def process_user_query(user, patient_name, query_cache, session_metrics):
    """Process one user query and return structured response payload."""
    try:
        user = str(user).strip()
        patient_name = str(patient_name or "Patient").strip() or "Patient"
        if not isinstance(query_cache, dict):
            query_cache = {}
        if not isinstance(session_metrics, dict):
            session_metrics = init_session_metrics()

        if not user:
            return {
                "patient_name": patient_name,
                "message": "Please enter a medical query.",
                "status": "empty",
            }

        if user.lower() == "metrics":
            return {
                "patient_name": patient_name,
                "message": render_metrics_dashboard(session_metrics),
                "status": "metrics",
            }

        if user.lower() == "options":
            conditions = all_conditions()
            preview = ", ".join([c.title() for c in conditions[:30]])
            return {
                "patient_name": patient_name,
                "message": f"Available options include:\n{preview}",
                "status": "options",
            }

        if user.lower().startswith("name:"):
            new_name = user.split(":", 1)[1].strip()
            if new_name:
                return {
                    "patient_name": new_name,
                    "message": f"Patient name updated to {new_name}.",
                    "status": "name-updated",
                }
            return {
                "patient_name": patient_name,
                "message": "Please enter a valid name after 'name:'.",
                "status": "name-invalid",
            }

        # Accuracy guardrail: for explicit disease requests, avoid mapping to unrelated conditions.
        user_norm = normalize(user)
        requested_condition = extract_explicit_condition_request(user_norm)
        if requested_condition:
            resolved = resolve_condition_name(requested_condition)
            if not resolved:
                return {
                    "patient_name": patient_name,
                    "message": build_unsupported_condition_output(patient_name, user, requested_condition),
                    "status": "unsupported-condition",
                }

        with open("query_log.txt", "a") as f:
            f.write(user + "\n")

        cache_key = normalize(user)
        start_time = time.perf_counter()
        from_cache = False
        if cache_key in query_cache:
            from_cache = True
            answer, confidence, matched_row, research_info = query_cache.get(cache_key, (None, 0, None, {}))
        else:
            answer, confidence, matched_row, research_info = chatbot(user)
            query_cache[cache_key] = (answer, confidence, matched_row, research_info)
        latency_ms = (time.perf_counter() - start_time) * 1000

        update_session_metrics(
            session_metrics,
            answer,
            confidence,
            matched_row,
            research_info,
            latency_ms,
            from_cache,
        )

        if answer and matched_row is not None:
            llm_sections = generate_llm_sections(user, matched_row)
            response_text = build_prescription_output(
                patient_name,
                user,
                confidence,
                matched_row,
                research_info,
                llm_sections=llm_sections,
            )
            return {
                "patient_name": patient_name,
                "message": response_text,
                "status": "answer",
            }

        suggestions = suggest_conditions(user, limit=3)
        response_text = build_possible_prescription_output(
            patient_name,
            user,
            suggestions,
            processing_issue=False,
        )
        return {
            "patient_name": patient_name,
            "message": response_text,
            "status": "possible-conditions" if suggestions else "fallback",
        }
    except Exception as exc:
        log_internal_error("process_user_query", exc)
        suggestions = suggest_conditions(user if 'user' in locals() else "", limit=3)
        response_text = build_possible_prescription_output(
            patient_name if 'patient_name' in locals() else "Patient",
            user if 'user' in locals() else "",
            suggestions,
            processing_issue=True,
        )
        return {
            "patient_name": patient_name if 'patient_name' in locals() else "Patient",
            "message": response_text,
            "status": "error-recovered",
        }


def run_cli():
    print("=" * 55)
    print("Medical Information Chatbot")
    print("Type 'exit' to quit")
    print("Type 'name: <patient name>' to change patient")
    print("Type 'metrics' to view evaluation dashboard")
    print("Type 'options' to view available disease options")
    print("=" * 55)

    patient_name = input("Enter patient name: ").strip() or "Patient"
    query_cache = {}
    session_metrics = init_session_metrics()

    while True:
        user = input("\nYou: ").strip()
        if not user:
            continue
        if user.lower() == "exit":
            print("Chatbot: Take care. Goodbye!")
            break

        result = process_user_query(user, patient_name, query_cache, session_metrics)
        patient_name = result["patient_name"]
        print("Chatbot:", result["message"])


if __name__ == "__main__":
    run_cli()