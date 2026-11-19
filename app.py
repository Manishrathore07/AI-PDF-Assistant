import os

import io

import re

import json

import hashlib

import faiss

import numpy as np

import streamlit as st

from rank_bm25 import BM25Okapi

from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

from google import genai

# =========================================================

# 1. PAGE CONFIGURATION

# =========================================================

st.set_page_config(

    page_title="PDF AI Assistant",

    page_icon="📚",

    layout="wide"

)

st.markdown("""

<style>

.stApp { background-color: #F8FAFC; }

header[data-testid="stHeader"] { background-color: #F8FAFC; }

.block-container {

    padding-top: 2rem;

    padding-bottom: 2rem;

    max-width: 1100px;

}

section[data-testid="stSidebar"] {

    background-color: #FFFFFF;

    border-right: 1px solid #E2E8F0;

}

h1, h2, h3 { color: #172554; }

p { color: #475569; }

div[data-testid="stMetric"] {

    background-color: #FFFFFF;

    border: 1px solid #E2E8F0;

    padding: 16px;

    border-radius: 12px;

}

.stButton button {

    border-radius: 10px;

    border: 1px solid #DBEAFE;

    background-color: #FFFFFF;

    color: #1D4ED8;

}

div[data-testid="stChatMessage"] {

    border: 1px solid #E2E8F0;

    border-radius: 12px;

    padding: 12px;

    background-color: #FFFFFF;

}

</style>

""", unsafe_allow_html=True)

# =========================================================

# 2. MODEL AND GEMINI CLIENT

# =========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    st.error("Gemini API key not found. Check your .env file.")

    st.stop()

client = genai.Client(api_key=api_key)

GEMINI_MODEL = "gemini-3.5-flash-lite"

@st.cache_resource

def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# =========================================================

# 3. TEXT CHUNKING

# =========================================================

def is_heading(line):

    line = line.strip()

    if not line or len(line) > 100:

        return False

    letters = [c for c in line if c.isalpha()]

    if not letters:

        return False

    if sum(c.isupper() for c in letters) / len(letters) >= 0.75:

        return True

    normalized = re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()

    generic = {

        "project", "projects", "project exhibition", "project experience",

        "personal projects", "academic projects", "selected projects",

        "experience", "education", "technical skills", "skills",

        "achievements", "certifications"

    }

    if normalized in generic or re.search(r"[.!?]$", line):

        return False

    first_word = line.split()[0].lower().strip("•-*–—|:")

    if first_word in {

        "developed", "built", "created", "implemented", "designed",

        "engineered", "trained", "used", "using", "worked", "showcased"

    }:

        return False

    words = line.split()

    if len(words) < 2 or len(words) > 12:

        return False

    has_marker = bool(re.search(

        r"\b(project|system|interface|application|platform|tool|assistant|"

        r"chatbot|recognition|detector|classifier|dashboard|website|automation|agent)\b",

        line, re.I

    ))

    has_separator = bool(re.search(r"[—–:|]", line))

    if not (has_marker or has_separator):

        return False

    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]

    title_case = sum(

        1 for w in alpha_words if w[0].isupper() or w.isupper()

    )

    return bool(

        has_separator or

        (alpha_words and title_case / len(alpha_words) >= 0.60)

    )

def create_chunks(pages, chunk_size=500):

    chunks = []

    for page in pages:

        page_number = page["page_number"]

        current_lines = []

        current_length = 0

        def save_chunk():

            nonlocal current_lines, current_length

            if current_lines:

                text = "\n".join(current_lines).strip()

                if text:

                    chunks.append({

                        "chunk_number": len(chunks) + 1,

                        "text": text,

                        "pages": [page_number]

                    })

            current_lines = []

            current_length = 0

        for raw_line in page["text"].splitlines():

            line = raw_line.strip()

            if not line:

                save_chunk()

                continue

            if is_heading(line):

                save_chunk()

                current_lines.append(line)

                current_length = len(line)

                save_chunk()

                continue

            if len(line) > chunk_size:

                save_chunk()

                words = line.split()

                for word in words:

                    if not current_lines:

                        current_lines.append(word)

                        current_length = len(word)

                    elif current_length + len(word) + 1 <= chunk_size:

                        current_lines.append(word)

                        current_length += len(word) + 1

                    else:

                        save_chunk()

                        current_lines.append(word)

                        current_length = len(word)

                continue

            if current_length + len(line) + 1 > chunk_size:

                save_chunk()

            current_lines.append(line)

            current_length += len(line) if current_length == 0 else len(line) + 1

        save_chunk()

    return chunks

# =========================================================

# 4. BM25 AND HYBRID RETRIEVAL

# =========================================================

def tokenize_text(text):

    return re.findall(r"[a-zA-Z0-9+#.]+", text.lower())

def build_bm25_index(chunks):

    tokenized = [tokenize_text(c["text"]) for c in chunks]

    return BM25Okapi(tokenized) if tokenized else None

def is_definition_question(question):

    q = question.lower().strip()

    patterns = [

        r"^what\s+is\s+",

        r"^what\s+are\s+",

        r"^define\s+",

        r"^definition\s+of\s+",

        r"^meaning\s+of\s+",

        r"^what\s+does\s+.+?\s+mean\??$"

    ]

    return any(re.search(pattern, q) for pattern in patterns)

def extract_definition_subject(question):

    q = question.lower().strip().rstrip("?.! ")

    patterns = [

        r"^what\s+is\s+(.+)$",

        r"^what\s+are\s+(.+)$",

        r"^define\s+(.+)$",

        r"^definition\s+of\s+(.+)$",

        r"^meaning\s+of\s+(.+)$",

        r"^what\s+does\s+(.+?)\s+mean$"

    ]

    for pattern in patterns:

        match = re.match(pattern, q)

        if match:

            subject = match.group(1).strip()

            subject = re.sub(r"^(a|an|the)\s+", "", subject)

            return subject

    return ""

def definition_search_variants(question):

    subject = extract_definition_subject(question)

    if not subject:

        return []

    return [

        f"{subject} definition meaning explanation",

        f"{subject} refers to means is a technique or method",

        f"how {subject} works and its purpose",

        f"explain {subject} concept in the document"

    ]

def is_heading_only_chunk(text):

    cleaned = re.sub(r"[*_`#]+", "", text).strip()

    words = re.findall(r"\b\w+\b", cleaned)

    if len(words) > 14:

        return False

    explanatory_cues = [

        " is ", " are ", " refers to ", " means ", " defined as ",

        " consists of ", " can be defined ", " is a technique ",

        " is a method ", " is the process ", " allows ", " enables "

    ]

    padded = f" {cleaned.lower()} "

    return not any(cue in padded for cue in explanatory_cues)

def hybrid_search(query_variants, faiss_index, bm25_index, chunks, top_k=12):

    if not chunks or bm25_index is None or faiss_index is None:

        return []

    total_chunks = len(chunks)

    rrf_scores = {}

    semantic_scores_lookup = {}

    bm25_scores_lookup = {}

    rrf_constant = 60

    candidate_k = min(total_chunks, max(30, (top_k or 12) * 4))

    # Add deterministic definition-focused variants. These are especially

    # useful when Gemini's query expansion misses the wording in the PDF.

    queries = list(query_variants or [])

    is_definition = any(is_definition_question(q) for q in queries)

    if is_definition:

        for variant in definition_search_variants(queries[0]):

            if variant.lower() not in {q.lower() for q in queries}:

                queries.append(variant)

    for query in queries:

        if not query.strip():

            continue

        embedding = np.asarray(model.encode([query]), dtype="float32")

        faiss.normalize_L2(embedding)

        semantic_k = min(total_chunks, candidate_k)

        semantic_scores, semantic_indices = faiss_index.search(

            embedding, semantic_k

        )

        for rank, raw_index in enumerate(semantic_indices[0]):

            if raw_index < 0:

                continue

            index = int(raw_index)

            score = float(semantic_scores[0][rank])

            semantic_scores_lookup[index] = max(

                semantic_scores_lookup.get(index, float("-inf")), score

            )

            rrf_scores[index] = rrf_scores.get(index, 0.0) + (

                1.0 / (rrf_constant + rank + 1)

            )

        keyword_query = query

        if "project" in query.lower():

            keyword_query += " project name description technologies tools frameworks"

        bm25_scores = bm25_index.get_scores(tokenize_text(keyword_query))

        # Ignore zero-score chunks. Previously they were still ranked and

        # added to RRF, allowing unrelated chunks to dilute useful matches.

        positive_indices = [

            i for i, score in enumerate(bm25_scores) if float(score) > 0

        ]

        positive_indices.sort(key=lambda i: float(bm25_scores[i]), reverse=True)

        positive_indices = positive_indices[:candidate_k]

        for rank, raw_index in enumerate(positive_indices):

            index = int(raw_index)

            score = float(bm25_scores[index])

            bm25_scores_lookup[index] = max(

                bm25_scores_lookup.get(index, float("-inf")), score

            )

            rrf_scores[index] = rrf_scores.get(index, 0.0) + (

                1.0 / (rrf_constant + rank + 1)

            )

    # Definition questions need explanatory passages, not isolated headings.

    if is_definition:

        subject = extract_definition_subject(query_variants[0])

        subject_terms = [

            term for term in tokenize_text(subject)

            if len(term) > 2

        ]

        definition_cues = [

            " is a ", " are a ", " refers to ", " means ",

            " is defined as ", " consists of ", " can be defined ",

            " is a technique ", " is a method ", " is the process ",

            " is used to ", " allows ", " enables "

        ]

        # Also consider chunks with direct lexical evidence even if their

        # semantic rank was just outside the normal candidate window.

        if subject_terms:

            for index, chunk in enumerate(chunks):

                text_lower = chunk["text"].lower()

                if any(term in text_lower for term in subject_terms):

                    if any(cue in f" {text_lower} " for cue in definition_cues):

                        rrf_scores.setdefault(index, 0.0)

                        semantic_scores_lookup.setdefault(index, 0.0)

                        bm25_scores_lookup.setdefault(index, 0.0)

        for index in list(rrf_scores):

            chunk_text = chunks[index]["text"].lower()

            has_subject = not subject_terms or any(

                term in chunk_text for term in subject_terms

            )

            has_definition_cue = any(

                cue in f" {chunk_text} " for cue in definition_cues

            )

            if has_subject and has_definition_cue:

                rrf_scores[index] += 0.025

            elif has_subject:

                rrf_scores[index] += 0.003

            if is_heading_only_chunk(chunks[index]["text"]):

                rrf_scores[index] *= 0.60

    ranked_chunks = sorted(

        rrf_scores.items(), key=lambda item: item[1], reverse=True

    )

    if top_k is not None:

        ranked_chunks = ranked_chunks[:top_k]

    results = []

    for index, fusion_score in ranked_chunks:

        results.append({

            "chunk_number": index + 1,

            "text": chunks[index]["text"],

            "pages": chunks[index]["pages"],

            "score": semantic_scores_lookup.get(index, 0.0),

            "bm25_score": bm25_scores_lookup.get(index, 0.0),

            "fusion_score": fusion_score

        })

    return results

def expand_query_with_gemini(question):

    prompt = f"""

Generate exactly four alternative search queries for a PDF question.

Preserve the meaning, use different wording, and do not answer.

Return only four lines with no numbering or commentary.

If the question asks "what is", "what are", "define", or asks for a

meaning/definition, make sure the alternatives search for:

- the definition and meaning of the concept

- wording such as "refers to", "means", "is a technique", or "is a method"

- what the concept consists of and its purpose

- how the concept works, if the PDF explains it

Do not turn a definition question into a list-only question.

Question: {question}

"""

    try:

        response = client.models.generate_content(

            model=GEMINI_MODEL,

            contents=prompt,

            config={"max_output_tokens": 250, "temperature": 0.2}

        )

        alternatives = []

        for line in (response.text or "").splitlines():

            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()

            line = line.strip('"').strip("'").strip()

            if line and line.lower() != question.lower():

                if line.lower() not in [x.lower() for x in alternatives]:

                    alternatives.append(line)

        return {"queries": [question] + alternatives[:4], "error": None}

    except Exception as error:

        return {"queries": [question], "error": str(error)}

# =========================================================

# 5. QUESTION TYPE DETECTION

# =========================================================

def detect_document_wide_mode(question):

    q = question.lower().strip()

    summary_patterns = [

        r"\bsummary\b", r"\bsummarize\b", r"\bsummarise\b",

        r"\boverview of (?:this|the) document\b",

        r"\bmain points\b", r"\bkey takeaways\b",

        r"\bwhat is this document about\b",

        r"\bbrief(?:ly)? describe (?:this|the) document\b"

    ]

    technology_patterns = [

        r"\btechnolog(?:y|ies)\b", r"\btech stack\b",

        r"\btools and frameworks\b", r"\bsoftware and tools\b",

        r"\blibraries\b", r"\bprogramming languages\b",

        r"\bplatforms\b", r"\btechnical components\b"

    ]

    topics_patterns = [

        r"\bmain topics\b", r"\btopics covered\b",

        r"\bkey concepts\b", r"\bsubject areas\b"

    ]

    if any(re.search(p, q) for p in summary_patterns):

        return "summary"

    if any(re.search(p, q) for p in technology_patterns) and any(

        term in q for term in ["document", "pdf", "mentioned", "all", "list", "this"]

    ):

        return "technologies"

    if any(re.search(p, q) for p in topics_patterns):

        return "topics"

    if re.search(r"\b(list|identify|extract)\b", q) and any(

        term in q for term in ["technolog", "tool", "framework", "library", "software"]

    ):

        return "technologies"

    return None

def chunk_source_label(chunk):

    pages = ", ".join(str(p) for p in chunk["pages"])

    return f"[Page {pages}, Chunk {chunk['chunk_number']}]"

def build_labeled_context(chunks):

    return "\n\n".join(

        f"{chunk_source_label(chunk)}\n{chunk['text']}"

        for chunk in chunks

    )

# =========================================================

# 6. DOCUMENT-WIDE BATCH PROCESSING

# =========================================================

def split_into_batches(chunks, batch_size=12):

    return [

        chunks[i:i + batch_size]

        for i in range(0, len(chunks), batch_size)

    ]

def call_gemini(prompt, max_tokens=1800, temperature=0.2):

    response = client.models.generate_content(

        model=GEMINI_MODEL,

        contents=prompt,

        config={"max_output_tokens": max_tokens, "temperature": temperature}

    )

    return (response.text or "").strip()

def generate_document_wide_answer(mode, question, all_chunks):

    # Keep stable document-wide chunk IDs before splitting into batches.

    numbered_chunks = []

    for index, chunk in enumerate(all_chunks, start=1):

        numbered_chunk = dict(chunk)

        numbered_chunk["chunk_number"] = index

        numbered_chunks.append(numbered_chunk)

    batches = split_into_batches(numbered_chunks, batch_size=12)

    if not batches:

        return "I could not find readable text in this PDF.", []

    batch_outputs = []

    used_sources = []

    if mode == "summary":

        task = """

Summarize the PDF passages in this batch. Identify the main subject,

sections, concepts, methods, examples, and conclusions if present.

Do not add outside knowledge. Every factual bullet or sentence must

end with citations copied exactly from the supplied passages.

If a batch contains only a fragment, summarize only that fragment.

"""

    elif mode == "technologies":

        task = """

Extract technical items explicitly mentioned in these PDF passages.

Include programming languages, frameworks, libraries, software,

platforms, protocols, hardware, tools, and technical systems when named.

Do not mistake general nouns or networking concepts for named products.

Group related items by type when possible. Do not infer technologies.

For every item, include the exact page/chunk citation from the passage.

If no such item appears in this batch, write NONE.

"""

    else:

        task = """

Identify the main topics and concepts covered in these PDF passages.

Use concise topic names and one-sentence descriptions.

Every factual item must include citations copied from the passages.

Do not add outside knowledge.

"""

    progress = st.progress(0, text="Reading the complete document in batches...")

    for index, batch in enumerate(batches):

        context = build_labeled_context(batch)

        prompt = f"""

You are analyzing a portion of an uploaded PDF.

Use only the source passages below. Do not invent details.

A citation is valid only if its exact page and chunk label appears below.

TASK:

{task}

USER'S REQUEST:

{question}

PDF BATCH {index + 1} OF {len(batches)}:

{context}

Return useful, concise findings with citations. Do not provide a generic

answer if the batch does not contain relevant information.

"""

        output = call_gemini(prompt, max_tokens=1600)

        if output and output.upper() != "NONE":

            batch_outputs.append(output)

            used_sources.extend(batch)

        progress.progress(

            (index + 1) / len(batches),

            text=f"Processed batch {index + 1} of {len(batches)}"

        )

    progress.empty()

    if not batch_outputs:

        return "I could not find this information in the PDF.", []

    combined = "\n\n".join(

        f"Findings from batch {i + 1}:\n{result}"

        for i, result in enumerate(batch_outputs)

    )

    if mode == "summary":

        final_instruction = """

Create a coherent document-wide summary from the batch findings.

Organize it into an overview and the main topics/sections. Remove

repetition, preserve important coverage across the document, and keep

citations beside every factual statement. Do not add facts that are

not present in the batch findings. If the document has no explicit

conclusion, do not invent one.

"""

    elif mode == "technologies":

        final_instruction = """

Combine the extracted items into one deduplicated inventory.

Group them under suitable headings such as languages, frameworks,

libraries, tools/software, platforms, protocols, or hardware only

where applicable. Keep the citation(s) for each item. Do not add

items based on outside knowledge. If the PDF mentions a concept but

not a named technology, do not label it as a product or tool.

"""

    else:

        final_instruction = """

Combine the findings into a concise document-wide topic outline.

Deduplicate repeated topics while preserving coverage of distinct

sections. Keep citations beside each topic description.

"""

    final_prompt = f"""

You are consolidating findings from every batch of one PDF.

Use only the findings below. Preserve their citations exactly.

Do not create new citations or introduce outside information.

{final_instruction}

USER REQUEST:

{question}

BATCH FINDINGS:

{combined}

Produce the final answer in clear Markdown.

"""

    final_answer = call_gemini(final_prompt, max_tokens=2400)

    source_map = {}

    for source in used_sources:

        source_map[source["chunk_number"]] = source

    final_sources = [source_map[k] for k in sorted(source_map)]

    return final_answer, final_sources

# =========================================================

# 7. FOLLOW-UP HANDLING

# =========================================================

def is_follow_up_question(question):

    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()

    if not normalized:

        return False

    if re.search(

        r"^(what about|how about|and|also|what does it|how does it|"

        r"why does it|where does it|what are its|explain its|"

        r"describe its|tell me more)\b", normalized

    ):

        return True

    if re.search(r"\b(it|its|they|them|their)\b", normalized):

        return True

    elliptical = {

        "technologies", "technology", "tools", "libraries", "features",

        "functionality", "workflow", "implementation", "architecture",

        "dataset", "classification", "accuracy", "advantages", "limitations",

        "applications", "tools used", "libraries used", "tech stack",

        "programming languages", "algorithm", "training process",

        "benefits", "drawbacks", "performance", "purpose", "objective"

    }

    return normalized in elliptical

def resolve_retrieval_question(question, previous_user_question):

    if is_follow_up_question(question) and previous_user_question:

        return f"{previous_user_question}. Follow-up: {question}"

    return question

# =========================================================

# 8. CITATION LOCATION VALIDATION AND REPAIR

# =========================================================

def get_citation_pattern():
    """Match one bracketed citation group, including multiple Page/Chunk pairs."""
    return re.compile(
        r"\[(?:\s*Page\s+[\d,\s]+,\s*Chunk\s+\d+\s*,?)+\s*\]",
        re.IGNORECASE
    )


def extract_citation_pairs(citation_text):
    """Return every (page-list, chunk-number) pair in a citation group."""
    return [
        (page_text.strip().rstrip(","), int(chunk_text))
        for page_text, chunk_text in re.findall(
            r"Page\s+([\d,\s]+?)\s*,\s*Chunk\s+(\d+)",
            citation_text,
            re.IGNORECASE
        )
    ]


def is_citation_group_valid(citation_text, lookup):
    pairs = extract_citation_pairs(citation_text)
    return bool(pairs) and all(
        is_citation_valid(page_text, chunk_number, lookup)
        for page_text, chunk_number in pairs
    )

def build_source_lookup(sources):

    lookup = {}

    for source in sources:

        lookup[source["chunk_number"]] = {

            int(page) for page in source["pages"]

        }

    return lookup

def is_citation_valid(page_text, chunk_text, lookup):
    cited_pages = {int(x) for x in re.findall(r"\d+", str(page_text))}
    chunk_number = int(chunk_text)
    return (
        chunk_number in lookup
        and bool(cited_pages)
        and cited_pages.issubset(lookup[chunk_number])
    )

def find_invalid_citations(answer, sources):
    lookup = build_source_lookup(sources)
    invalid = []
    for match in get_citation_pattern().finditer(answer):
        if not is_citation_group_valid(match.group(0), lookup):
            invalid.append(match.group(0))
    return invalid

def repair_invalid_citations(answer, sources):

    """

    Ask Gemini to rewrite the draft using the actual retrieved source labels.

    This is attempted only when the first answer contains invalid references.

    """

    invalid_before = find_invalid_citations(answer, sources)

    if not invalid_before:

        return answer, {

            "attempted": False,

            "success": True,

            "message": "No citation repair was needed.",

            "invalid_before": [],

            "invalid_after": []

        }

    context = build_labeled_context(sources)

    prompt = f"""

You are repairing citations in an answer produced by a PDF question-answering

system. The answer may contain incorrect page/chunk references.

Use ONLY the PDF source passages below as factual evidence.

Rewrite the answer so that:

1. Keep every claim only if the supplied passages support it.

2. Attach a citation immediately to every factual sentence or bullet.

3. Copy citation labels EXACTLY from the supplied source passages.

4. Never invent, guess, renumber, or modify a page or chunk number.

5. If the sources do not support a claim, remove that claim.

6. Preserve useful headings and the answer's clear Markdown structure.

7. Do not mention that you are repairing citations.

8. If the sources do not answer the question, say:

   "I could not find this information in the PDF."

Valid source labels and passages:

{context}

Draft answer that needs citation repair:

{answer}

Return ONLY the corrected answer.

"""

    try:

        repaired = call_gemini(prompt, max_tokens=2200, temperature=0.0)

        if not repaired:

            raise ValueError("Gemini returned an empty repaired answer.")

        invalid_after = find_invalid_citations(repaired, sources)

        return repaired, {

            "attempted": True,

            "success": len(invalid_after) == 0,

            "message": (

                "Citation references were repaired and rechecked."

                if not invalid_after else

                "A repair was attempted, but some invalid references remain."

            ),

            "invalid_before": invalid_before,

            "invalid_after": invalid_after

        }

    except Exception as error:

        return answer, {

            "attempted": True,

            "success": False,

            "message": f"Citation repair could not be completed: {error}",

            "invalid_before": invalid_before,

            "invalid_after": invalid_before

        }

def remove_remaining_invalid_citation_lines(answer, sources):
    """Remove only lines containing a citation group with an invalid pair."""
    lookup = build_source_lookup(sources)
    pattern = get_citation_pattern()
    kept_lines, removed = [], []
    for line in answer.splitlines():
        matches = list(pattern.finditer(line))
        invalid = [m for m in matches if not is_citation_group_valid(m.group(0), lookup)]
        if invalid:
            removed.extend(m.group(0) for m in invalid)
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines).strip()
    if not cleaned and removed:
        cleaned = "I could not provide a reliable cited answer from the retrieved PDF passages."
    return cleaned, removed

def validate_citations(answer, sources):
    pattern = get_citation_pattern()
    lookup = build_source_lookup(sources)
    matches = list(pattern.finditer(answer))
    valid, invalid = [], []
    for match in matches:
        for page_text, chunk_number in extract_citation_pairs(match.group(0)):
            label = f"Page {page_text}, Chunk {chunk_number}"
            if is_citation_valid(page_text, chunk_number, lookup):
                valid.append({"citation": label})
            else:
                invalid.append({"citation": label, "reason": "Citation does not match the source chunks used."})
    return {
        "total_citations": sum(len(extract_citation_pairs(m.group(0))) for m in matches),
        "valid_citations": valid,
        "invalid_citations": invalid,
        "no_citations": len(matches) == 0
    }

def strip_markdown_prefix(line):

    return re.sub(

        r"^\s*(?:#{1,6}\s*|[-*+]\s+|\d+[.)]\s+)", "", line

    ).strip()

def extract_claims_for_audit(answer):
    """Extract factual answer lines while ignoring headings and lead-ins."""
    claims = []
    citation_pattern = get_citation_pattern()
    intro_patterns = [
        r"^based on (?:the )?(?:provided )?(?:pdf|document|sources?),?\s+",
        r"^here(?:'s| is)\s+(?:what|an overview|a summary|the)\b",
        r"^the following\s+(?:are|is)\b",
        r"^according to (?:the )?(?:pdf|document|provided text),?\s*$"
    ]
    factual_verb_pattern = re.compile(
        r"\b(is|are|was|were|has|have|had|uses|used|using|provides|provide|"
        r"contains|consists|includes|include|allows|allow|enables|enable|"
        r"supports|support|connects|connect|creates|create|means|refers|"
        r"defines|defined|works|work|operates|operating|requires|require|"
        r"transmits|transmit|divides|divide|establishes|establish|"
        r"allocates|allocate|depends|depend|offers|offer|forms|form|"
        r"communicates|communicate)\b", re.IGNORECASE
    )
    for line_number, raw_line in enumerate(answer.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("```") or re.fullmatch(r"[-*_`#\s]+", line):
            continue
        if re.match(r"^\s*#{1,6}\s+", line):
            continue
        cleaned = strip_markdown_prefix(line)
        if not cleaned or cleaned.startswith("#"):
            continue
        plain = citation_pattern.sub("", cleaned)
        plain = re.sub(r"[*_`~]", "", plain).strip()
        if not plain or plain.endswith(":"):
            continue
        if any(re.search(p, plain, re.IGNORECASE) for p in intro_patterns):
            continue
        has_factual_verb = bool(factual_verb_pattern.search(plain))
        if len(plain.split()) <= 10 and not has_factual_verb:
            continue
        if not has_factual_verb and re.match(
            r"^(in this section|in summary|overall|to summarize|the document)\b", plain, re.IGNORECASE
        ):
            continue
        claims.append({
            "line_number": line_number,
            "claim": cleaned,
            "citations": [m.group(0) for m in citation_pattern.finditer(line)]
        })
    return claims

def extract_json_object(text):

    if not text:

        raise ValueError("The model returned an empty response.")

    cleaned = re.sub(

        r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.I

    )

    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    start = cleaned.find("{")

    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:

        raise ValueError("No JSON object was found in the model response.")

    return json.loads(cleaned[start:end + 1])

def semantic_citation_audit(answer, sources):

    claims = extract_claims_for_audit(answer)

    if not claims:

        return {

            "status": "not_run",

            "message": "There were no prose claims to evaluate.",

            "claims": [],

            "counts": {}

        }

    source_by_chunk = {

        source["chunk_number"]: source for source in sources

    }

    citation_pattern = get_citation_pattern()

    source_lookup = build_source_lookup(sources)

    judge_items = []

    audited_claims = []

    for claim_id, item in enumerate(claims, start=1):

        valid_sources = []

        invalid_citations = []

        for citation in item["citations"]:
            pairs = extract_citation_pairs(citation)
            if not pairs:
                invalid_citations.append(citation)
                continue
            for page_text, chunk_number in pairs:
                if not is_citation_valid(page_text, chunk_number, source_lookup):
                    invalid_citations.append(f"[Page {page_text}, Chunk {chunk_number}]")
                    continue
                source = source_by_chunk.get(chunk_number)
                if source:
                    valid_sources.append(source)

        unique_sources = {}

        for source in valid_sources:

            unique_sources[source["chunk_number"]] = source

        valid_sources = list(unique_sources.values())

        record = {

            "claim_id": claim_id,

            "line_number": item["line_number"],

            "claim": item["claim"],

            "citations": item["citations"],

            "status": None,

            "reason": "",

            "evidence": [

                {

                    "chunk_number": s["chunk_number"],

                    "pages": s["pages"],

                    "text": s["text"]

                }

                for s in valid_sources

            ],

            "invalid_citations": invalid_citations

        }

        if invalid_citations:

            record["status"] = "unsupported"

            record["reason"] = (

                "At least one citation does not match the retrieved source chunks."

            )

        elif not item["citations"]:

            record["status"] = "missing_citation"

            record["reason"] = "No citation was attached to this answer line."

        elif not valid_sources:

            record["status"] = "uncertain"

            record["reason"] = "No usable cited source text was available."

        else:

            evidence_text = "\n\n".join(

                f"[Page {', '.join(map(str, s['pages']))}, "

                f"Chunk {s['chunk_number']}]\n{s['text']}"

                for s in valid_sources

            )

            judge_items.append({

                "claim_id": claim_id,

                "claim": item["claim"],

                "evidence": evidence_text

            })

        audited_claims.append(record)

    if judge_items:

        judge_prompt = f"""

You are an evidence verifier for a PDF question-answering system.

Compare each claim ONLY with its cited source excerpt(s).

Do not use outside knowledge. Do not assume a claim is true because

it sounds plausible.

Choose exactly one status:

- supported: the cited excerpt directly supports the full claim.

- partially_supported: the excerpt supports only part of the claim.

- unsupported: the excerpt does not support the claim or contradicts it.

- uncertain: the excerpt is too ambiguous or incomplete to decide.

Return ONLY valid JSON:

{{

  "results": [

    {{

      "claim_id": 1,

      "status": "supported",

      "reason": "Brief explanation based only on the excerpt."

    }}

  ]

}}

Evaluate every claim ID exactly once.

CLAIMS AND CITED EVIDENCE:

{json.dumps(judge_items, ensure_ascii=False)}

"""

        try:

            response_text = call_gemini(

                judge_prompt, max_tokens=1800, temperature=0.0

            )

            parsed = extract_json_object(response_text)

            results = parsed.get("results", [])

            allowed = {

                "supported", "partially_supported", "unsupported", "uncertain"

            }

            result_map = {}

            for result in results:

                try:

                    cid = int(result.get("claim_id"))

                except (TypeError, ValueError):

                    continue

                status = str(result.get("status", "uncertain")).lower().strip()

                if status not in allowed:

                    status = "uncertain"

                result_map[cid] = {

                    "status": status,

                    "reason": str(result.get("reason", "")).strip()

                }

            for record in audited_claims:

                if record["status"] is not None:

                    continue

                judged = result_map.get(record["claim_id"])

                if judged:

                    record["status"] = judged["status"]

                    record["reason"] = judged["reason"] or (

                        "Gemini compared the claim with its cited evidence."

                    )

                else:

                    record["status"] = "uncertain"

                    record["reason"] = (

                        "The verifier did not return a result for this claim."

                    )

        except Exception as error:

            for record in audited_claims:

                if record["status"] is None:

                    record["status"] = "uncertain"

                    record["reason"] = (

                        "Semantic verification could not be completed: "

                        f"{error}"

                    )

    counts = {}

    for record in audited_claims:

        counts[record["status"]] = counts.get(record["status"], 0) + 1

    return {

        "status": "completed",

        "message": (

            "AI-assisted evidence check. Treat the labels as a review aid, "

            "not a formal proof of factual correctness."

        ),

        "claims": audited_claims,

        "counts": counts

    }

def display_semantic_audit(audit):

    if not audit:

        return

    st.markdown("**AI evidence check**")

    st.caption(audit.get(

        "message",

        "AI-assisted comparison of answer claims with cited PDF passages."

    ))

    if audit.get("status") != "completed":

        st.info(audit.get("message", "Semantic verification was not completed."))

        return

    counts = audit.get("counts", {})

    metric_columns = st.columns(5)

    metric_items = [

        ("Supported", counts.get("supported", 0)),

        ("Partial", counts.get("partially_supported", 0)),

        ("Unsupported", counts.get("unsupported", 0)),

        ("Uncertain", counts.get("uncertain", 0)),

        ("No citation", counts.get("missing_citation", 0))

    ]

    for column, (label, value) in zip(metric_columns, metric_items):

        with column:

            st.metric(label, value)

    with st.expander("Review claim-by-claim evidence", expanded=False):

        for record in audit.get("claims", []):

            status = record.get("status", "uncertain")

            label = status.replace("_", " ").title()

            st.markdown(

                f"**Claim {record['claim_id']} — {label}** "

                f"(answer line {record['line_number']})"

            )

            st.write(record["claim"])

            st.caption(record.get("reason", ""))

            if record.get("citations"):

                st.caption("Citations: " + ", ".join(record["citations"]))

            if record.get("invalid_citations"):

                st.error(

                    "Invalid citation(s): "

                    + ", ".join(record["invalid_citations"])

                )

            for evidence in record.get("evidence", []):

                pages = ", ".join(map(str, evidence["pages"]))

                st.markdown(

                    f"**Evidence — Page(s) {pages}, "

                    f"Chunk {evidence['chunk_number']}**"

                )

                st.write(evidence["text"])

            st.divider()

# =========================================================

# 10. SESSION STATE

# =========================================================

DEFAULT_STATE = {

    "chunks": [],

    "index": None,

    "bm25_index": None,

    "chat_history": [],

    "processed_file": None,

    "pending_question": None,

    "document_name": None

}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value

# =========================================================

# 11. SIDEBAR AND PDF UPLOAD

# =========================================================

with st.sidebar:

    st.title("📚 PDF AI Assistant")

    st.caption("Your personal document assistant")

    st.divider()

    st.subheader("Upload Document")

    uploaded_file = st.file_uploader(

        "Choose a PDF file", type=["pdf"]

    )

    st.divider()

    st.subheader("Document Information")

    if st.session_state.index is not None:

        st.success("Document processed successfully")

        st.write("**File:**", st.session_state.document_name)

        st.metric("Total Chunks", len(st.session_state.chunks))

        st.metric("FAISS Vectors", st.session_state.index.ntotal)

    else:

        st.info("Upload a PDF to get started.")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):

        st.session_state.chat_history = []

        st.rerun()

# =========================================================

# 12. PROCESS THE PDF

# =========================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    chunking_version = "document-wide-batch-v2-definition-aware-retrieval"

    file_hash = hashlib.sha256(

        file_bytes + chunking_version.encode()

    ).hexdigest()

    if file_hash != st.session_state.processed_file:

        st.session_state.chunks = []

        st.session_state.index = None

        st.session_state.bm25_index = None

        st.session_state.chat_history = []

        st.session_state.document_name = None

        st.session_state.processed_file = None

        try:

            with st.spinner("Processing your PDF..."):

                reader = PdfReader(io.BytesIO(file_bytes))

                pages = []

                for page_number, page in enumerate(reader.pages, start=1):

                    page_text = page.extract_text()

                    if page_text and page_text.strip():

                        pages.append({

                            "page_number": page_number,

                            "text": page_text.strip()

                        })

                if not pages:

                    st.error(

                        "No readable text was extracted. This may be a scanned "

                        "PDF; OCR would be needed for that document."

                    )

                else:

                    chunks = create_chunks(pages, chunk_size=500)

                    if not chunks:

                        st.error("No text chunks could be created.")

                    else:

                        texts = [chunk["text"] for chunk in chunks]

                        embeddings = np.asarray(

                            model.encode(texts, show_progress_bar=False),

                            dtype="float32"

                        )

                        faiss.normalize_L2(embeddings)

                        index = faiss.IndexFlatIP(embeddings.shape[1])

                        index.add(embeddings)

                        st.session_state.chunks = chunks

                        st.session_state.index = index

                        st.session_state.bm25_index = build_bm25_index(chunks)

                        st.session_state.document_name = uploaded_file.name

                        st.session_state.processed_file = file_hash

                        st.success("PDF processed successfully!")

        except Exception as error:

            st.session_state.chunks = []

            st.session_state.index = None

            st.session_state.bm25_index = None

            st.session_state.processed_file = None

            st.error(f"Error processing PDF: {error}")

# =========================================================

# 13. MAIN DASHBOARD

# =========================================================

st.title("PDF AI Assistant")

st.write(

    "Chat with your documents and get answers with references "

    "to the original PDF."

)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(

        "Document Status",

        "Ready" if st.session_state.index is not None else "Waiting"

    )

with col2:

    st.metric("Total Chunks", len(st.session_state.chunks))

with col3:

    count = st.session_state.index.ntotal if st.session_state.index else 0

    st.metric("Total Vectors", count)

st.divider()

# =========================================================

# 14. SUGGESTED QUESTIONS

# =========================================================

if st.session_state.index is not None:

    st.subheader("Suggested Questions")

    left, right = st.columns(2)

    with left:

        if st.button("Summarize this document", use_container_width=True):

            st.session_state.pending_question = "Give me a summary of this document."

            st.rerun()

        if st.button("What are the main topics?", use_container_width=True):

            st.session_state.pending_question = "What are the main topics in this document?"

            st.rerun()

    with right:

        if st.button("Find important details", use_container_width=True):

            st.session_state.pending_question = "Tell me the important details in this document."

            st.rerun()

        if st.button("List the technologies", use_container_width=True):

            st.session_state.pending_question = "List the technologies mentioned in this document."

            st.rerun()

# =========================================================

# 15. DISPLAY CHAT HISTORY

# =========================================================

st.subheader("Conversation")

if not st.session_state.chat_history:

    st.info(

        "Your conversation will appear here. Upload a PDF and ask your first question!"

    )

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        validation = message.get("citation_validation")

        if message["role"] == "assistant" and validation:

            if validation["invalid_citations"]:

                st.warning("Some citations failed reference validation.")

            elif validation.get("removed_citations", 0):

                st.warning(

                    f"{validation['removed_citations']} invalid citation(s) "

                    "were removed after the repair attempt."

                )

            elif validation["no_citations"]:

                st.warning("No citations were detected in this answer.")

            else:

                st.success(

                    f"{len(validation['valid_citations'])} citation reference(s) "

                    "match the retrieved page/chunk locations."

                )

        repair_info = message.get("citation_repair")

        if message["role"] == "assistant" and repair_info:

            if repair_info.get("attempted"):

                if repair_info.get("success"):

                    st.info("Citation repair was attempted and passed location validation.")

                else:

                    st.warning(repair_info.get("message", "Citation repair was incomplete."))

        if message["role"] == "assistant" and message.get("semantic_audit"):

            display_semantic_audit(message["semantic_audit"])

        if message.get("expanded_queries") and not message.get("document_mode"):

            with st.expander("View search queries used"):

                st.markdown("**Original question:**")

                st.write(message["expanded_queries"][0])

                alternatives = message["expanded_queries"][1:]

                if alternatives:

                    st.markdown("**Generated alternative queries:**")

                    for number, query_text in enumerate(alternatives, start=1):

                        st.write(f"{number}. {query_text}")

                if message.get("query_expansion_error"):

                    st.warning(message["query_expansion_error"])

        if message.get("sources"):

            with st.expander("View retrieved sources"):

                for source in message["sources"]:

                    pages_text = ", ".join(str(p) for p in source["pages"])

                    st.markdown(

                        f"**Chunk {source['chunk_number']}** | "

                        f"Page(s): {pages_text} | "

                        f"Semantic: {source.get('score', 0):.4f} | "

                        f"BM25: {source.get('bm25_score', 0):.4f} | "

                        f"RRF: {source.get('fusion_score', 0):.4f}"

                    )

                    st.write(source["text"])

# =========================================================

# 16. CHAT INPUT

# =========================================================

question = st.chat_input("Ask anything about your PDF...")

if st.session_state.pending_question is not None:

    question = st.session_state.pending_question

    st.session_state.pending_question = None

# =========================================================

# 17. PROCESS THE QUESTION

# =========================================================

if question:

    if st.session_state.index is None:

        st.warning("Please upload a readable PDF first.")

        st.stop()

    previous_history = st.session_state.chat_history.copy()

    previous_user_question = None

    for message in reversed(previous_history):

        if message["role"] == "user":

            previous_user_question = message["content"]

            break

    st.session_state.chat_history.append({

        "role": "user",

        "content": question

    })

    retrieval_question = resolve_retrieval_question(

        question, previous_user_question

    )

    answer = ""

    sources = []

    citation_validation = None

    semantic_audit = None

    citation_repair = None

    removed_citations = []

    expanded_queries = []

    query_expansion_error = None

    document_mode = detect_document_wide_mode(question)

    try:

        if document_mode:

            with st.spinner(

                "Analyzing the entire document. This may take a little longer..."

            ):

                answer, sources = generate_document_wide_answer(

                    document_mode,

                    question,

                    st.session_state.chunks

                )

        else:

            with st.spinner("Generating alternative search queries..."):

                expansion = expand_query_with_gemini(retrieval_question)

            expanded_queries = expansion["queries"]

            query_expansion_error = expansion["error"]

            with st.spinner("Searching the PDF..."):

                sources = hybrid_search(

                    expanded_queries,

                    st.session_state.index,

                    st.session_state.bm25_index,

                    st.session_state.chunks,

                    top_k=12

                )

            if not sources:

                answer = "I could not find this information in the PDF."

            else:

                context = build_labeled_context(sources)

                history_text = "\n".join(

                    f"{m['role'].capitalize()}: {m['content']}"

                    for m in previous_history[-6:]

                ) or "No previous conversation."

                prompt = f"""

You are a careful tutor answering questions about an uploaded PDF.

Use ONLY the retrieved PDF context as factual evidence.

Do not use outside knowledge or invent missing details.

Use conversation history only to understand follow-up references, never as evidence.

Every factual statement must have an immediately relevant citation.

Use exactly this format: [Page 1, Chunk 3].

Cite only source labels included in the retrieved context.

A citation must genuinely support the claim beside it.

For definition questions (for example, "What is X?"), prioritize passages

that actually explain or define X. Do not treat a heading, title, or a list

of names as a definition. If the retrieved passages contain a definition,

explain it clearly in your own words while preserving the PDF's meaning.

If the PDF only lists types or mentions the term without defining it, say

that the retrieved passages do not provide a complete definition.

If information is missing, say so clearly.

If no relevant information is available, say:

"I could not find this information in the PDF."

Explain clearly and avoid repetition. Use headings or bullets when useful.

Previous conversation:

{history_text}

Retrieved PDF context:

{context}

Original user question:

{question}

Resolved retrieval question:

{retrieval_question}

Answer the question using the PDF context and cite every factual claim.

"""

                with st.spinner("Generating your answer..."):

                    answer = call_gemini(prompt, max_tokens=2000)

        # -------------------------------------------------

        # Citation repair and validation pipeline

        # -------------------------------------------------

        if sources and answer:

            invalid_before_repair = find_invalid_citations(answer, sources)

            if invalid_before_repair:

                with st.spinner(

                    "Correcting citation references using the retrieved PDF passages..."

                ):

                    answer, citation_repair = repair_invalid_citations(

                        answer, sources

                    )

            else:

                citation_repair = {

                    "attempted": False,

                    "success": True,

                    "message": "No citation repair was needed.",

                    "invalid_before": [],

                    "invalid_after": []

                }

            # Final local validation. Invalid citation lines are removed only

            # if the repair attempt still failed to correct them.

            answer, removed_citations = remove_remaining_invalid_citation_lines(

                answer, sources

            )

            citation_validation = validate_citations(answer, sources)

            citation_validation["removed_citations"] = len(removed_citations)

            citation_validation["removed_citation_details"] = removed_citations

            with st.spinner("Checking whether citations support the answer..."):

                semantic_audit = semantic_citation_audit(answer, sources)

    except Exception as error:

        answer = (

            "Sorry, an error occurred while processing your question: "

            f"{error}"

        )

        sources = []

        citation_validation = None

        semantic_audit = None

        citation_repair = None

    st.session_state.chat_history.append({

        "role": "assistant",

        "content": answer,

        "sources": sources,

        "citation_validation": citation_validation,

        "citation_repair": citation_repair,

        "semantic_audit": semantic_audit,

        "expanded_queries": expanded_queries,

        "query_expansion_error": query_expansion_error,

        "document_mode": document_mode

    })

    st.rerun()
