import re
import spacy
from spacy.pipeline import EntityRuler
from typing import List, Dict, Tuple, Any
from loguru import logger

# Load spaCy English model (medium model for better accuracy)
try:
    nlp = spacy.load("en_core_web_md")
    logger.info("Loaded spaCy medium model (en_core_web_md)")
except OSError:
    logger.warning("spaCy medium model (en_core_web_md) not found. Falling back to en_core_web_sm.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.load("en")

# ─── Custom EntityRuler — catches patterns spaCy sm misses ────────────────────
# Add BEFORE ner so the ruler's labels take priority
if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    _dept_names = [
        "Engineering", "Security", "Research", "Finance", "Human Resources",
        "Legal", "Product", "Customer Success", "Infrastructure", "Data Science",
    ]
    _custom_features = [
        "AI Assistant", "Authentication", "Analytics",
        "Project Atlas", "Project Orion", "Project Nebula",
        "Atlas", "Orion", "Nebula",
    ]
    
    dept_patterns = [
        {"label": "ORG", "pattern": [{"LOWER": name.lower()}, {"LOWER": "department"}]}
        for name in _dept_names
    ] + [
        # Also catch bare department names used as org references
        {"label": "ORG", "pattern": [{"LOWER": name.lower()}]}
        for name in _dept_names
    ]
    
    feature_patterns = [
        {"label": "PRODUCT", "pattern": [{"LOWER": word.lower()} for word in name.split()]}
        for name in _custom_features
    ]
    
    ruler.add_patterns(dept_patterns + feature_patterns)

# Define target entity types we care about for knowledge graph construction
TARGET_ENTITIES = {
    "ORG",         # Companies, agencies, institutions, departments
    "PERSON",      # People, including fictional
    "GPE",         # Countries, cities, states
    "NORP",        # Nationalities, religious or political groups
    "PRODUCT",     # Objects, vehicles, foods, etc. (often tech stacks/software)
    "LOC",         # Non-GPE locations, mountain ranges, bodies of water
    "FAC",         # Buildings, airports, highways, bridges
    "LAW",         # Named laws, policies
    "WORK_OF_ART", # Books, song titles, etc.
}

# --- Old spaCy extract_entities (basic, without EntityRuler) ---
# def extract_entities(text: str) -> List[Dict[str, str]]:
#     doc = nlp(text)
#     entities = {}
#     for ent in doc.ents:
#         name = ent.text.strip()
#         label = ent.label_
#         if label in TARGET_ENTITIES and len(name) > 1:
#             clean_name = name.replace("\n", " ").strip()
#             if clean_name not in entities:
#                 entities[clean_name] = label
#     return [{"name": name, "label": label} for name, label in entities.items()]

# --- LLM-based extract_entities (commented out — too slow for large document ingestion) ---
# def extract_entities(text: str) -> List[Dict[str, str]]:
#     """Uses gpt-4.1-nano — only suitable for short query strings, NOT full documents."""
#     import os, json
#     from langchain_openai import ChatOpenAI
#     from langchain_core.messages import HumanMessage
#     from src.config import settings
#     from loguru import logger
#     prompt = (
#         "You are an expert entity extractor. Extract the main named entities from the text below.\n"
#         "Specifically look for departments, organizations, people, products, and location names.\n"
#         "Return ONLY a valid JSON: {\"entities\": [{\"name\": ..., \"label\": ...}]}\n\n"
#         f"Text: {text}"
#     )
#     try:
#         api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
#         llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0, api_key=api_key)
#         response = llm.invoke([HumanMessage(content=prompt)])
#         content = response.content.strip().strip("```json").strip("```").strip()
#         data = json.loads(content)
#         return data.get("entities", [])
#     except Exception as e:
#         logger.error(f"LLM entity extraction failed: {e}")
#         return []


def _clean_and_validate_node(name: str) -> str:
    """
    Clean and validate entity or relation node names.
    - Strips leading/trailing '#' characters and whitespace/newlines.
    - Normalizes internal whitespaces/newlines.
    - Strips leading "the " (case-insensitive).
    - Skips if the node starts with '#' or digits followed by whitespace (e.g. '# Rohan', '15\n\n Rohan').
    - Skips if the node is longer than 60 characters or empty/single-char.
    - Skips if the node is a merged phrase or contains verbs/conjunctions that indicate it's a clause.
    """
    if not name:
        return ""
    
    # Normalize whitespaces to check prefix conditions
    norm_temp = re.sub(r"\s+", " ", name).strip()
    if re.match(r"^#\s", norm_temp) or re.match(r"^\d+\s", norm_temp):
        return ""
        
    # Strip leading/trailing '#' and whitespace/newlines/punctuation/quotes
    cleaned = name.strip("# \t\n\r,.-'\"")
    # Normalize internal whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    # Strip leading "the " (case-insensitive)
    if cleaned.lower().startswith("the "):
        cleaned = cleaned[4:].strip()
        
    # Clean again after stripping "the"
    cleaned = cleaned.strip("# \t\n\r,.-'\"")
    
    if len(cleaned) > 60 or len(cleaned) <= 1:
        return ""
        
    # Skip if contains verbs/conjunctions indicating it's a merged sentence chunk
    # e.g., "Aarav Mehta Works", "Project Atlas with", "reports quarterly", etc.
    lower_cleaned = cleaned.lower()
    
    # If it ends with or starts with a verb/conjunction/preposition
    words = lower_cleaned.split()
    if not words:
        return ""
    bad_words = {"with", "to", "from", "for", "and", "in", "on", "of", "about", "works", "contributes", "reports", "attends", "collaborates", "depends"}
    if words[0] in bad_words or words[-1] in bad_words:
        return ""
        
    # If the entity contains a verb like "works", "contributes", "collaborates", "reports" as a separate word, it's a merged sentence
    if re.search(r"\b(works|contributes|collaborates|reports|attends|depends)\b", lower_cleaned):
        return ""
        
    return cleaned


def extract_entities(text: str) -> List[Dict[str, str]]:
    """
    Extract unique entities using improved spaCy NER + custom EntityRuler.

    Improvements over the basic version:
    - EntityRuler pre-labels "X Department" compound patterns as ORG before NER runs.
    - NORP label added (groups, nationalities, team names).
    - Longer-match deduplication: if a shorter entity name is a substring of a
      longer one already seen, prefer the longer one.
    - Whitespace/newline normalisation.
    - Expands entities to full noun chunks if they represent proper names (e.g. "Aarav" -> "Aarav Mehta").
    - Title-cases PERSON names that are not properly capitalized.
    """
    clean_text = re.sub(r"\s+", " ", text).strip()

    # If the input is all-lowercase (typical for user queries), also try
    # a title-cased version so spaCy can recognise proper nouns.
    texts_to_try = [clean_text]
    if clean_text == clean_text.lower() and len(clean_text) < 500:
        texts_to_try.append(clean_text.title())

    # Build a dict: normalised_name -> (original_name, label)
    # For deduplication we prefer the LONGER variant
    entities: Dict[str, tuple] = {}  # key=lower_name, val=(name, label)

    for current_text in texts_to_try:
        doc = nlp(current_text)

        for ent in doc.ents:
            raw_name = ent.text.strip()
            label = ent.label_

            if label not in TARGET_ENTITIES or len(raw_name) < 2:
                continue

            # Expands entity to full proper noun chunk if applicable
            token = ent.root
            expanded_name = raw_name
            if doc.noun_chunks:
                for chunk in doc.noun_chunks:
                    if token in chunk:
                        # Skip leading determiners
                        words = [t.text for t in chunk if t.pos_ != "DET"]
                        chunk_text = " ".join(words).strip()

                        # Title-case if it's a PERSON or consists of proper nouns
                        is_proper_chunk = all(t.pos_ == "PROPN" for t in chunk if t.text.lower() not in ("the", "a", "an"))
                        if label == "PERSON" or is_proper_chunk:
                            chunk_text = chunk_text.title()

                        # Check if all words in the proper noun chunk start with upper case
                        chunk_words = chunk_text.split()
                        if len(chunk_words) <= 4 and all(w[0].isupper() for w in chunk_words if w and w[0].isalpha()):
                            expanded_name = chunk_text
                        break

            # Clean and validate the extracted entity name
            name = _clean_and_validate_node(expanded_name)
            if not name:
                # Fall back to cleaning raw name if the expanded chunk was rejected
                name = _clean_and_validate_node(raw_name)
                if not name:
                    continue

            # Title case PERSON names
            if label == "PERSON":
                name = name.title()

            lower = name.lower()

            # Skip tokens that are purely numeric or single characters
            if re.fullmatch(r"[\d\W]+", name):
                continue

            # Prefer longer names: if a shorter version is already stored, replace it.
            # Also skip if this name is already a substring of a stored longer name.
            dominated = False
            to_delete = []
            for stored_lower, (stored_name, stored_label) in entities.items():
                if lower == stored_lower:
                    # Exact duplicate — keep whichever is longer
                    if len(name) > len(stored_name):
                        to_delete.append(stored_lower)
                    else:
                        dominated = True
                    break
                if lower in stored_lower:
                    # New name is substring of an existing longer name — skip it
                    dominated = True
                    break
                if stored_lower in lower:
                    # Existing entry is substring of new name — replace it
                    to_delete.append(stored_lower)

            for k in to_delete:
                del entities[k]

            if not dominated:
                entities[lower] = (name, label)

    return [{"name": name, "label": label} for name, label in entities.values()]


def extract_svo_triplets(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract (Subject, Verb/Relation, Object) triplets using spaCy dependency parser.
    """
    doc = nlp(text)
    triplets = []
    
    for sent in doc.sents:
        # Process each sentence
        for token in sent:
            # Look for verbs that act as the main relationship
            if token.pos_ == "VERB":
                subj = None
                obj = None
                
                # Find subject and object linked to this verb
                for child in token.children:
                    # Subject relations
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = _get_noun_chunk(child)
                    # Object relations
                    elif child.dep_ in ("dobj", "attr", "oprd"):
                        obj = _get_noun_chunk(child)
                    # Prepositional objects
                    elif child.dep_ == "prep":
                        for prep_child in child.children:
                            if prep_child.dep_ in ("pobj", "pcomp"):
                                obj = _get_noun_chunk(prep_child)
                                
                # If we found both subject and object, create a triplet
                if subj and obj:
                    relation = token.lemma_.lower()  # Normalize verb to its base form
                    # If verb is 'be', include the preposition/adjective if possible
                    if relation == "be":
                        # Look for attributes or prepositions
                        prep_parts = [child.text for child in token.children if child.dep_ in ("prep", "attr")]
                        if prep_parts:
                            relation = f"is {' '.join(prep_parts)}".lower()
                    
                    triplets.append((subj, relation, obj))
                    
    return triplets

def _get_noun_chunk(token) -> str:
    """
    Helper to reconstruct the full noun phrase/chunk for a given subject/object token.
    """
    # Try to grab the full noun chunk if the token is part of one
    if token.doc.noun_chunks:
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                # Filter out leading determiners (the, a, an)
                words = [t.text for t in chunk if t.pos_ != "DET"]
                return " ".join(words).strip()
    
    # Fallback to token subtree for compound nouns
    words = []
    for t in token.subtree:
        # Only take modifiers, compound words, or the noun itself to keep it clean
        if t.dep_ in ("compound", "amod", "flat") or t == token:
            words.append(t.text)
    return " ".join(words).strip()


def extract_knowledge_graph_elements(text: str) -> Dict[str, Any]:
    """
    Combines Entity extraction and Relation/SVO extraction to construct
    knowledge graph nodes and edges.
    """
    entities = extract_entities(text)
    triplets = extract_svo_triplets(text)
    
    # Filter triplets so that subject and object relate to extracted entities if possible,
    # or keep them if they are clean noun phrases.
    cleaned_relations = []
    entity_names = {ent["name"].lower() for ent in entities}
    
    for subj, rel, obj in triplets:
        # Clean and validate the original subject and object first
        cleaned_subj = _clean_and_validate_node(subj)
        cleaned_obj = _clean_and_validate_node(obj)
        if not cleaned_subj or not cleaned_obj:
            continue
            
        # Check if subject/object matches or contains any named entities
        subj_match = next((ent["name"] for ent in entities if ent["name"].lower() in cleaned_subj.lower()), cleaned_subj)
        obj_match = next((ent["name"] for ent in entities if ent["name"].lower() in cleaned_obj.lower()), cleaned_obj)
        
        # Clean and validate the resolved matches
        subj_match = _clean_and_validate_node(subj_match)
        obj_match = _clean_and_validate_node(obj_match)
        
        if subj_match and obj_match and subj_match != obj_match:
            cleaned_relations.append({
                "source": subj_match,
                "type": rel.upper().replace(" ", "_"),
                "target": obj_match
            })
            
    return {
        "entities": entities,
        "relations": cleaned_relations
    }

# ─── Quick Test ───────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     test_text = (
#         "In September 2025, Microsoft partnered with OpenAI to integrate GPT-4.1 into Azure services. "
#         "Satya Nadella emphasized that this collaboration would accelerate enterprise adoption of generative AI. "
#         "Meanwhile, Google announced Gemini 2.0 at its Mountain View headquarters, highlighting multimodal reasoning capabilities. "
#         "The European Union passed the AI Act in December 2025, requiring companies like Meta and Amazon to comply with strict transparency rules. "
#         "At the same time, Neo4j expanded its partnership with Snowflake to enable real-time graph analytics for financial institutions. "
#         "Dr. Priya Sharma, co-founder of Nexus AI Research Institute, explained that adaptive AI twins could transform healthcare by modeling patient histories. "
#         "Salesforce licensed NARI’s technology to enhance its Einstein platform. "
#         "In Tokyo, SoftBank invested $500 million into robotics startups focusing on humanoid assistants. "
#         "Lionel Messi collaborated with FIFA to promote AI-driven match analytics during the 2026 World Cup. "
#         "The leave policy at Nexus AI Research Institute allows researchers 30 days of paid time off annually."
#     )
#     elements = extract_knowledge_graph_elements(test_text)
#     print("Extracted Entities:")
#     for ent in elements["entities"]:
#         print(f"  - {ent['name']} ({ent['label']})")
        
#     print("\nExtracted Relations:")
#     for rel in elements["relations"]:
#         print(f"  - ({rel['source']}) -[{rel['type']}]-> ({rel['target']})")
