import sys
import re
from datetime import datetime

def debugprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)

def pad_fields(docs):
    fields = set()
    for doc in docs:
        keys = doc.metadata.keys()
        types = [type(doc.metadata[key]) for key in keys]
        ktypes = list(zip(keys, types))
        fields.update(ktypes)
    for doc in docs:
        for field, ftype in fields:
            if field not in doc.metadata:
                doc.metadata[field] = ftype()

_GOOD_FIELD_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
def rename_fields(docs):
    fields = set()
    for doc in docs:
        keys = doc.metadata.keys()
        fields.update(keys)
    new_fields = {}
    for field in fields:
        new_name = field.replace(" ", "_")
        new_name = "".join(c if c in _GOOD_FIELD_CHARS else '_' for c in new_name)
        new_fields[field] = new_name
    for doc in docs:
        for field, new_name in new_fields.items():
            doc.metadata[new_name] = doc.metadata.pop(field)



def extract_year(creationdate: str) -> str:
    # Case 1: ISO 8601 style
    try:
        dt = datetime.fromisoformat(creationdate)
        return str(dt.year)
    except (ValueError, TypeError):
        pass

    # Case 2: PDF-style like "D:20230919124054Z00'00'"
    match = re.match(r"D:(\d{4})", creationdate)
    if match:
        return match.group(1)

    return "n.d."  # If all else fails



import re

def normalize(text: str) -> str:
    # Lowercase, remove extra spaces, normalize punctuation
    text = text.lower()
    text = re.sub(r'[\n\r]', ' ', text)  # Normalize newlines
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = re.sub(r'([.,;:!?"])', r' \1 ', text)  # Space around punctuation
    text = re.sub(r'\s+', ' ', text)  # Clean again
    return text.strip()

def split_quote(quote: str) -> list:
    # Split at ellipses and punctuation to make checkpoints
    quote = quote.replace('...', '...')  # Ensure clean ellipsis
    parts = re.split(r'\.\.\.|[.,;:!?]', quote)
    parts = [part.strip() for part in parts if part.strip()]
    return parts

def verify_quote_in_source(quote_text: str, source_text: str) -> bool:
    normalized_source = normalize(source_text)
    normalized_quote = normalize(quote_text)

    segments = split_quote(normalized_quote)
    
    cursor = 0  # Start of search
    for segment in segments:
        if len(segment.split()) < 2:
            # Ignore segments that are too short (too fuzzy to be reliable)
            continue
        
        found_at = normalized_source.find(segment, cursor)
        if found_at == -1:
            return False  # Segment not found after last cursor
        cursor = found_at + len(segment)  # Move cursor after this match

    return True
    