from fuzzywuzzy import process

def fuzzy_correct_value(value, valid_values, threshold=80):
    """
    Fix value if not exactly matching. 
    Returns closest match or original if none good enough.
    """
    match, score = process.extractOne(value, valid_values)
    return match if score >= threshold else value