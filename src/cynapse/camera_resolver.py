"""Convert an extracted camera mention into one of the ten supported cameras.

Accepts exact names, acronyms, known aliases and small spelling mistakes using
Python rules, without model or database calls. Receives a camera mention such
as "Kranji Highway", not a whole request such as "Show Kranji frames today".
Ambiguous or unrecognized mentions raise CameraResolutionError so the caller
can ask the user to clarify instead of querying the wrong camera.
"""
import re
import unicodedata
from difflib import SequenceMatcher

from cynapse.data_models import Camera


def normalize(mention: str):
    """Put a camera mention into a consistent form for text comparisons.

    Standardizes Unicode width, ignores letter case, replaces dashes and
    parentheses with spaces, and trims whitespace, quotes and sentence punctuation.
    This  cleans text.

    Args:
        mention: Camera text to clean, for example "  Kranji   HIGHWAY ".

    Returns:
        A normalized string, for example "kranji highway".
    """
    text = unicodedata.normalize("NFKC", mention).casefold()
    text = re.sub(r"[-‐‑–—()]", " ", text)
    return " ".join(text.split()).strip(" .!?,\"\'")


def stem(name: str):
    """Extract the distinctive route name from a canonical camera name.

    Normalizes the name, then removes a final "expressway" or "parkway".
    Comparing route names only prevents the shared word "Expressway" from making
    unrelated cameras appear similar.

    Args:
        name: Camera name, for example "Tampines Expressway".

    Returns:
        The normalized route name, for example "tampines".
        If neither suffix is present, returns the normalized name unchanged.
    """
    return re.sub(r" (?:expressway|parkway)$", "", normalize(name))


# Camera enum -> distinctive name, e.g. Camera.TPE -> "tampines".
STEMS = {camera: stem(camera.value) for camera in Camera}
# Normalized accepted name/variation -> Camera enum. Built once on import.
ALIASES = {}
for camera, name in STEMS.items():
    aliases = {
        camera.name,
        camera.value,
        f"{camera.value} ({camera.name})",
        f"{camera.name} ({camera.value})",
        f"{name} highway",
        f"{name} express way",
    }
    # Geographic shorthands; 'Central' alone is too vague.
    if camera != Camera.CTE:
        aliases.add(name)
    for alias in aliases:
        ALIASES[normalize(alias)] = camera


class CameraResolutionError(ValueError):
    """Signal that a camera mention needs clarification before a query can run.

    The exception message explains the problem and may suggest camera acronyms.
    The graph returns this message to the user without executing a database query.
    """
    pass


def edit_distance(left: str, right: str):
    """Count character edits needed to change one string into another.

    Insertion, deletion, substitution and swapping adjacent characters each
    cost one edit. This helper is case-sensitive; callers normalize names first.
    Keeps only three calculation rows rather than a full two-dimensional table.

    Args:
        left: First string, for example "tampins".
        right: Target string, for example "tampines".

    Returns:
        A nonnegative integer edit distance: 0 for identical strings, or 1
        for the example above because inserting "e" corrects the name.
    """
    row = list(range(len(right) + 1))
    prior_row = None
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
            if prior_row is not None and j > 1 and a == right[j - 2] and left[i - 2] == b:
                next_row[-1] = min(next_row[-1], prior_row[j - 2] + 1)
        prior_row, row = row, next_row
    return row[-1]


def resolve_camera(mention: str):
    """Identify a camera from an exact name, alias or sufficiently small typo.

    Checks normalized aliases first, including acronyms and full names.
    Otherwise, rejects short unknown acronyms, alternatives, exclusions and
    extra command text before attempting conservative spelling correction.

    Typo matching compares distinctive route names after removing a recognized
    road suffix. A match needs similarity >= 0.84, a gap >= 0.12 over the next
    candidate, and at most one edit for canonical route names up to eight
    characters or two edits for longer names. Suffix correction separately
    requires similarity >= 0.85 and at most two edits. These scores are matching
    heuristics, not probabilities that the selected camera is correct.

    Args:
        mention: One extracted camera mention, for example "CTE",
            "Kranji Highway" or "Tampines Expresway"; not a full user question.

    Returns:
        The matched Camera enum, for example Camera.KJE for "Kranji Highway".
        Its .value contains the canonical database name, "Kranji Expressway".

    Raises:
        CameraResolutionError: If the mention is unknown, ambiguous, contains
            unsupported extra text, or fails the typo thresholds. Its message
            asks for clarification and may include candidate acronyms.
    """
    name = normalize(mention)
    if name in ALIASES:
        return ALIASES[name]
    # Never fuzzy-match short acronyms, operators, negations or command fragments.
    if len(name) <= 4 or len(name) > 100 or not re.fullmatch(r"[a-z ]+", name):
        raise CameraResolutionError("Please provide one camera's exact acronym or full name.")
    if set(name.split()) & {"and", "or", "not", "except", "both", "all", "ignore", "without"}:
        raise CameraResolutionError("Please select one camera without alternatives or exclusions.")
    if set(name.split()) & {camera.name.casefold() for camera in Camera}:
        raise CameraResolutionError("The camera mention contains extra text. Please provide one camera.")

    # Remove a recognized (or slightly misspelled) road descriptor. Match the
    # distinctive route name so shared 'Expressway' cannot inflate confidence.
    parts = name.split()
    had_descriptor = False
    if len(parts) > 1:
        if any(SequenceMatcher(None, parts[-1], word, autojunk=False).ratio() >= 0.85
               and edit_distance(parts[-1], word) <= 2
               for word in ("expressway", "highway", "parkway", "expy")):
            parts.pop()
            had_descriptor = True
        elif parts[-2:] == ["express", "way"]:
            parts = parts[:-2]
            had_descriptor = True
    route = " ".join(parts)
    if len(route) < 5 or route in {"expressway", "highway", "parkway"} or (route == "central" and not had_descriptor):
        raise CameraResolutionError("Which expressway camera do you mean? Please use its acronym or full name.")
    ranked = sorted(
        ((SequenceMatcher(None, route, candidate, autojunk=False).ratio(), camera)
         for camera, candidate in STEMS.items()), reverse=True,
    )
    score, camera = ranked[0]
    distance = edit_distance(route, STEMS[camera])
    max_edits = 1 if len(STEMS[camera]) <= 8 else 2
    if score >= 0.84 and score - ranked[1][0] >= 0.12 and distance <= max_edits:
        return camera
    suggestions = ", ".join(item[1].name for item in ranked[:2]) if score >= 0.55 else None
    message = f"Did you mean {suggestions}? Please provide one exact camera name or acronym." if suggestions else "Camera not recognized. Please provide a supported camera name or acronym."
    raise CameraResolutionError(message)
