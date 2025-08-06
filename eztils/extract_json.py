def extract_json_from_text(
    text: str,
    *,
    return_raw: bool = False,
    use_repair: bool = True,
    allow_arrays: bool = True,
    strict: bool = True,
    unique_only: bool = True,
):
    """
    Extract and parse JSON objects (and optionally arrays) from free-form text produced by an LLM.

    Requirements:
      - pip install regex
      - (optional) pip install json-repair

    Args:
      text: Arbitrary text that may contain one or more JSON objects/arrays.
      return_raw: If True, return the raw matched JSON strings instead of parsed Python values.
      use_repair: If True and json_repair is available, attempt to repair malformed JSON before parsing.
      allow_arrays: If True, also extract top-level JSON arrays in addition to objects.
      strict: If True, use json.loads (strict JSON). If False and repair is unavailable, try a few light fixes.
      unique_only: If True, de-duplicate by the raw matched substring (order preserved).

    Returns:
      A list of parsed Python objects (dicts/lists) or raw JSON strings (if return_raw=True).

    Raises:
      RuntimeError: If the required 'regex' module is not installed.
    """
    # 1) dependencies
    try:
        import regex as re
    except Exception as e:
        raise RuntimeError(
            "The 'regex' module is required. Install with: pip install regex"
        ) from e

    import json

    # json_repair is optional
    repair = None
    if use_repair:
        try:
            from json_repair import repair_json as _repair_json
            repair = _repair_json
        except Exception:
            repair = None  # continue without repair

    # 2) recursive patterns that respect nesting + quoted strings (PCRE-style, supported by 'regex')
    #    We match either:
    #      - an object: { ...nested... }
    #      - optionally, an array:  [ ...nested... ]
    #
    #    Each pattern:
    #      - allows normal text that is not brackets/quotes,
    #      - allows quoted strings with escapes,
    #      - and recurses on itself with (?R) for nested structures.
    obj_pat = r"""
        \{
            (?:                               # zero or more of:
                [^{}"\\]+                     #   non-brace, non-quote, non-backslash chars
              | \\"                           #   escaped quote in a string (covered below too)
              | "(?:\\.|[^"\\])*"             #   a JSON string (handles escapes)
              | (?R)                          #   recurse to match nested {...}
            )*
        \}
    """
    arr_pat = r"""
        \[
            (?:                               # zero or more of:
                [^\[\]"\\]+                   #   non-bracket, non-quote, non-backslash chars
              | \\"                           
              | "(?:\\.|[^"\\])*"             #   a JSON string
              | (?R)                          #   recurse to match nested [...]
            )*
        \]
    """

    # Build one combined pattern that finds either an object or (optionally) an array.
    if allow_arrays:
        combined = rf"(?:{obj_pat}|{arr_pat})"
    else:
        combined = rf"(?:{obj_pat})"

    pattern = re.compile(combined, re.VERBOSE | re.DOTALL)

    # 3) find all candidates
    matches = list(pattern.finditer(text))
    raws = [m.group(0) for m in matches]

    # 4) optionally de-duplicate while preserving order
    if unique_only:
        seen = set()
        unique_raws = []
        for s in raws:
            if s not in seen:
                seen.add(s)
                unique_raws.append(s)
        raws = unique_raws

    if return_raw:
        return raws

    # 5) parse each candidate; optionally repair before strict parsing
    results = []
    for s in raws:
        candidate = s
        parsed = None
        tried_repair = False

        # Try strict parse first (fast path)
        if strict:
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = None

        # If strict failed or strict=False, try repair (if available)
        if parsed is None and repair is not None:
            try:
                tried_repair = True
                fixed = repair(candidate)
                parsed = json.loads(fixed)
            except Exception:
                parsed = None

        # Light, conservative fixes only if strict=False and repair unavailable or failed
        if parsed is None and not strict and (repair is None or tried_repair):
            try:
                # Minimal heuristics: strip trailing commas before } or ]
                import re as pyre
                fixed = pyre.sub(r",\s*([}\]])", r"\1", candidate)
                # Remove BOM if present
                fixed = fixed.lstrip("\ufeff")
                parsed = json.loads(fixed)
            except Exception:
                parsed = None

        if parsed is not None:
            results.append(parsed)

    return results
