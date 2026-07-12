from src.preprocessor import normalize_text


def test_rejoins_hyphenated_line_wraps():
    raw = "This clause covers termina-\ntion of the agreement."
    assert "termina-\ntion" not in normalize_text(raw)
    assert "termination of the agreement" in normalize_text(raw)


def test_strips_standalone_page_numbers():
    raw = "Section 1. Purpose.\n\n4\n\nSection 2. Term."
    out = normalize_text(raw)
    assert "\n4\n" not in out


def test_collapses_excess_whitespace():
    raw = "Party A    shall    notify Party B.\n\n\n\n\nParty B agrees."
    out = normalize_text(raw)
    assert "    " not in out
    assert "\n\n\n" not in out


def test_normalizes_smart_quotes_and_dashes():
    raw = "the parties\u2019 obligations \u2013 including confidentiality \u2014 survive"
    out = normalize_text(raw)
    assert "\u2019" not in out
    assert "\u2013" not in out
    assert "\u2014" not in out


def test_idempotent_on_clean_text():
    clean = "This is already normalized text."
    assert normalize_text(clean) == clean
