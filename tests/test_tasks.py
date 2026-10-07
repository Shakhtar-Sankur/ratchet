from ratchet.tasks import extract_answer, reference_answer, reward


def test_reference_answers():
    assert reference_answer("She has 3 + 4 = <<3+4=7>>7 apples.\n#### 7") == 7
    assert reference_answer("...\n#### 1,250") == 1250
    assert reference_answer("#### -3.5") == -3.5


def test_extracting_the_models_answer():
    assert extract_answer("3 + 4 = 7, so 7 apples.\n#### 7") == 7
    assert extract_answer("First 12, then 15.\n#### $1,200.") == 1200
    assert extract_answer("so the total is \\boxed{42}") == 42
    assert extract_answer("The answer is 18 dollars") == 18
    assert extract_answer("It costs 5 and then 9") == 9          # fallback: last number
    assert extract_answer("#### 3 ... wait, 4 more ... #### 8") == 8  # the last marker wins
    assert extract_answer("no numbers here") is None


def test_reward():
    assert reward("so #### 72", 72) == 1.0
    assert reward("so #### 71", 72) == 0.0
    assert reward("so #### 72.0", 72) == 1.0
    assert reward("nothing", 72) == 0.0
