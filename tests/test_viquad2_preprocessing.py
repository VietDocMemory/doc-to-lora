from ctx_to_lora.data.definitions import VIQUAD2_ABSTENTION, VIQUAD2_SYSTEM_PROMPT
from ctx_to_lora.data.preprocessing_fn import get_preprocessing_fn


def test_viquad2_answerable_uses_annotated_answer():
    preprocess = get_preprocessing_fn("viquad2", is_eval=False)
    result = preprocess(
        {
            "context": "Hà Nội là thủ đô của Việt Nam.",
            "question": "Thủ đô của Việt Nam là gì?",
            "answers": {"text": ["Hà Nội"], "answer_start": [0]},
            "is_impossible": False,
            "plausible_answers": {"text": ["Huế"]},
        }
    )

    assert result["response"] == "Hà Nội"
    assert result["system_message"] == VIQUAD2_SYSTEM_PROMPT


def test_viquad2_unanswerable_ignores_plausible_answer():
    preprocess = get_preprocessing_fn("viquad2", is_eval=False)
    result = preprocess(
        {
            "context": "Hà Nội là thủ đô của Việt Nam.",
            "question": "Hà Nội có bao nhiêu quận vào năm 1900?",
            "answers": {"text": []},
            "is_impossible": True,
            "plausible_answers": {"text": ["12 quận"]},
        }
    )

    assert result["response"] == VIQUAD2_ABSTENTION
    assert "12 quận" not in result["response"]
