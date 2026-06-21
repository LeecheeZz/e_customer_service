from e_customer_service.data import apply_template


def test_apply_template_minimal(tokenizer=None):
    # This is a smoke test for the templating function. It does not run a
    # tokenizer by default; a real integration test should load a tokenizer.
    sample = {"messages": [{"role": "user", "content": "hello"}]}

    class DummyTok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
            return "USER: hello\nASSISTANT: Hi"

    res = apply_template(DummyTok(), sample)
    assert "text" in res
    assert isinstance(res["text"], str)
