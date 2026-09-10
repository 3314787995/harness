from qwen3vl_agent.models.qwen3vl import Qwen3VLModel


def test_build_messages_injects_media_once() -> None:
    model = Qwen3VLModel("fake")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    prepared = model._build_messages(messages, videos=["v.mp4"], images=["i.jpg"])

    user_content = prepared[1]["content"]
    assert [part["type"] for part in user_content] == ["video", "image", "text"]
    assert sum(
        part.get("type") == "video"
        for message in prepared
        for part in message["content"]
    ) == 1


def test_existing_multimodal_content_is_not_duplicated() -> None:
    model = Qwen3VLModel("fake")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "already.jpg"},
                {"type": "text", "text": "question"},
            ],
        }
    ]
    prepared = model._build_messages(messages, videos=["ignored.mp4"], images=None)
    assert prepared == messages


def test_text_parts_receive_external_media() -> None:
    model = Qwen3VLModel("fake")
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "question"}],
        }
    ]
    prepared = model._build_messages(messages, videos=["video.mp4"], images=None)
    assert [part["type"] for part in prepared[0]["content"]] == ["video", "text"]
