import asyncio
from types import SimpleNamespace

from app.llm.gemini import RequirementParser
from app.models import DatasetRequirement


class FakeModels:
    async def generate_content(self, **kwargs):
        assert kwargs["config"].response_mime_type == "application/json"
        return SimpleNamespace(
            parsed=DatasetRequirement(
                raw_request="ignored",
                modality="image",
                domain="road traffic",
                task="object detection",
                target_samples=10000,
                formats=["YOLO"],
                classes=["vehicle"],
                search_queries=["road vehicle detection dataset", "traffic YOLO images"],
            )
        )


class FakeClient:
    def __init__(self):
        self.aio = SimpleNamespace(models=FakeModels())


def test_parse_structured_requirement():
    raw = "I need road vehicle images in YOLO format, at least 10,000 images."
    requirement = asyncio.run(RequirementParser(client=FakeClient()).parse(raw))

    assert requirement.raw_request == raw
    assert requirement.modality == "image"
    assert requirement.task == "object detection"
    assert requirement.target_samples >= 10000
    assert "YOLO" in requirement.formats
    assert requirement.search_queries
