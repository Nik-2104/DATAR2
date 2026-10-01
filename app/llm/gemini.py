import json
import re
import logging
from typing import Any

from google import genai
from google.genai import types

from app.config import settings
from app.models import (
    DatasetCandidate,
    DatasetPlan,
    DatasetRequirement,
    TransformationPlan,
)

logger = logging.getLogger(__name__)


class RequirementParser:
    """Parse a natural language request into a validated requirement."""

    def __init__(self, client: Any | None = None, model: str | None = None):
        if client is None:
            if not settings.GEMINI_API_KEY:
                raise RuntimeError(
                    "GEMINI_API_KEY is missing. Set it in the environment or .env."
                )
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.client = client
        self.model = model or settings.GEMINI_MODEL

    async def parse(self, raw_request: str) -> DatasetRequirement:
        request = raw_request.strip()
        if not request:
            raise ValueError("Dataset request must not be empty.")

        prompt = f"""Extract only explicitly stated dataset requirements from the user's request.
Use null or empty collections when unspecified. Generate 2-4 concise, useful repository search queries.
Identify task where apparent (for example object detection, classification, segmentation, or speech recognition).
Put geographic and language constraints in filters with clear keys. Do not invent size, classes, or formats.
Return the structured DatasetRequirement matching the schema.

User request:\n{request}"""
        logger.info("Parsing dataset requirement")
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DatasetRequirement,
                ),
            )
        except Exception:
            logger.exception("Gemini requirement parsing failed")
            raise

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, DatasetRequirement):
            data = parsed.model_dump()
        elif parsed is not None:
            data = parsed
        else:
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Gemini returned an empty structured response.")
            try:
                data = json.loads(re.sub(r"^```json\s*|\s*```$", "", text.strip()))
            except json.JSONDecodeError as exc:
                raise RuntimeError("Gemini returned invalid structured output.") from exc

        if not isinstance(data, dict):
            if hasattr(data, "model_dump"):
                data = data.model_dump()
            else:
                raise RuntimeError("Gemini returned an unsupported structured response.")
        data["raw_request"] = raw_request
        result = DatasetRequirement.model_validate(data)
        logger.info("Parsed requirement with %d search queries", len(result.search_queries))
        return result


class DatasetDiscoveryService:
    """Convenience orchestration for parse-then-search."""

    def __init__(self, requirement_parser: RequirementParser | None = None, search_service=None):
        from app.services.dataset_search import DatasetSearchService

        self.requirement_parser = requirement_parser or RequirementParser()
        self.search_service = search_service or DatasetSearchService()

    async def search(self, raw_request: str) -> tuple[DatasetRequirement, list[DatasetCandidate]]:
        requirement = await self.requirement_parser.parse(raw_request)
        candidates = await self.search_service.search(requirement)
        return requirement, candidates


class GeminiPlanner:
    """
    Handles all communication with the Gemini API.

    Gemini is responsible for:
    1. Understanding the user's dataset requirement.
    2. Generating useful search queries.
    3. Selecting the best dataset from candidates.

    Gemini does NOT download or modify datasets.
    """


    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. "
                "Please add it to your configuration."
            )

        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        self.model = settings.GEMINI_MODEL


    def _json_call(self, prompt: str) -> dict:
        """
        Send a prompt to Gemini and return the response as JSON.
        """

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        text = response.text.strip()

        # Sometimes models still return markdown code fences.
        text = re.sub(
            r"^```json\s*|\s*```$",
            "",
            text,
        ).strip()

        try:
            return json.loads(text)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini returned invalid JSON:\n{text}"
            ) from exc


    def parse_requirement(
        self,
        user_request: str,
    ) -> DatasetRequirement:
        """
        Convert the user's natural-language request into
        a structured DatasetRequirement.
        """

        prompt = f"""
You are the requirement-understanding component of DataR².

Your job is to understand exactly what dataset the user wants.

Do NOT search for datasets.
Do NOT download anything.
Do NOT invent requirements.

Convert the user's request into the following JSON structure:

{{
    "raw_request": "string",

    "modality": "image|tabular|audio|video|text|other|null",

    "domain": "string|null",

    "task": "string|null",

    "target_size_bytes": "integer|null",

    "target_samples": "integer|null",

    "formats": ["string"],

    "classes": ["string"],

    "columns": ["string"],

    "filters": {{}},

    "search_queries": ["string"]
}}

Rules:

1. raw_request must contain the original user request.

2. modality describes the type of data requested.

3. domain describes the subject/domain of the dataset.

4. Convert MB and GB into bytes.

   1 MB = 1024 * 1024 bytes
   1 GB = 1024 * 1024 * 1024 bytes

5. target_samples means the number of:
   - rows
   - records
   - images
   - audio files
   - videos
   depending on the dataset type.

6. formats contains explicitly requested formats.

7. classes contains explicitly requested categories/classes.

8. columns contains explicitly requested CSV/tabular columns.

9. filters contains explicit filtering requirements.

10. search_queries should contain several useful search phrases
    that can be used to search dataset repositories.

11. Never invent a requirement that the user did not request.

12. If information is unavailable, use null or an empty list/object.

User request:

{user_request}
"""

        data = self._json_call(prompt)

        # Always preserve the exact request supplied by the user.
        data["raw_request"] = user_request

        return DatasetRequirement.model_validate(data)


    def choose_dataset(
        self,
        requirement: DatasetRequirement,
        candidates: list[DatasetCandidate],
    ):
        """
        Ask Gemini to select the most appropriate dataset
        from datasets discovered by the MCP servers.
        """

        if not candidates:
            return None, "No dataset candidates were found."

        candidate_data = []

        for index, candidate in enumerate(candidates):

            candidate_data.append(
                {
                    "index": index,
                    "source": candidate.source,
                    "dataset_id": candidate.dataset_id,
                    "name": candidate.name,
                    "description": candidate.description,
                    "downloads": candidate.downloads,
                    "likes": candidate.likes,
                    "metadata": candidate.metadata,
                }
            )

        prompt = f"""
You are the dataset-selection component of DataR².

The user has requested a dataset.

Your job is to select the BEST candidate from the datasets
returned by the available MCP servers.

Do NOT download anything.

Do NOT invent information about a dataset.

Only use information contained in the candidate metadata.

Prioritize:

1. Semantic relevance.
2. Required data type/modality.
3. Required classes/categories.
4. Required columns.
5. Domain relevance.
6. Dataset quality.
7. Dataset popularity as a secondary factor.

User requirement:

{requirement.model_dump_json(indent=2)}

Available datasets:

{json.dumps(candidate_data, indent=2)}

Return ONLY this JSON:

{{
    "selected_index": 0,
    "reasoning": "Short explanation of why this dataset is the best match."
}}
"""

        data = self._json_call(prompt)

        try:
            index = int(data["selected_index"])

        except (KeyError, TypeError, ValueError):
            index = 0

        # Safety check.
        if index < 0 or index >= len(candidates):
            index = 0

        reasoning = data.get(
            "reasoning",
            "Selected based on dataset relevance.",
        )

        return candidates[index], reasoning


    # ============================================================
    # Step 3 — Create Transformation Plan
    # ============================================================

    def build_transformation(
        self,
        requirement: DatasetRequirement,
    ) -> TransformationPlan:
        """
        Convert the user's requirements into deterministic
        processing instructions.
        """

        return TransformationPlan(
            target_size_bytes=requirement.target_size_bytes,
            target_samples=requirement.target_samples,
            classes=requirement.classes,
            columns=requirement.columns,
            filters=requirement.filters,
            output_format="zip",
        )


    def build_plan(
        self,
        requirement: DatasetRequirement,
        candidates: list[DatasetCandidate],
    ) -> DatasetPlan:
        """
        Create the complete execution plan.
        """

        selected_dataset, reasoning = self.choose_dataset(
            requirement=requirement,
            candidates=candidates,
        )

        transformation = self.build_transformation(
            requirement
        )

        return DatasetPlan(
            requirement=requirement,
            selected_dataset=selected_dataset,
            transformation=transformation,
            reasoning=reasoning,
        )
