import asyncio
import json

from app.models import DatasetRequirement
from app.services.dataset_search import DatasetSearchService


async def main():

    print("=" * 60)
    print("DataR² — COMBINED DATASET SEARCH")
    print("=" * 60)

    requirement = DatasetRequirement(
        raw_request="I need a road traffic image dataset",
        modality="image",
        domain="road traffic",
        target_size_bytes=None,
        target_samples=None,
        formats=[],
        classes=[],
        columns=[],
        filters={},
        search_queries=[
            "road traffic images"
        ],
    )

    service = DatasetSearchService()

    candidates = await service.search(
        requirement
    )

    print()
    print("=" * 60)
    print(f"FOUND {len(candidates)} DATASET CANDIDATES")
    print("=" * 60)

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        print()
        print(f"[{index}]")
        print(f"Source      : {candidate.source}")
        print(f"Dataset ID  : {candidate.dataset_id}")
        print(f"Name        : {candidate.name}")
        print(f"Description : {candidate.description[:200]}")
        print(f"Downloads   : {candidate.downloads}")
        print(f"Likes       : {candidate.likes}")

    print()
    print("=" * 60)
    print("RAW CANDIDATE DATA")
    print("=" * 60)

    print(
        json.dumps(
            [
                candidate.model_dump()
                for candidate in candidates
            ],
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())