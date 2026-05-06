"""LLM-based reranker provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

from .base_reranker import BaseReranker


class LLMReranker(BaseReranker):
    """Rank candidates by asking an LLM to return ordered candidate ids."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        super().__init__(config)
        self.prompt_path = str(
            self.config.get("prompt_path", "config/prompts/rerank.txt")
        ).strip()
        self.prompt_template = self._load_prompt_template()
        llm_config = self._resolve_llm_config()
        self.llm = llm or LLMFactory.create({"llm": llm_config})

    def rerank(
        self,
        query: str,
        candidates: list[Mapping[str, Any]],
        trace=None,
    ) -> list[dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)

        id_to_candidate = {str(candidate["id"]): dict(candidate) for candidate in candidates}
        candidate_payload = [
            {"id": str(candidate["id"]), "text": str(candidate["text"])}
            for candidate in candidates
        ]

        prompt = self._build_prompt(query, candidate_payload)
        response = self.llm.chat([{"role": "user", "content": prompt}])
        ranked_ids = self._parse_ranked_ids(response, expected_ids=list(id_to_candidate))
        return [id_to_candidate[item_id] for item_id in ranked_ids]

    def _build_prompt(self, query: str, candidates: list[dict[str, str]]) -> str:
        candidate_json = json.dumps(candidates, ensure_ascii=False, indent=2)
        return (
            self.prompt_template.replace("{query}", query).replace(
                "{candidates}", candidate_json
            )
        )

    def _load_prompt_template(self) -> str:
        prompt_path = Path(self.prompt_path)
        if not prompt_path.exists():
            raise ValueError(f"llm reranker prompt not found: {prompt_path}")
        template = prompt_path.read_text(encoding="utf-8").strip()
        if not template:
            raise ValueError("llm reranker prompt template must not be empty")
        return template

    def _resolve_llm_config(self) -> dict[str, Any]:
        nested = self.config.get("llm")
        if isinstance(nested, Mapping):
            llm_config = dict(nested)
        else:
            llm_config = dict(self.config)

        llm_config.pop("llm", None)
        llm_config.pop("prompt_path", None)
        llm_config.pop("provider", None)

        if "provider" not in llm_config:
            llm_config["provider"] = "openai"
        return llm_config

    def _parse_ranked_ids(self, response: str, expected_ids: list[str]) -> list[str]:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("llm reranker response must be valid JSON") from exc

        ranked_ids: list[str] | None = None
        if isinstance(payload, list):
            ranked_ids = payload
        elif isinstance(payload, dict):
            ids = payload.get("ranked_ids")
            if isinstance(ids, list):
                ranked_ids = ids

        if ranked_ids is None:
            raise ValueError("llm reranker response must contain ranked_ids")
        if any(not isinstance(item, str) or not item.strip() for item in ranked_ids):
            raise ValueError("llm reranker ranked_ids must contain non-empty strings")

        expected_set = set(expected_ids)
        if set(ranked_ids) != expected_set or len(ranked_ids) != len(expected_ids):
            raise ValueError("llm reranker ranked_ids must exactly match candidate ids")
        return ranked_ids


__all__ = ["LLMReranker"]
