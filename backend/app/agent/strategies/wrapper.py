import json
import logging
import re
from typing import Any, Dict, Optional

from .base import Strategy
from ..providers.base import Provider
from ..exceptions import StrategyError

logger = logging.getLogger(__name__)

FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

class JSONWrapper(Strategy):
    async def __call__(
        self,
        prompt: str,
        provider: Provider,
        **generation_args: Any
    ) -> Dict[str,Any]:
        """
        Wrapper strategy to format the prompt as JSON with the help of LLM
        """
        response = await provider(prompt, **generation_args)
        response = response.strip()
        logger.info(f"provider response: {response}")

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        for fence_match in FENCE_PATTERN.finditer(response):
            fenced = fence_match.group(1).strip()
            try:
                return json.loads(fenced)
            except json.JSONDecodeError:
                continue

        obj_start, obj_end = response.find("{"), response.rfind("}")

        candidates: List[Tuple[int, str]] = []
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            candidates.append((obj_start, response[obj_start:obj_end + 1]))

        for _, candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                candidate2 = candidate.replace("```", "").strip()
                try:
                    return json.loads(candidate2)
                except json.JSONDecodeError:
                    continue

        if candidates:
            _err_preview = response if len(response) <= 2000 else response[:2000] + "... (truncated)"
            logger.error(
                "provider returned non-JSON. failed to parse candidate blocks - response: %s",
                _err_preview,
            )
        raise StrategyError("JSON parsing error: failed to parse candidate JSON blocks")

        logger.error(
            "provider response contained no JSON object braces: %s", response
        )
        raise StrategyError("JSON parsing error: no JSON object detected in provider response")

class MDWrapper(Strategy):
    async def __call__(
        self,
        prompt: str,
        provider: Provider,
        **generation_args: Any
    ) -> str:
        """
        Wrapper strategy to format the prompt as Markdown with the help of LLM
        """
        logger.info(f"prompt given to provider : \n{prompt}")
        response = await provider(prompt, **generation_args)
        logger.info(f"provider response: {response}")
        try:
            response = (
                "```md\n" + response + "```" if "```md" not in response else response
            )
            return response
        except Exception as e:
            logger.error(
                f"provider returned non-md. parsing error: {e} - response: {response}"
            )
            raise StrategyError(f"Markdown parsing error: {e}") from e
    