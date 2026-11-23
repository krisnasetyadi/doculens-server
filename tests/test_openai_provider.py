"""OpenAI GPT as an alternative to Gemini (MS-1192): the config default, loading
the model, the model list the chat dropdown reads, and Gap Check following the
default provider. No network, no database."""

import asyncio
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "conversation-local-test-secret-not-for-production")

from config import Config, LLMProvider, config
from router.llm_models import get_available_models
from services.rag.processor import processor
from tests.fakes import fake_llm


class DefaultProviderTests(unittest.TestCase):
    def test_only_openai_changes_the_default(self):
        for value, provider in (("gemini", LLMProvider.GEMINI), ("huggingface", LLMProvider.GEMINI),
                                ("openai", LLMProvider.OPENAI)):
            with patch.dict(os.environ, {"LLM_PROVIDER": value}):
                self.assertEqual(Config().default_llm_provider, provider, value)

    def test_default_model_follows_the_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_MODEL": "gpt-test"}):
            self.assertEqual(Config().default_llm_model, "gpt-test")
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-test"}):
            self.assertEqual(Config().default_llm_model, "gemini-test")


class LoadOpenAITests(unittest.TestCase):
    def test_get_llm_builds_a_metered_non_reasoning_chat_model(self):
        with patch.object(config, "openai_api_key", "sk-test"), patch.object(processor, "_llm_cache", {}):
            llm, model_id = processor.get_llm("openai")
        self.assertEqual(model_id, f"openai/{config.openai_model}")
        self.assertEqual(llm.model_name, config.openai_model)
        # Without stream_usage a streamed answer is metered as 0 tokens.
        self.assertTrue(llm.stream_usage)
        self.assertEqual(llm.reasoning_effort, "none")

    def test_requested_model_is_used(self):
        with patch.object(config, "openai_api_key", "sk-test"), patch.object(processor, "_llm_cache", {}):
            _, model_id = processor.get_llm("openai", "gpt-6.1-terra")
        self.assertEqual(model_id, "openai/gpt-6.1-terra")

    def test_missing_key_is_a_clear_error(self):
        with patch.object(config, "openai_api_key", None):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                processor._load_openai_llm("gpt-test")

    def test_a_requested_model_that_cannot_load_is_an_error_not_a_switch_to_gemini(self):
        with patch.object(config, "openai_api_key", None), patch.object(processor, "_llm_cache", {}), \
                patch.object(processor, "_load_gemini_llm") as gemini:
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                processor.get_llm("openai", "gpt-5.4-mini")
        gemini.assert_not_called()

    def test_an_unknown_provider_is_an_error_not_the_default(self):
        with patch.object(processor, "_llm_cache", {}), patch.object(processor, "_load_gemini_llm") as gemini:
            with self.assertRaises(ValueError):
                processor.get_llm("openia")
        gemini.assert_not_called()


class AvailableModelsTests(unittest.TestCase):
    def test_openai_is_listed_only_with_a_key(self):
        with patch.object(config, "openai_api_key", None):
            self.assertNotIn("openai", asyncio.run(get_available_models())["available_models"])
        with patch.object(config, "openai_api_key", "sk-test"):
            listed = asyncio.run(get_available_models())["available_models"]
        self.assertEqual(listed["openai"][0], "gpt-5.4-mini")
        self.assertIn("gemini", listed)


class GapCheckProviderTests(unittest.TestCase):
    def test_gap_check_uses_the_default_provider(self):
        with patch.object(processor, "_get_reference_text", return_value="DOC"), \
                patch.object(processor, "get_llm", return_value=(fake_llm('["A.1 x"]'), "m/x")) as get_llm:
            processor.extract_framework_items(["c1"], "ISO", "en")
        get_llm.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
