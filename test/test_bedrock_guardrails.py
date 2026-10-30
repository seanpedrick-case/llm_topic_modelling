"""Bedrock Guardrails on Converse (config + call_aws_bedrock)."""

import unittest
from unittest.mock import MagicMock, patch

from tools.llm_funcs import (
    ResponseObject,
    _bedrock_guardrail_config_for_converse,
    call_aws_bedrock,
)


class TestBedrockGuardrailConfig(unittest.TestCase):
    def test_guardrail_config_none_when_disabled(self):
        with patch(
            "tools.config.ENABLE_BEDROCK_GUARDRAILS",
            False,
        ):
            self.assertIsNone(_bedrock_guardrail_config_for_converse())

    def test_guardrail_config_when_enabled(self):
        with (
            patch(
                "tools.config.ENABLE_BEDROCK_GUARDRAILS",
                True,
            ),
            patch(
                "tools.config.BEDROCK_GUARDRAIL_IDENTIFIER",
                "abc123",
            ),
            patch(
                "tools.config.BEDROCK_GUARDRAIL_VERSION",
                "2",
            ),
            patch(
                "tools.config.BEDROCK_GUARDRAIL_TRACE",
                "enabled",
            ),
        ):
            cfg = _bedrock_guardrail_config_for_converse()
            self.assertEqual(
                cfg,
                {
                    "guardrailIdentifier": "abc123",
                    "guardrailVersion": "2",
                    "trace": "enabled",
                },
            )

    def test_guardrail_config_omits_trace_when_disabled(self):
        with (
            patch(
                "tools.config.ENABLE_BEDROCK_GUARDRAILS",
                True,
            ),
            patch(
                "tools.config.BEDROCK_GUARDRAIL_IDENTIFIER",
                "abc123",
            ),
            patch(
                "tools.config.BEDROCK_GUARDRAIL_VERSION",
                "1",
            ),
            patch(
                "tools.config.BEDROCK_GUARDRAIL_TRACE",
                "disabled",
            ),
        ):
            cfg = _bedrock_guardrail_config_for_converse()
            self.assertEqual(
                cfg,
                {
                    "guardrailIdentifier": "abc123",
                    "guardrailVersion": "1",
                },
            )


class TestCallAwsBedrockGuardrails(unittest.TestCase):
    def _mock_converse_response(self, text: str, stop_reason: str = "end_turn"):
        return {
            "output": {"message": {"content": [{"text": text}]}},
            "usage": {"inputTokens": 1, "outputTokens": 2},
            "stopReason": stop_reason,
            "ResponseMetadata": {},
        }

    def test_converse_includes_guardrail_config_when_enabled(self):
        runtime = MagicMock()
        runtime.converse.return_value = self._mock_converse_response("ok")

        with (
            patch(
                "tools.config.ENABLE_BEDROCK_GUARDRAILS",
                True,
            ),
            patch(
                "tools.config.BEDROCK_GUARDRAIL_IDENTIFIER",
                "gr-id",
            ),
            patch(
                "tools.config.BEDROCK_GUARDRAIL_VERSION",
                "1",
            ),
            patch(
                "tools.config.BEDROCK_GUARDRAIL_TRACE",
                "disabled",
            ),
        ):
            resp = call_aws_bedrock(
                "user prompt",
                "system",
                0.5,
                100,
                "anthropic.claude-haiku-4-5-20251001-v1:0",
                runtime,
            )

        self.assertEqual(resp.text, "ok")
        kwargs = runtime.converse.call_args.kwargs
        self.assertIn("guardrailConfig", kwargs)
        self.assertEqual(kwargs["guardrailConfig"]["guardrailIdentifier"], "gr-id")
        self.assertEqual(kwargs["guardrailConfig"]["guardrailVersion"], "1")
        self.assertNotIn("trace", kwargs["guardrailConfig"])

    def test_converse_omits_guardrail_config_when_disabled(self):
        runtime = MagicMock()
        runtime.converse.return_value = self._mock_converse_response("ok")

        with patch(
            "tools.config.ENABLE_BEDROCK_GUARDRAILS",
            False,
        ):
            call_aws_bedrock(
                "user prompt",
                "system",
                0.5,
                100,
                "anthropic.claude-haiku-4-5-20251001-v1:0",
                runtime,
            )

        kwargs = runtime.converse.call_args.kwargs
        self.assertNotIn("guardrailConfig", kwargs)

    def test_guardrail_intervention_sets_response_flags(self):
        runtime = MagicMock()
        runtime.converse.return_value = {
            **self._mock_converse_response(
                "Sorry, blocked.", stop_reason="guardrail_intervened"
            ),
            "trace": {
                "guardrail": {
                    "inputAssessment": {
                        "gr-1": {
                            "contentPolicy": {
                                "filters": [{"type": "INSULTS", "action": "BLOCKED"}]
                            }
                        }
                    }
                }
            },
        }

        with patch(
            "tools.config.ENABLE_BEDROCK_GUARDRAILS",
            False,
        ):
            resp = call_aws_bedrock(
                "user prompt",
                "system",
                0.5,
                100,
                "anthropic.claude-haiku-4-5-20251001-v1:0",
                runtime,
            )

        self.assertTrue(resp.guardrail_intervened)
        self.assertEqual(resp.stop_reason, "guardrail_intervened")
        self.assertEqual(resp.text, "Sorry, blocked.")
        self.assertIn("Content (INSULTS)", resp.guardrail_intervention_categories)

    def test_intervention_categories_when_trace_disabled(self):
        runtime = MagicMock()
        runtime.converse.return_value = self._mock_converse_response(
            "blocked", stop_reason="guardrail_intervened"
        )

        with patch("tools.config.ENABLE_BEDROCK_GUARDRAILS", False):
            resp = call_aws_bedrock(
                "user prompt",
                "system",
                0.5,
                100,
                "anthropic.claude-haiku-4-5-20251001-v1:0",
                runtime,
            )

        self.assertIn("Tracing disabled", resp.guardrail_intervention_categories)

    def test_process_requests_does_not_retry_after_guardrail_intervention(self):
        from tools.llm_funcs import process_requests

        runtime = MagicMock()
        blocked = ResponseObject(
            text="blocked",
            usage_metadata={},
            stop_reason="guardrail_intervened",
        )

        with patch(
            "tools.llm_funcs.call_aws_bedrock",
            return_value=blocked,
        ) as mock_call:
            (
                _responses,
                _history,
                _whole,
                _meta,
                _text,
                guardrail_intervened,
            ) = process_requests(
                ["prompt"],
                "system",
                [],
                [],
                [],
                None,
                None,
                "anthropic.claude-haiku-4-5-20251001-v1:0",
                0.5,
                bedrock_runtime=runtime,
                model_source="AWS Bedrock",
                batch_no=1,
                local_model=None,
                tokenizer=None,
            )

        self.assertEqual(mock_call.call_count, 1)
        self.assertTrue(guardrail_intervened)


if __name__ == "__main__":
    unittest.main()
