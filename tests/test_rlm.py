"""Targeted tests for RLMEnv fixes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset
from prime_sandboxes import CommandTimeoutError

from verifiers.envs.experimental.rlm_env import RLMEnv


@pytest.fixture
def mock_sandbox_client():
    """Create a mock AsyncSandboxClient."""
    client = MagicMock()
    client.create = AsyncMock(return_value=MagicMock(id="sandbox_123"))
    client.delete = AsyncMock()
    client.bulk_delete = AsyncMock()
    client.wait_for_creation = AsyncMock()
    client.execute_command = AsyncMock(return_value=MagicMock(stdout="", stderr=""))
    return client


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for RLMEnv."""
    return Dataset.from_dict(
        {
            "question": ["What is 2+2?"],
            "answer": ["4"],
            "info": [{}],
        }
    )


@pytest.fixture
def rlm_env(mock_sandbox_client, mock_dataset):
    """Create an RLMEnv instance with mocked dependencies."""
    with (
        patch("verifiers.envs.sandbox_env.AsyncSandboxClient") as mock_client_cls,
        patch("verifiers.envs.sandbox_env.CreateSandboxRequest"),
    ):
        mock_client_cls.return_value = mock_sandbox_client
        env = RLMEnv(
            dataset=mock_dataset,
            max_iterations=10,
            max_output_length=1000,
        )
        env.sandbox_client = mock_sandbox_client
        yield env
        env.active_sandboxes.clear()


@pytest.mark.asyncio
async def test_execute_code_timeout_restarts_sandbox(rlm_env):
    rlm_env.abort_on_code_timeout = False
    rlm_env.code_execution_timeout = 1
    rlm_env.sandbox_client.execute_command = AsyncMock(
        side_effect=CommandTimeoutError("sandbox_123", "command", 1)
    )
    rlm_env._recreate_sandbox = AsyncMock(side_effect=lambda state: state)
    rlm_env._prepare_sandbox_and_start_worker = AsyncMock()

    state = {
        "sandbox_id": "sandbox_123",
        "rlm_context": {"input_data": None, "input_data_metadata": {}},
    }
    result = await rlm_env._execute_code("sandbox_123", "print(1)", state)

    assert result["status"] == "error"
    assert "sandbox was restarted" in result["result"].lower()
    rlm_env._recreate_sandbox.assert_awaited_once()
    rlm_env._prepare_sandbox_and_start_worker.assert_awaited_once()
    assert state["_exec_seq"] == 0


def test_sub_llm_timeouts_clamped_to_code_timeout(mock_sandbox_client, mock_dataset):
    with (
        patch("verifiers.envs.sandbox_env.AsyncSandboxClient") as mock_client_cls,
        patch("verifiers.envs.sandbox_env.CreateSandboxRequest"),
    ):
        mock_client_cls.return_value = mock_sandbox_client
        env = RLMEnv(dataset=mock_dataset, code_execution_timeout=5)

    assert env.sub_llm_api_timeout == 4
    assert env.sub_llm_timeout == 4
