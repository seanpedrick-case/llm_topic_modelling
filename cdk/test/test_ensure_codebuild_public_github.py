"""Tests for runtime CodeBuild public GitHub source fixup."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

from cdk_functions import ensure_codebuild_public_github_source


def test_ensure_codebuild_updates_oauth_project(monkeypatch):
    client = MagicMock()
    client.batch_get_projects.return_value = {
        "projects": [
            {
                "name": "MyProject",
                "source": {
                    "type": "GITHUB",
                    "location": "https://github.com/old/repo.git",
                    "auth": {"type": "CODECONNECTIONS", "resource": "arn:connection"},
                    "reportBuildStatus": True,
                    "buildspec": "version: 0.2\nphases: {}\n",
                },
                "sourceVersion": "main",
                "triggers": {"webhook": True},
            }
        ]
    }
    monkeypatch.setattr("cdk_functions.boto3.client", lambda *args, **kwargs: client)

    updated = ensure_codebuild_public_github_source(
        "MyProject",
        "seanpedrick-case",
        "llm_topic_modeller",
        "main",
        aws_region="eu-west-2",
    )

    assert updated is True
    client.update_project.assert_called_once()
    kwargs = client.update_project.call_args.kwargs
    assert kwargs["name"] == "MyProject"
    assert kwargs["source"]["type"] == "GITHUB"
    assert (
        kwargs["source"]["location"]
        == "https://github.com/seanpedrick-case/llm_topic_modeller.git"
    )
    assert "auth" not in kwargs["source"]
    assert kwargs["sourceVersion"] == "main"
    assert kwargs["triggers"] == {"webhook": False}


def test_ensure_codebuild_skips_when_already_public(monkeypatch):
    client = MagicMock()
    client.batch_get_projects.return_value = {
        "projects": [
            {
                "name": "MyProject",
                "source": {
                    "type": "GITHUB",
                    "location": "https://github.com/seanpedrick-case/llm_topic_modeller.git",
                    "reportBuildStatus": False,
                },
                "sourceVersion": "main",
            }
        ]
    }
    monkeypatch.setattr("cdk_functions.boto3.client", lambda *args, **kwargs: client)

    updated = ensure_codebuild_public_github_source(
        "MyProject",
        "seanpedrick-case",
        "llm_topic_modeller",
        "main",
        aws_region="eu-west-2",
    )

    assert updated is False
    client.update_project.assert_not_called()
