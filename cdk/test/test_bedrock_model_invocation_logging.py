"""Synth coverage for Bedrock model invocation logging (S3 + CloudWatch)."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


def test_add_bedrock_s3_policy_documents_prefix_paths():
    from aws_cdk import App, Environment, Stack
    from aws_cdk import aws_s3 as s3
    from cdk_functions import add_bedrock_model_invocation_s3_bucket_policy

    app = App()
    stack = Stack(
        app,
        "BedrockS3PolicyTest",
        env=Environment(account="063418083240", region="eu-west-2"),
    )
    bucket = s3.Bucket(stack, "LogBucket", bucket_name="test-summarisation-s3-logs")
    add_bedrock_model_invocation_s3_bucket_policy(
        bucket,
        region="eu-west-2",
        account="063418083240",
        key_prefix="bedrock-logs",
    )

    template = app.synth().get_stack_by_name("BedrockS3PolicyTest").template
    policies = [
        r
        for r in template["Resources"].values()
        if r["Type"] == "AWS::S3::BucketPolicy"
    ]
    assert policies
    blob = str(policies)
    assert "bedrock.amazonaws.com" in blob
    assert "BedrockModelInvocationLogs" in blob
    assert "bedrock-logs" in blob
    assert "063418083240" in blob


def test_create_bedrock_model_invocation_logging_synth():
    from aws_cdk import App, Environment, RemovalPolicy, Stack
    from aws_cdk import aws_kms as kms
    from aws_cdk import aws_s3 as s3
    from cdk_functions import create_bedrock_model_invocation_logging

    app = App()
    stack = Stack(
        app,
        "BedrockLoggingTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    key = kms.Key(stack, "SharedKey", removal_policy=RemovalPolicy.DESTROY)
    bucket = s3.Bucket(
        stack,
        "LogBucket",
        encryption=s3.BucketEncryption.KMS,
        encryption_key=key,
        removal_policy=RemovalPolicy.DESTROY,
        auto_delete_objects=True,
    )
    create_bedrock_model_invocation_logging(
        stack,
        "BedrockModelInvocationLogging",
        log_bucket=bucket,
        region="eu-west-2",
        account="123456789012",
        key_prefix="bedrock-logs",
        log_group_name="/aws/bedrock/test-model-invocations",
        log_retention_days=90,
        kms_key=key,
    )

    template = app.synth().get_stack_by_name("BedrockLoggingTest").template
    resources = template["Resources"]
    types = {r["Type"] for r in resources.values()}

    assert "AWS::Logs::LogGroup" in types
    assert "AWS::IAM::Role" in types
    assert "AWS::S3::BucketPolicy" in types
    assert "Custom::AWS" in types or any(
        "Custom::AWS" in r.get("Type", "") for r in resources.values()
    )

    log_groups = [r for r in resources.values() if r["Type"] == "AWS::Logs::LogGroup"]
    assert any(
        r["Properties"].get("LogGroupName") == "/aws/bedrock/test-model-invocations"
        for r in log_groups
    )

    roles = [r for r in resources.values() if r["Type"] == "AWS::IAM::Role"]
    bedrock_roles = [
        r
        for r in roles
        if "bedrock.amazonaws.com"
        in str(r["Properties"].get("AssumeRolePolicyDocument", {}))
    ]
    assert bedrock_roles

    policy_blob = str(
        [r for r in resources.values() if r["Type"] == "AWS::IAM::Policy"]
        + [r for r in resources.values() if r["Type"] == "AWS::KMS::Key"]
    )
    assert "GenerateDataKey" in policy_blob or "kms:GenerateDataKey" in str(resources)
