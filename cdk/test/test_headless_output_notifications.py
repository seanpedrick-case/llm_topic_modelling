"""Headless S3 output notification resources and installer validation."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

from cdk_functions import sanitize_headless_metric_filter_id


def test_sanitize_headless_metric_filter_id():
    assert sanitize_headless_metric_filter_id("Prod-Summarisation-s3-output-put") == (
        "Prod-Summarisation-s3-output-put"
    )
    assert sanitize_headless_metric_filter_id("bad id!!") == "bad-id"


def test_validate_notify_email():
    from cdk_install import validate_notify_email

    assert validate_notify_email("user@example.com") is None
    assert validate_notify_email("") is not None
    assert validate_notify_email("not-an-email") is not None


def test_headless_output_notifications_synth():
    from aws_cdk import App, Environment, Stack
    from aws_cdk import aws_s3 as s3
    from aws_cdk.assertions import Match, Template
    from cdk_functions import create_headless_output_notifications

    app = App()
    stack = Stack(
        app,
        "HeadlessNotifyTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    bucket = s3.Bucket(stack, "OutputBucket", bucket_name="test-output-bucket")

    create_headless_output_notifications(
        stack,
        "Notify",
        output_bucket=bucket,
        output_prefix="output/",
        notify_email="analyst@example.com",
        iam_user_name="test-s3-output-reader",
        metric_filter_id="test-s3-output-put",
        sns_topic_name="test-llm-topic-s3-save-sns",
        alarm_name="test-cloudwatch-alarm-new-output-s3",
    )

    template = Template.from_stack(stack)
    template.resource_count_is("AWS::SNS::Topic", 1)
    template.resource_count_is("AWS::CloudWatch::Alarm", 1)
    template.resource_count_is("AWS::IAM::User", 1)
    template.has_resource_properties(
        "AWS::S3::BucketPolicy",
        {
            "PolicyDocument": {
                "Statement": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Effect": "Allow",
                                "Action": Match.array_with(["s3:GetObject"]),
                            }
                        )
                    ]
                )
            }
        },
    )
    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "MetricName": "PutRequests",
            "Namespace": "AWS/S3",
            "ComparisonOperator": "GreaterThanThreshold",
        },
    )
