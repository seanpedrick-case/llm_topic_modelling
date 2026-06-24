"""Tests for scheduled DynamoDB usage log export Lambda and installer schedule helpers."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


def test_build_dynamo_export_cron_expression():
    import cdk_install as inst

    assert inst.build_dynamo_export_cron_expression(6, 0, "daily") == (
        "cron(0 6 ? * * *)"
    )
    assert inst.build_dynamo_export_cron_expression(7, 30, "weekdays") == (
        "cron(30 7 ? * MON-FRI *)"
    )
    assert inst.build_dynamo_export_cron_expression(0, 15, "weekends") == (
        "cron(15 0 ? * SAT-SUN *)"
    )


def test_validate_schedule_time_hhmm():
    import cdk_install as inst

    assert inst.validate_schedule_time_hhmm("06:00") is None
    assert inst.validate_schedule_time_hhmm("25:00") is not None
    assert inst.validate_schedule_time_hhmm("bad") is not None


def test_build_env_values_includes_dynamo_export_schedule():
    import cdk_install as inst

    answers = inst.InstallAnswers(
        profile="demo",
        aws_account_id="123456789012",
        aws_region="eu-west-2",
        cdk_prefix="Test-Summarisation-",
        cognito_domain_prefix="test-summarisation",
        vpc_mode="existing",
        vpc_name="test-vpc",
        enable_dynamo_usage_log_export=True,
        dynamo_export_schedule_time="07:30",
        dynamo_export_schedule_days="weekdays",
        dynamo_export_s3_key="reports/usage.csv",
    )
    values = inst.build_env_values(answers)
    assert values["ENABLE_DYNAMODB_USAGE_LOG_EXPORT"] == "True"
    assert values["DYNAMODB_USAGE_LOG_EXPORT_SCHEDULE"] == ("cron(30 7 ? * MON-FRI *)")
    assert values["DYNAMODB_USAGE_LOG_EXPORT_S3_KEY"] == "reports/usage.csv"


def test_validate_env_values_rejects_dynamo_export_without_dynamodb_logging():
    import cdk_install as inst

    values = inst.build_env_values(
        inst.InstallAnswers(
            profile="demo",
            aws_account_id="123456789012",
            aws_region="eu-west-2",
            cdk_prefix="Test-Summarisation-",
            cognito_domain_prefix="test-summarisation",
            vpc_mode="existing",
            vpc_name="test-vpc",
            enable_dynamo_usage_log_export=True,
        )
    )
    values["SAVE_LOGS_TO_DYNAMODB"] = "False"
    errors = inst.validate_env_values(values)
    assert any("ENABLE_DYNAMODB_USAGE_LOG_EXPORT" in e for e in errors)


def test_dynamo_usage_log_export_lambda_synth():
    from aws_cdk import App, Environment, Stack
    from aws_cdk import aws_dynamodb as dynamodb
    from aws_cdk import aws_s3 as s3
    from aws_cdk.assertions import Match, Template
    from cdk_functions import create_dynamo_usage_log_export_lambda

    app = App()
    stack = Stack(
        app,
        "DynamoExportTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    table = dynamodb.Table(
        stack,
        "UsageTable",
        partition_key=dynamodb.Attribute(name="id", type=dynamodb.AttributeType.STRING),
    )
    bucket = s3.Bucket(stack, "OutputBucket")

    create_dynamo_usage_log_export_lambda(
        stack,
        "DynamoUsageLogExport",
        function_name="test-dynamo-usage-export",
        lambda_asset_path=str(CDK_DIR / "lambda_dynamo_logs_export"),
        dynamodb_table=table,
        output_bucket=bucket,
        s3_output_key="reports/dynamodb-usage/dynamodb_logs_export.csv",
        schedule_expression="cron(0 6 ? * MON-FRI *)",
        dynamodb_table_name="test-usage-logs",
        date_attribute="timestamp",
    )

    template = Template.from_stack(stack)
    template.resource_count_is("AWS::Lambda::Function", 1)
    template.resource_count_is("AWS::Events::Rule", 1)
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "Handler": "lambda_function.lambda_handler",
            "Environment": {
                "Variables": Match.object_like(
                    {
                        "DYNAMODB_TABLE_NAME": "test-usage-logs",
                        "S3_OUTPUT_KEY": (
                            "reports/dynamodb-usage/dynamodb_logs_export.csv"
                        ),
                    }
                )
            },
        },
    )
    template.has_resource_properties(
        "AWS::Events::Rule",
        {"ScheduleExpression": "cron(0 6 ? * MON-FRI *)"},
    )
