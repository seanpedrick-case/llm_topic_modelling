"""Synth coverage for application tags and tag-based Resource Groups."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


def test_apply_application_tags_to_app():
    from aws_cdk import App, Environment, Stack
    from aws_cdk import aws_s3 as s3
    from cdk_application_tags import apply_application_tags_to_app

    app = App()
    apply_application_tags_to_app(
        app,
        tag_key="Application",
        application_name="demo-llm-topic-modeller",
        repository_url="https://github.com/example/repo.git",
    )
    stack = Stack(
        app,
        "TagTestStack",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    s3.Bucket(stack, "TaggedBucket")

    template = app.synth().get_stack_by_name("TagTestStack").template
    buckets = [
        r for r in template["Resources"].values() if r["Type"] == "AWS::S3::Bucket"
    ]
    assert buckets
    tags = buckets[0]["Properties"].get("Tags", [])
    tag_map = {t["Key"]: t["Value"] for t in tags}
    assert tag_map.get("Application") == "demo-llm-topic-modeller"
    assert tag_map.get("ManagedBy") == "cdk"
    assert tag_map.get("Repository") == "https://github.com/example/repo.git"


def test_create_application_resource_group_synth():
    from aws_cdk import App, Environment, Stack
    from cdk_application_tags import create_application_resource_group

    app = App()
    stack = Stack(
        app,
        "ResourceGroupTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    create_application_resource_group(
        stack,
        "ApplicationResourceGroup",
        group_name="demo-llm-topic-modeller-resources",
        tag_key="Application",
        application_name="demo-llm-topic-modeller",
    )

    template = app.synth().get_stack_by_name("ResourceGroupTest").template
    groups = [
        r
        for r in template["Resources"].values()
        if r["Type"] == "AWS::ResourceGroups::Group"
    ]
    assert len(groups) == 1
    props = groups[0]["Properties"]
    assert props["Name"] == "demo-llm-topic-modeller-resources"
    assert props["ResourceQuery"]["Type"] == "TAG_FILTERS_1_0"
    tag_filters = props["ResourceQuery"]["Query"]["TagFilters"]
    assert any(
        f.get("Key") == "Application"
        and "demo-llm-topic-modeller" in f.get("Values", [])
        for f in tag_filters
    )
    outputs = template.get("Outputs", {})
    assert any("ApplicationResourceGroup" in name for name in outputs)
