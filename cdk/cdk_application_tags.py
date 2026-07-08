"""Application tagging helpers and tag-based AWS Resource Groups."""

from typing import Optional

from aws_cdk import CfnOutput, CfnTag, Tags
from aws_cdk import aws_resourcegroups as resourcegroups
from constructs import Construct


def apply_application_tags_to_app(
    app: Construct,
    *,
    tag_key: str,
    application_name: str,
    repository_url: str,
) -> None:
    """
    Tag all CDK-managed resources under ``app`` with a shared application identity.

    Imported / looked-up resources (existing VPC, buckets, etc.) are not retagged
    in AWS; only resources created by this CDK app receive the tags.
    """
    key = (tag_key or "Application").strip() or "Application"
    name = (application_name or "").strip()
    if not name:
        raise ValueError("application_name is required for application tags.")
    Tags.of(app).add(key, name)
    Tags.of(app).add("ManagedBy", "cdk")
    repo = (repository_url or "").strip()
    if repo:
        Tags.of(app).add("Repository", repo)


def create_application_resource_group(
    scope: Construct,
    logical_id: str,
    *,
    group_name: str,
    tag_key: str,
    application_name: str,
    description: Optional[str] = None,
) -> resourcegroups.CfnGroup:
    """
    Create a tag-based Resource Group for inventory/monitoring without AppRegistry.

    Members are all resources in the account/region that carry ``tag_key=application_name``.
    """
    key = (tag_key or "Application").strip() or "Application"
    name = (application_name or "").strip()
    group = (group_name or "").strip()
    if not name:
        raise ValueError("application_name is required for the Resource Group.")
    if not group:
        raise ValueError("group_name is required for the Resource Group.")

    cfn_group = resourcegroups.CfnGroup(
        scope,
        logical_id,
        name=group,
        description=description
        or (
            f"Tag-based group for {name} "
            f"(filter: {key}={name}). Prefer over AppRegistry after Jul 2026."
        ),
        resource_query=resourcegroups.CfnGroup.ResourceQueryProperty(
            type="TAG_FILTERS_1_0",
            query=resourcegroups.CfnGroup.QueryProperty(
                resource_type_filters=["AWS::AllSupported"],
                tag_filters=[
                    resourcegroups.CfnGroup.TagFilterProperty(
                        key=key,
                        values=[name],
                    )
                ],
            ),
        ),
        tags=[
            CfnTag(key=key, value=name),
            CfnTag(key="ManagedBy", value="cdk"),
        ],
    )

    CfnOutput(
        scope,
        f"{logical_id}Name",
        value=cfn_group.name,
        description="AWS Resource Groups name for this application's tagged resources",
    )
    CfnOutput(
        scope,
        f"{logical_id}Arn",
        value=cfn_group.attr_arn,
        description="AWS Resource Groups ARN for this application's tagged resources",
    )
    return cfn_group
