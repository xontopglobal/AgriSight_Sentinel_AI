from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError, BotoCoreError

load_dotenv(".env", override=True)

try:
    sts = boto3.client("sts")
    identity = sts.get_caller_identity()
    print("AWS authentication successful!")
    print("Account:", identity["Account"])
    print("ARN:", identity["Arn"])
except ClientError as e:
    print("AWS service error:", e.response["Error"]["Code"])
    print("Message:", e.response["Error"]["Message"])
except BotoCoreError as e:
    print("AWS connection/configuration error:", str(e))
