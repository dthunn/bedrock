import json
import boto3
import botocore
from datetime import datetime
import base64

def lambda_handler(event, context):
    event = json.loads(event['body'])
    message = event['message']

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
        config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3})
    )
    s3 = boto3.client('s3')

    payload = {
        "prompt": message,
        "mode": "text-to-image",
        "aspect_ratio": "1:1",
        "output_format": "png",  # must be "png" or "jpeg"
        "seed": 0
    }

    response = bedrock.invoke_model(
        body=json.dumps(payload),
        modelId='stability.sd3-5-large-v1:0',
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response.get("body").read())
    print("Full response:", json.dumps(response_body))

    # The correct key is "images" (list of base64 strings)
    if "images" not in response_body:
        return {
            'statusCode': 500,
            'body': json.dumps("‘images’ key missing in Bedrock response: " + json.dumps(response_body))
        }

    b64_img = response_body["images"][0]
    image_content = base64.b64decode(b64_img)

    bucket = 'dthunn-bedrock-course-bucket'
    current_time = datetime.now().strftime('%H%M%S')
    key = f"output-images/{current_time}.{payload['output_format']}"

    s3.put_object(Bucket=bucket, Key=key, Body=image_content, ContentType=f"image/{payload['output_format']}")

    return {
        'statusCode': 200,
        'body': json.dumps(f"Image saved to s3://{bucket}/{key}")
    }