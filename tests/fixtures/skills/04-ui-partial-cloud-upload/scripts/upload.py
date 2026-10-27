import sys

import boto3


def upload(path: str, dest: str) -> None:
    if dest.startswith("s3://"):
        bucket, _, key = dest[len("s3://"):].partition("/")
        boto3.client("s3").upload_file(path, bucket, key)
    else:
        raise SystemExit(f"unsupported destination: {dest}")


if __name__ == "__main__":
    upload(sys.argv[1], sys.argv[2])
