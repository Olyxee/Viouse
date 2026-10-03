from pathlib import Path
import boto3
from .config import settings

def client(): return boto3.client('s3', endpoint_url=settings.s3_endpoint_url or None)

def key(path): return Path(path).resolve().relative_to(settings.data_dir).as_posix()

def publish_private(path):
    if settings.s3_bucket:
        client().upload_file(str(path),settings.s3_bucket,key(path),ExtraArgs={'ServerSideEncryption':'AES256'})
    return key(path)

def delete_project_objects(project_id):
    if not settings.s3_bucket: return
    s3 = client()
    prefix = f'projects/{project_id}/'
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=settings.s3_bucket,Prefix=prefix):
        objects = [{'Key':o['Key']} for o in page.get('Contents',[])]
        if objects:
            result = s3.delete_objects(Bucket=settings.s3_bucket,Delete={'Objects':objects})
            if result.get('Errors'): raise RuntimeError('Unable to remove some private storage objects')
