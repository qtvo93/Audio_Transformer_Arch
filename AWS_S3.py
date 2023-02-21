import boto3 
from io import BytesIO 
import joblib

from dotenv import load_dotenv
import os

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION_NAME = os.getenv('REGION_NAME')

def get_credentials():

    session = boto3.Session(region_name=REGION_NAME , aws_access_key_id=AWS_ACCESS_KEY_ID ,
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    s3 = session.client("s3")
    print(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
    return s3
# s3.list_buckets()

# s3.download_file('pretrained-model-qtvo', 'l-epoch=79-acc=0.968.ckpt', "l-epoch=79-acc=0.968.ckpt")


def upload_obj(s3):
    with BytesIO() as f:
        file = 'l-epoch=79-acc=0.968.ckpt'
        joblib.dump(file, f)
        f.seek(0)
        s3.upload_fileobj(Bucket="pretrained-model-qtvo", Key="cli-l-epoch=79-acc=0.968.ckpt", Fileobj=f)

def download_obj(s3):
    print("this is called")
    with BytesIO() as f:
        s3.download_fileobj(Bucket="pretrained-model-qtvo", Key="cli-l-epoch=79-acc=0.968.ckpt", Fileobj=f)
        f.seek(0)
        file = joblib.load(f)

    return file
    
# with open("./trained_model/l-epoch=79-acc=0.968.ckpt", "rb") as f:
#     s3.upload_fileobj(f, "pretrained-model-qtvo", "l-epoch=79-acc=0.968.ckpt")
# def write_joblib(file, path):
#     ''' 
#        Function to write a joblib file to an s3 bucket or local directory.
#        Arguments:
#        * file: The file that you want to save 
#        * path: an s3 bucket or local directory path. 
#     '''

#     # Path is an s3 bucket
#     if path[:5] == 's3://':
#         s3_bucket, s3_key = path.split('/')[2], path.split('/')[3:]
#         s3_key = '/'.join(s3_key)

#         print(s3_bucket, s3_key)
#         with BytesIO() as f:
#             joblib.dump(file, f)
#             f.seek(0)
#             boto3.client("s3").upload_fileobj(Bucket=s3_bucket, Key=s3_key, Fileobj=f)
    
#     # Path is a local directory 
#     else:
#         with open(path, 'wb') as f:
#             joblib.dump(file, f)


# def read_joblib(path):
#     ''' 
#        Function to load a joblib file from an s3 bucket or local directory.
#        Arguments:
#        * path: an s3 bucket or local directory path where the file is stored
#        Outputs:
#        * file: Joblib file loaded
#     '''

#     # Path is an s3 bucket
#     if path[:5] == 's3://':
#         s3_bucket, s3_key = path.split('/')[2], path.split('/')[3:]
#         s3_key = '/'.join(s3_key)
#         with BytesIO() as f:
#             boto3.client("s3").download_fileobj(Bucket=s3_bucket, Key=s3_key, Fileobj=f)
#             f.seek(0)
#             file = joblib.load(f)
    
#     # Path is a local directory 
#     else:
#         with open(path, 'rb') as f:
#             file = joblib.load(f)
    
#     return file


# write_joblib('./trained_model/l-epoch=79-acc=0.968.ckpt', 's3://pretrained-model-qtvo/mdl_dict.joblib')
