# Quoc Thinh Vo
# Drexel University
# Playground built from a pre-trained model for visualizing ML

import psycopg2
import pickle
from dotenv import load_dotenv
import os
import torch
import gzip
import bz2

load_dotenv()
DB_USER= os.getenv('DB_USER')
DB_PASSWORD= os.getenv('DB_PASSWORD')
DB_HOST= os.getenv('DB_HOST')
DATABASE= os.getenv('DATABASE')

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

    return s3

def upload_obj(s3):
    with BytesIO() as f:
        file = 'dummy.pkl'
        joblib.dump(file, f)
        f.seek(0)
        s3.upload_fileobj(Bucket="pretrained-model-qtvo", Key="dummy.pkl", Fileobj=f)

def download_obj(s3):
    print("this is called")
    with BytesIO() as f:
        s3.download_fileobj(Bucket="pretrained-model-qtvo", Key="dummy.pkl", Fileobj=f)
        f.seek(0)
        file = joblib.load(f)

    return file


# data = torch.load("model.ckpt", map_location="cpu")

# with open("dummy.pkl", "wb") as outfile:
#     outfile.write(pickle.dumps(data["state_dict"]))

# ofile = bz2.BZ2File("BinaryData",'wb')
# pickle.dump(data,ofile)
# ofile.close()
# ifile = bz2.BZ2File("BinaryData",'rb')
# # writing into file. This will take long time
# # fp = gzip.open("dummy.data",'wb')
# # pickle.dump(file, fp)
# # fp.close()

# #read the file
# #fp = gzip.open('dummy.data','rb') #This assumes that tfidf.data is already packed with gzip
# with open("dummy.pkl", 'rb') as pickle_file:
    # content = pickle.load(pickle_file)
# print(tfidf)
#

# fp.close()

# fp = gzip.compress(pickle.dumps(file))

# connection = psycopg2.connect(user=DB_USER,
#                                 password=DB_PASSWORD,
#                                 host=DB_HOST,
#                                 database=DATABASE)

# model_insert_sql = "INSERT INTO saved_model VALUES(%s, %s, %s)"
# # insert_tuple = (1, 'HTS-AT', psycopg2.Binary(pickle.loads(fp)) )
# insert_tuple = (1, 'HTS-AT', psycopg2.Binary(ifile) )
# cursor = connection.cursor()
# cursor.execute(model_insert_sql, insert_tuple)

# cursor.close()
# connection.commit()

