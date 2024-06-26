# Quoc Thinh Vo
# Drexel University
# Playground built from a pre-trained model for visualizing ML

import streamlit as st
import esc_config as config
import librosa
import torch
from model.htsat import HTSAT_Swin_Transformer
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import pickle
from db_setup.insert_model_db import *

load_dotenv()
DB_USER= os.getenv('DB_USER')
DB_PASSWORD= os.getenv('DB_PASSWORD')
DB_HOST= os.getenv('DB_HOST')
DATABASE= os.getenv('DATABASE')

st.title("Pretrained Audio Transformer to Classify Air Acoustic Sounds")
st.header("Current Trained Classes:")
st.write("Data from ESC-50 Dataset")
st.image("Capture.PNG")


input_file = st.file_uploader('Upload a .wav file - Recommend > 5 seconds long')
st.write("Verify your input sound")
st.audio(input_file)

button_clicked=st.button("Visualize spectrogram and perform prediction")

class Audio_Classification:
    def __init__(self, model_path, config):
        super().__init__()

        self.device = torch.device('cpu')
        self.sed_model = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config = config,
            depths = config.htsat_depth,
            embed_dim = config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
        )
      
        # ckpt = torch.load(model_path, map_location="cpu")
        @st.cache_resource
        def load_model():
            # return torch.load(model_path, map_location="cpu")
            return model_path

        ckpt = load_model()
        temp_ckpt = {}
        # for key in ckpt["state_dict"]:
        for key in ckpt:
            temp_ckpt[key[10:]] = ckpt[key]
        self.sed_model.load_state_dict(temp_ckpt)
        self.sed_model.to(self.device)
        self.sed_model.eval()
        #print(self.sed_model.eval())

    def predict(self, audiofile):

        if audiofile:

            waveform, sr = librosa.load(audiofile, sr=32000)
 
            f, ax = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(8, 2))

            spec = librosa.feature.melspectrogram(y=waveform, sr=32000, n_fft=2048, hop_length=config.hop_size)
            spec = librosa.power_to_db(spec)


            ax.imshow(spec, origin='lower', interpolation=None, cmap='viridis', aspect=1.1)
            ax.set_title("Spectrogram Visualization", fontsize=11)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            f.tight_layout()

            st.write(f)

            # plt.savefig(f'{audiofile}.png', bbox_inches='tight', dpi=72)

            with torch.no_grad():
                x = torch.from_numpy(waveform).float().to(self.device)
                output_dict = self.sed_model(x[None, :], None, True)
                pred = output_dict['clipwise_output']
                pred_post = pred[0].detach().cpu().numpy()
                pred_label = np.argmax(pred_post)
                pred_prob = np.max(pred_post)
      
            return pred_label, pred_prob, pred_post
        

# model_path="l-epoch=79-acc=0.968.ckpt"
# connection = psycopg2.connect(user=DB_USER,
#                                 password=DB_PASSWORD,
#                                 host=DB_HOST,
#                                 database=DATABASE)

# cursor = connection.cursor()
# select_query = "select * from saved_model"
# cursor.execute(select_query)
# models = cursor.fetchall()
# model_path = pickle.loads(models[0][2])

meta = np.loadtxt('esc50.csv' , delimiter=',', dtype='str', skiprows=1)
# input_file="mixkit-medium-size-angry-dog-bark-54.wav"
class_name = {}
for label in meta:
    category = label[3]
    target = label[2]
    class_name[target]=category

if button_clicked and input_file != None:

    
    s3 = get_credentials()
    model_path = download_obj(s3)

    Audiocls = Audio_Classification(model_path, config)       
    pred_label, pred_prob, pred_post = Audiocls.predict(input_file)
    # st.write('Audiocls predict output: ', pred_label, pred_prob, class_name[str(pred_label)])
    st.header("===*** Prediction ***===")
    st.write("Class name:", class_name[str(pred_label)])
    st.write("Target Class Index:", pred_label)
    st.write("Max Probability:", pred_prob)
    df = pd.DataFrame(pred_post, columns=["Prob"])
    st.dataframe(df.style.format("{: 0.6f}"))

 

    




