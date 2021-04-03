import numpy as np
import os
from model import CNN , LSTMModel
import gc
import scipy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
matplotlib.rcParams['toolbar'] = 'None'
# plt.ion()
# from matplotlib.animation import FuncAnimation

from wordcloud import WordCloud

import librosa
import librosa.display
from librosa.effects import split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as transforms
import torch.optim as optim
from PIL import Image

# from inaSpeechSegmenter import Segmenter

from collections import Counter
import argparse

class RagaDetector():
    def __init__(self , model_path , lstm_model_path):
        self.model_path = model_path
        self.lstm_model_path = lstm_model_path
        self.transforms = transforms.Compose([transforms.Resize(size=(324,216)),
                               transforms.ToTensor(),
                               ]) 
        self.raga_to_idx = {"Abhogi":0, "Ahir Bhairav":1, "Alhaiya Bilawal":2, "Bageshri":3 ,"Bairagi":4 , "Basant":5 , "Bhairav":6,
                            "Bhoopali":7 , "Bihag":8,  "Bilaskhani Todi":9 , "Darbari":10, "Desh":11 , "Gaud Malhar":12, 
                            "Hansadhwani":13 , "Jog":14 , "Kedar":15 , "Khamaj":16 , "Lalit":17 , "Madhukauns":18 ,
                            "Madhuvanti":19 , "Malkauns":20 , "Maru Bihag":21 , "Marwa":22 , "Miyan Ki Malhar":23 ,
                            "Puriya Dhanashree":24 , "Rageshri":25 , "Shree":26 , "Shuddh Sarang":27 , "Todi":28 , "Yaman":29 }
        
        self.idx_to_raga = {0:"Abhogi", 1:"Ahir Bhairav", 2:"Alhaiya Bilawal", 3:"Bageshri" , 4:"Bairagi" , 5:"Basant", 6:"Bhairav",
                            7:"Bhoopali" , 8:"Bihag",  9:"Bilaskhani Todi" , 10:"Darbari", 11:"Desh" , 12:"Gaud Malhar", 
                            13:"Hansadhwani" , 14:"Jog" , 15:"Kedar" , 16:"Khamaj" , 17:"Lalit" , 18:"Madhukauns" ,
                            19:"Madhuvanti" , 20:"Malkauns" , 21:"Maru Bihag" , 22:"Marwa" , 23:"Miyan Ki Malhar" ,
                            24:"Puriya Dhanashree" , 25:"Rageshri" , 26:"Shree" , 27:"Shuddh Sarang" , 28:"Todi" , 29:"Yaman"}
        
        self.seq_length = 5

        # Dashboard figure and subplots declaration
        self.fig_result = plt.figure(figsize=(20,8))
        self.ax_wordcloud = self.fig_result.add_subplot(236)
        self.ax_cnn = self.fig_result.add_subplot(233)
        self.ax_lstm = self.fig_result.add_subplot(231)
        self.ax_raga = self.fig_result.add_subplot(235)
        self.ax_spec = self.fig_result.add_subplot(234)
        self.ax_text = self.fig_result.add_subplot(232)

        # To store history of predictions
        self.cnn_output_history = []
        self.lstm_output_history = []

    def show_final_graph(self):
        ragas = {}

        for raga in self.idx_to_raga.values():
            if raga not in self.result_history["cnn_agg"].keys():
                ragas[raga] = 0
            else:
                ragas[raga] = self.result_history["cnn_agg"][raga]

        plt.bar(ragas.keys() , ragas.values())
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.21)
        plt.title("CNN prediction")
        plt.show()

    def show_final_graph_lstm(self):
        ragas = {}

        for raga in self.idx_to_raga.values():
            if raga not in self.result_history["lstm_agg"].keys():
                ragas[raga] = 0
            else:
                ragas[raga] = self.result_history["lstm_agg"][raga]

        plt.bar(ragas.keys() , ragas.values())
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.21)
        plt.title("LSTM prediction")
        plt.show()


    def show_raga_graph(self , idx):
        arr = []
        lstm_arr = [0 , 0 , 0 , 0 , 0]
        for output in self.cnn_output_history:
            arr.append(output[idx])

        if len(self.lstm_output_history) > 0:
            for output in self.lstm_output_history:
                lstm_arr.append(output[idx])


        if hasattr(self , "raga_plot"):
            # self.ax_raga.set_ydata(arr , label="CNN")
            # self.ax_raga.set_ydata(lstm_arr , label="LSTM")
            self.ax_raga.clear()
            self.ax_raga.plot(arr , label="CNN" , c="b")
            self.ax_raga.plot(lstm_arr , label="LSTM" , c="r")
            # self.ax_raga.legend()
            self.ax_raga.set_title(self.idx_to_raga[idx])
            plt.draw()
        else:
            self.raga_plot = self.ax_raga.plot(arr , label="CNN" , c="b")
            self.ax_raga.plot(lstm_arr , label="LSTM" , c="r")
            self.ax_raga.legend()
            self.ax_raga.set_title(self.idx_to_raga[idx])
            plt.draw()

    def show_raga_graph_lstm(self , idx):
        arr = []

        for output in self.lstm_output_history:
            arr.append(output[idx])

        plt.plot(arr)
        plt.title("Confidence of Raga " + str(self.idx_to_raga[idx]) + " with time according to LSTM")
        plt.show()


    def update_cnn_graph(self):
        ragas = {}
        for raga in self.idx_to_raga.values():
            if raga not in self.result_history["cnn_agg"].keys():
                ragas[raga] = 0
            else:
                ragas[raga] = self.result_history["cnn_agg"][raga]

        if hasattr(self , 'cnn_plot'):
            print("Updating CNN")
            self.ax_cnn.clear()
            self.ax_cnn.bar(ragas.keys() , ragas.values())
            self.ax_cnn.set_title("CNN predictions")
            # plt.xticks(rotation=90)
            self.ax_cnn.set_xticklabels(ragas.keys() , rotation=90)
            plt.ion()
            plt.tight_layout()
            plt.draw()
            
        else:
            self.cnn_plot  = self.ax_cnn.bar(ragas.keys() , ragas.values())
            # plt.show()
            # self.fig_cnn.canvas.draw()
            print("Displaying CNN")
            plt.ion()
            self.ax_cnn.set_xticklabels(self.ax_cnn.get_xticklabels() , rotation=90)
            self.ax_cnn.set_title("CNN prediction")
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)

    def update_lstm_graph(self):
        ragas = {}
        for raga in self.idx_to_raga.values():
            if raga not in self.result_history["lstm_agg"].keys():
                ragas[raga] = 0
            else:
                ragas[raga] = self.result_history["lstm_agg"][raga]

        if hasattr(self , 'lstm_plot'):
            print("Updating LSTM")
            self.ax_lstm.clear()
            self.ax_lstm.bar(ragas.keys() , ragas.values())
            self.ax_lstm.set_title("LSTM predictions")
            # plt.xticks(rotation=90)
            self.ax_lstm.set_xticklabels(ragas.keys() , rotation=90)
            plt.ion()
            plt.tight_layout()
            plt.draw()
            
        else:
            self.lstm_plot  = self.ax_lstm.bar(ragas.keys() , ragas.values())
            # plt.show()
            # self.fig_cnn.canvas.draw()
            print("Displaying CNN")
            plt.ion()
            self.ax_lstm.set_xticklabels(ragas.keys() , rotation=90)
            self.ax_lstm.set_title("LSTM prediction")
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)

    def update_prediction_text(self):
        font_size = 20

        cnn_conf = self.cnn_output_history[-1][self.raga_to_idx[self.result_history["cnn"][-1]]].item()
        self.ax_text.clear()
        self.ax_text.axis("off")
        self.ax_text.text(0.5 , 0.1 , "Raga ++" , horizontalalignment="center" , verticalalignment="center" , fontsize = font_size)
        self.ax_text.text(0.5 , 0.5 , "CNN : " + self.result_history["cnn"][-1] + " ( " + str(cnn_conf)[:5] + " )", horizontalalignment="center" , verticalalignment="center" , fontsize = font_size)
        
        if len(self.result_history["lstm"] ) > 0:
            lstm_conf = self.lstm_output_history[-1][self.raga_to_idx[self.result_history["lstm"][-1]]].item()
            self.ax_text.text(0.5 , 0.7 , "LSTM : " + self.result_history["lstm"][-1] + " ( " + str(lstm_conf)[:5] + " )", horizontalalignment="center" , verticalalignment="center" , fontsize = font_size)
        # self.ax_text.text(1 , 5 , "LSTM : " + self.result_history['lstm'][-1])

    def display_wordcloud_cnn(self):
        wordcloud = WordCloud().generate_from_frequencies(self.result_history["cnn_agg"])
        self.wordcloud = self.ax_wordcloud.imshow(wordcloud)
        print("Displaying wordcloud")
        
        plt.tight_layout(pad=0)
        plt.ion()
        self.ax_wordcloud.set_axis_off()
        self.ax_wordcloud.set_title("Wordcloud of CNN")
        plt.pause(0.001)

    def update_wordcloud_cnn(self):
        wordcloud = WordCloud().generate_from_frequencies(self.result_history['cnn_agg'])
        self.wordcloud.set_data(wordcloud)
        print("Updating wordcloud")
        plt.ion()
        # plt.axis("off")
        plt.draw()
        plt.pause(0.001)

    def display_spec(self , image):
        plt.ion()

        if hasattr(self , "spec_image"):
            self.spec_image.set_data(image)
            plt.draw()
            plt.pause(0.001)
        else:
            self.spec_image = self.ax_spec.imshow(image)
            self.ax_spec.set_ylabel("")
            self.ax_spec.tick_params(axis="both" ,bottom=False , left=False)
            self.ax_spec.axis("off")
            self.ax_spec.set_title("Current spectrogram")
            plt.show()



    def load_model(self): 
        """
        Load CNN and LSTM models
        """
        # model = torch.load(self.model_path)
        model = CNN(output_size = 30)
        model.load_state_dict(torch.load(self.model_path , map_location=torch.device("cpu")) )
        model = model.eval()
        self.cnn_model = model
        self.cnn_half = nn.Sequential(*list(model.children())[:8])

        model_seq = LSTMModel(input_size = 7488 , output_size = 30)
        model_seq.load_state_dict(torch.load(self.lstm_model_path , map_location=torch.device("cpu")) )
        model_seq = model_seq.eval()
        self.lstm_model = model_seq

    def preprocess(self , audio_path):
        # self.separate_vocals(audio_path , " ./")
        # self.separate_slience("sample/vocals.wav" , "./")
        # self.make_spec("sample.wav" , "./temp")
        # self.remove_slient()
        pass


    def predict_continuous(self , audio , fs , save=False , output_path = "./temp"):
        """
        Predict continuously 
        """
        feature_vector_queue = []
        self.result_history = {"cnn":[] , "lstm":[] , "cnn_agg":{} , "lstm_agg":{}}
        
        # Parameters for generating spectrograms
        window_length_in_sec = 10
        window_hop_in_sec = 5

        start = 0
        end = window_length_in_sec * fs
        idx = 0

        # While audio doesn not end
        while end < len(audio):

            # Perform STFT operation
            D_block = librosa.stft(audio[start:end] , center=False , n_fft = 4096 , hop_length=64)

            # Generationg of spectrogram
            plt.ioff()
            fig , ax = plt.subplots()

            canvas = FigureCanvasAgg(fig)

            librosa.display.specshow(D_block , y_axis="linear" , ax=ax )
            ax.set_ylim([0,1024])
            ax.set_ylabel("")
            plt.tick_params(axis="both" ,bottom=False , left=False)
            ax.get_yaxis().set_visible(False)
            fig.tight_layout(pad=0)

            # Save spectrogram
            if save:
                plt.savefig(os.path.join(output_path , str(idx) + ".png" ) , bbox_inches="tight" , format="png" , pad_inches=0)
            canvas.draw()

            # Converting maplotlib figure to numpy array
            image = np.fromstring(canvas.tostring_rgb() , dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = Image.fromarray(image)

            # Get CNN output and store in queue
            feature_vector  , pred , output = self.get_cnn_output(image)
            print("CNN : " , idx*window_hop_in_sec , "-", idx*window_hop_in_sec + window_length_in_sec , "  " , self.idx_to_raga[pred] , "  Confidence : " , output[0][pred])
            self.result_history["cnn"].append(self.idx_to_raga[pred])
            self.cnn_output_history.append(output[0])

            # Storing results in dictionary
            if self.idx_to_raga[pred] not in self.result_history["cnn_agg"]:
                self.result_history["cnn_agg"][self.idx_to_raga[pred]] = 1
            else:
                self.result_history["cnn_agg"][self.idx_to_raga[pred]] += 1

            # if queue does not have seq_length feature vectors in it, simply append
            if len(feature_vector_queue) < self.seq_length:
                feature_vector_queue.append(feature_vector)
            # else pop the first feature vector and enqueue new feature vector
            else:
                feature_vector_queue.pop(0)
                feature_vector_queue.append(feature_vector)

                # Get LSTM output
                pred , output = self.get_lstm_output(feature_vector_queue)
                print("LSTM : " ,idx*window_hop_in_sec , "-", idx*window_hop_in_sec + window_length_in_sec, "  " , self.idx_to_raga[pred] , "  COnfidence : " , output[pred])
                self.result_history['lstm'].append(self.idx_to_raga[pred])
                self.lstm_output_history.append(output)

                # Storing results in dictionary
                if self.idx_to_raga[pred] not in self.result_history["lstm_agg"]:
                    self.result_history["lstm_agg"][self.idx_to_raga[pred]] = 1
                else:
                    self.result_history["lstm_agg"][self.idx_to_raga[pred]] += 1

            # Closing figure and calling garbage collector for memory optimization
            fig.clf()
            plt.close(fig=fig)
            gc.collect()

            idx += 1
            start += window_hop_in_sec * fs
            end += window_hop_in_sec * fs

            # Update dashboard
            if idx == 1:
                self.display_wordcloud_cnn()
            else:
                self.update_wordcloud_cnn()
            self.show_raga_graph(self.track_idx)
            self.update_cnn_graph()
            self.update_lstm_graph()
            self.update_prediction_text()
            self.display_spec(image)
            
        print(self.result_history)

    def get_cnn_output(self , image):
        """
        Get output from CNN
        """
        with torch.no_grad():
            # Transform image to Tensor
            image = self.transforms(image)
            image = image.unsqueeze(0)
            
            # Get output of entire CNN model
            output = self.cnn_model(image)
            idx = np.argmax(output)

            # Get feature vector output
            feature_vector = self.cnn_half(image)

        return feature_vector , idx.item() , torch.exp(output)

    def get_lstm_output(self , queue):
        """
        Get output from LSTM
        """
        with torch.no_grad():
            data = torch.stack(queue*16)
            data = torch.squeeze(data , dim=0)
            data = data.reshape(5, -1, 7488)

            # Get output from LSTM model
            output = self.lstm_model(data)
            pred = output.data.max(1, keepdim=True)[1]
            output = output[0]
            return pred[0].item() , torch.exp(output)

    def separate_vocals(self , audio_path , output_path):
        """
        Perform audio source separation using spleeter
        """
        os.system('spleeter separate -i ' + audio_path + " -o " + output_path + " -p spleeter:2stems -d 1200")

    def make_spec(self , audio , fs , output_path):
        window_length_in_sec = 10
        window_hop_in_sec = 5

        start = 0
        end = window_length_in_sec * fs
        idx = 0
        while end < len(audio):
            D_block = librosa.stft(audio[start:end] , center=False , n_fft = 2048 , hop_length=64)
            fig , ax = plt.subplots()
            librosa.display.specshow(D_block , y_axis="linear" , ax=ax)
            ax.set_ylim([0,1024])
            ax.set_ylabel("")
            plt.tick_params(axis="both" ,bottom=False , left=False)
            plt.yticks()
            ax.get_yaxis().set_visible(False)

            plt.savefig(os.path.join(output_path , str(idx) + ".png" ) , bbox_inches="tight" , format="png" , pad_inches=0)
            fig.clf()
            plt.close()
            gc.collect()

            idx += 1
            start += window_hop_in_sec * fs
            end += window_hop_in_sec * fs

    def remove_slient(self , audio , fs , save=False):
        """
        Remove slient parts by thresholding
        """
        intervals = split(audio)
        result = np.array([])
        for interval in intervals:
            result = np.append(result , audio[interval[0]:interval[1]])

        if save:
            scipy.io.wavfile.write(os.path.join(str(index)+".wav") , fs , result)
        
        return result

if __name__ == "__main__":
    # print(help(separator.Separator))
    parser = argparse.ArgumentParser()
    parser.add_argument("-c" , "--cnn" , help="Path to CNN model")
    parser.add_argument("-l" , "--lstm" , help="Path to LSTM model")
    parser.add_argument("-i" , "--input" , help="Input file path")
    parser.add_argument("-s" , "--spleeter" , help="Vocal separation output path")
    parser.add_argument("-t" , "--track" , help="Raga to track")

    args = parser.parse_args()

    raga = RagaDetector(args.cnn , args.lstm)

    # TODO automate loading of audio while using spleeter
    if args.spleeter:
        raga.separate_vocals(args.input , "./")
        # audio , fs = librosa.load(args.spleeter)
    else:
        raga.load_model()
        audio , fs = librosa.load(args.input)
        audio = raga.remove_slient(audio , fs)
        raga.track_idx = raga.raga_to_idx[args.track]
        raga.predict_continuous(audio,fs)

        if args.track:
            raga.show_raga_graph(raga.raga_to_idx[args.track])
            raga.show_raga_graph_lstm(raga.raga_to_idx[args.track])
