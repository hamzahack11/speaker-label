from flask import Flask, render_template, request, jsonify
import subprocess
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
import whisper
from pydub import AudioSegment
import datetime
import os
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
from collections import OrderedDict
import json
import subprocess


app = Flask(__name__)




@app.route('/')
def index():
    return 'Hello'


model = whisper.load_model("medium")
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
audio = pyannote.audio.Audio(mono="downmix")


UPLOAD_FOLDER = os.path.join(os.curdir, "uploads")
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ulaw', 'mulaw', 'ogg', 'opus', 'mov', 'm4v', 'mp4'}

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mkv', 'avi'}

def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

@app.route('/transcribe-speaker', methods=['POST'])
def transcribe_api():
    if 'file' not in request.files:
        return "error: no file"
    file = request.files['file']
    num_speakers = int(request.form['num_speakers'])

    if file.filename == '':
        return "error: no filename"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        try:
            # Check if the uploaded file is a video
            if allowed_video_file(file.filename):
                # Extract audio from the video
                audio_path = extract_audio_from_video(save_path)
                if audio_path is None:
                    return "error: failed to extract audio from video"

                # Convert the extracted audio to WAV format
                wav_path, error = convert_to_wav(audio_path)
                if error:
                    return jsonify({"data": [], "message": error, "status": 400})

                # Perform transcription with the converted audio
                result = transcribe(wav_path, num_speakers)
                delete_saved_file(wav_path)
                return result
                

            # Handle audio files
            else:
                # Convert audio file to WAV format
                wav_path, error = convert_to_wav(save_path)
                if error:
                    return jsonify({"data": [], "message": error, "status": 400})

                # Perform transcription with the converted audio
                result = transcribe(wav_path, num_speakers)
                return result

        except Exception as e:
            return "error: whisper error: " + str(e)

        finally:
            os.remove(save_path)

    else:
        return "error: File not Found"



# -----------------Below are THE Transcribe SPEAKER LABEL FUNCTIONS---------------------------------

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        


def delete_saved_file(file_path):
    try:
        os.remove(file_path)
    except Exception as e:
        print("Error deleting file:", str(e))

def transcribe(path, num_speakers):
    duration = get_duration(path)
    result = model.transcribe(path)
    segments = result["segments"]

    try:
        # Create embedding
        def segment_embedding(segment):
            audio = pyannote.audio.Audio(mono="downmix")
            start = segment["start"]
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(path, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)

        if num_speakers == 0:
            # Find the best number of speakers
            score_num_speakers = {}

            for num_speakers in range(2, 10 + 1):
                clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
                score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
                score_num_speakers[num_speakers] = score
            best_num_speaker = max(score_num_speakers, key=lambda x: score_num_speakers[x])
        else:
            best_num_speaker = num_speakers

        # Assign speaker label based on the order of detection
        clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
        labels = clustering.labels_
        unique_labels = np.unique(labels)
        num_unique_labels = len(unique_labels)
        speaker_mapping = {}
        for i in range(len(segments)):
            if labels[i] not in speaker_mapping:
                if len(speaker_mapping) == 0:
                    speaker_mapping[labels[i]] = "SPEAKER 1"
                elif len(speaker_mapping) == 1:
                    speaker_mapping[labels[i]] = "SPEAKER 2"
                else:
                    speaker_mapping[labels[i]] = f"SPEAKER {len(speaker_mapping) + 1}"
            speaker_label = speaker_mapping[labels[i]]
            segments[i]["speaker"] = speaker_label

        # Make output
        objects = []
        text = ""
        previous_end_time = None
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                if i != 0:
                    objects.append({
                        'start_time': str(convert_time(previous_end_time)),
                        'end_time': str(convert_time(segments[i - 1]["end"])),
                        'speaker': segments[i - 1]["speaker"],
                        'text': text
                    })
                text = ""
            text += segment["text"] + " "
            previous_end_time = segments[i]["end"]

        # Add the last segment
        objects.append({
            'start_time': str(convert_time(previous_end_time)),
            'end_time': str(convert_time(segments[-1]["end"])),
            'speaker': segments[-1]["speaker"],
            'text': text
        })

        # Update the start time of the first segment
        objects[0]['start_time'] = "0:00:00"

        # Update the start time of subsequent segments
        start_time = objects[0]['start_time']
        for i in range(1, len(objects)):
            objects[i]['start_time'] = objects[i-1]['end_time']
            
            
        print(json.dumps(objects))
        return json.dumps(objects, sort_keys=False)

    except Exception as e:
        return "Error during transcription: " + str(e)

def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))

def extract_audio_from_video(video_path):
    try:
        audio_path = video_path.rsplit('.', 1)[0] + '.wav'
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)
        video_clip.close()
        return audio_path
    except Exception as e:
        print("Error extracting audio from video:", str(e))
        return None


def convert_to_wav(save_path):
    try:
        output_path = save_path
        audio = AudioSegment.from_file(save_path)
        audio.export(output_path, format='wav')
        return output_path, None
    except Exception as e:
        return None, f"Error converting audio to WAV: {e}"


def get_duration(path):
    try:
        with contextlib.closing(wave.open(path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except (wave.Error, FileNotFoundError):
        return 0.0

def time(secs):
    return datetime.timedelta(seconds=round(secs))  

if __name__ == '__main__':
    app.run()
 