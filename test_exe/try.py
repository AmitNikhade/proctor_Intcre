import argparse 
import sys
from queue import Queue
import logging
import speech_recognition as sr
from tempfile import NamedTemporaryFile
import torch
import io, time
import threading


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main_vad(running_event):            
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy_threshold", default=1000, help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2, help="How real time the recording is in seconds.", type=float)
    if 'linux' in sys.platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    
    # Important for linux users. 
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in sys.platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    record_timeout = args.record_timeout
    temp_file = NamedTemporaryFile().name
    
    with source:
        recorder.adjust_for_ambient_noise(source)
        logger.info("Adjusted for ambient noise.")

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        data = audio.get_raw_data()
        data_queue.put(data)
        logger.debug("Audio data received and added to queue.")

    # Load silero VAD model
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    logger.info("VAD model loaded.")

    # Start the recording process
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    logger.info("Recording started.")

    while running_event.is_set():
        try:
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                logger.debug("Processing audio data from queue.")
                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                wav = read_audio(temp_file, sampling_rate=16000)
                speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

                if speech_timestamps:
                    logger.info('Speech Detected!')
                    # Here you can add code to handle the detected speech
                    # For example, you might want to transcribe it or perform some action
                else:
                    logger.info('Silence Detected')

                # Clear the last sample to start fresh
                last_sample = bytes()
            else:
                # Sleep briefly to avoid busy-waiting
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            # Optionally break the loop if there's a critical error
            # break

    # Cleanup code for VAD
    logger.info("VAD thread stopping...")


running_event = threading.Event()

main_vad(running_event)