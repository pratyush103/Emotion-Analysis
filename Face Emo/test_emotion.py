import cv2
import pyaudio
import wave
import threading
import time
import os
from deepface import DeepFace

# Global variables for audio recording
FRAME_LENGTH_MS = 50  # Frame length in milliseconds
FRAME_SHIFT_MS = 12.5  # Frame shift in milliseconds
SAMPLE_RATE = 22050  # Sample rate in Hz
RECORD_SECONDS = 30
output_directory = "Output"
WAVE_OUTPUT_FILENAME = os.path.join(output_directory, "temp_audio.wav")
VIDEO_OUTPUT_FILENAME = os.path.join(output_directory, "output.avi")
audio_frames = []
final_emotion = None

# Initialize rectangle coordinates
x, y, w, h = 0, 0, 0, 0

# Audio recording function
def record_audio():
    audio = pyaudio.PyAudio()
    
    # Calculate CHUNK size based on the desired frame length and sample rate
    chunk_size = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)
    
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording audio...")
    
    frames = []
    
    for i in range(0, int(SAMPLE_RATE / chunk_size * RECORD_SECONDS)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Audio recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Video recording and face sentiment analysis function
def record_video_and_analyze():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(VIDEO_OUTPUT_FILENAME, fourcc, 20.0, (640, 480))

    print("Recording video and analyzing face sentiment...")
    start_time = time.time()
    while (int(time.time() - start_time) < RECORD_SECONDS):
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face analysis and sentiment detection
        try:
            results = DeepFace.analyze(frame, actions=['emotion'])
            for result in results:
                if len(result) > 0:
                    # Update rectangle coordinates to snap to the detected face
                    x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                    
                    # Store the last detected emotion
                    emotion = result['emotion']
                    final_emotion = max(emotion, key=emotion.get)

                    # Display the detected emotion as a percentage
                    emotion_percentage = emotion[final_emotion]
                    color = (0, 255, 0) if final_emotion == 'happy' else (0, 0, 255)
                    cv2.putText(frame, f"{final_emotion}: {emotion_percentage:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    print("Detected Emotion: ", final_emotion)
                else:
                    print("No face detected")
        except Exception as e:
            print("Error analyzing face:", str(e))
        
        # Write frame to video
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Video recording and face sentiment analysis finished.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print final emotion result
    if final_emotion:
        print("Final Emotion:", final_emotion)
    else:
        print("No emotion detected.")

if __name__ == "__main__":
    audio_thread = threading.Thread(target=record_audio)
    video_thread = threading.Thread(target=record_video_and_analyze)

    audio_thread.start()
    video_thread.start()

    audio_thread.join()
    video_thread.join()
