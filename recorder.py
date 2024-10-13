from pvrecorder import PvRecorder
import wave
import struct
import os
if not os.path.exists("test_recordings"): os.mkdir("test_recordings")
recorder = PvRecorder(device_index=-1, frame_length=512)
audio = []


try:
    recorder.start()


    while True:
        frame = recorder.read()
        audio.extend(frame)
except KeyboardInterrupt:
    recorder.stop()
    with wave.open("test_recordings/4.wav", 'w') as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        f.writeframes(struct.pack("h" * len(audio), *audio))
finally:
    recorder.delete()