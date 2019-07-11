import alsaaudio, time

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, card="2")
inp.setchannels(1)
inp.setrate(44100)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
# inp.setperiodsize(160)

while True:
    l, data = inp.read()
    if l:
        print(audio.max(data, 2))
    time.sleep(0.001)
