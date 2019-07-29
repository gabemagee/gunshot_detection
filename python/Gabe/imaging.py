import PIL
import librosa
import numpy as np


def numpy_to_image(numpy_array):
    sz = numpy_array.shape
    img = PIL.Image.new(mode = "L",size=sz)
    img.putdata(numpy_array)
    return img


example = np.random.randint(255, size=(100, 100))
print(example)
img = numpy_to_image(example)
img.show()
