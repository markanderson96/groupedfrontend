import os
import numpy as np

#torchaudio
import torchaudio
import platform
if platform.system() == 'Windows':
    torchaudio.set_audio_backend("soundfile") # using torchaudio on a windows machine

from groupedfrontend.frontend import GroupedFrontend


import unittest

class GammatoneTest(unittest.TestCase):
    def setUp(self):
        self.audio, self.fs = torchaudio.load("test/resources/arctic_b0041.wav")
        print("ok?!")

    def default_test(self):
        frontend = GroupedFrontend(
                n_filters=40,
                min_freq=80.,
                max_freq=8000.,
                sample_rate=16000,
                compression=None,
                init_filter="mel",
                filter_type="gammatone"
            )

        x = frontend(self.audio)
        x = x.detach().numpy()
        # x = x.squeeze().T
        # x.astype(np.float32).tofile(d[2])
