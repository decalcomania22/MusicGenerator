from tensorflow.keras import models,utils
from preprocess import SEQUENCE_LENGTH,MAPPING_PATH
import numpy as np

class MelodyGenerator:

    def __init__(self,model_path="model.h5"):

        self.model_path-model_path
        self.model=models.load_model(model_path)

        with open(MAPPING_PATH,"r") as fp:
            self._mappings=json.load(fp)

        self._start_symbols=["/"]* SEQUENCE_LENGTH

    def generate_melody(self,seed,nums_steps,max_sequence_length,temperature):

        #create seed with start symbols
        seed=seed.split()
        melody=seed
        seed=self._start_symbols+seed

        #map seed to int
        seed=[self._mappings[symbol] for symbol in seed]

        for _ in range(nums_steps):

            #limit the seed to max_sequence_length
            seed=seed[-max_sequence_length:]

            #one-hot encode the seed
            onehot_seed=utils.to_categorical(seed,num_classes=len(self._mappings))

            onehot_seed=onehot_seed[np.newaxis,...]

            #make a prediction
            probabilities=self.model.predict(onehot_seed)[0]

            output_int=self._sample_with_temperature(probabilities,temperature)

    def _sample_with_temperature(self,probabilities,temperature):
        