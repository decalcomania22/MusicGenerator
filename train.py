from preprocess import generating_training_sequences, SEQUENCE_LENGTH
from tensorflow.keras import layers, Model,optimizers

OUTPUT_UNITS=38
LOSS="sparse_categorical_crossentropy"
LEARNING_RATE=0.001
NUM_UNITS=[256,128]
EPOCHS=50
BATCH_SIZE= 64
SAVE_MODEL_PATH="model.h5"

def build_model(output_units,num_units ,learning_rate,loss=LOSS):

    #create the model architecture
    input=layers.Input(shape=(None,output_units))
    x = layers.LSTM(num_units[0])(input)
    x=layers.Dropout(0.2)(x)

    output=layers.Dense(output_units,activation="softmax")(x)
    model = Model(input,output)
    #compile model

    model.compile(loss=loss,optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=["accuracy"])
    
    model.summary()

    return model


def train(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):

    #generate the training sequences
    inputs,targets = generating_training_sequences(SEQUENCE_LENGTH)

    #build the network
    model = build_model(output_units,num_units,loss,learning_rate)

    #train the model
    model.fit(inputs,targets,epochs=EPOCHS,batch_size=BATCH_SIZE)

    #save model
    model.save(SAVE_MODEL_PATH)

if __name__=="__main__":
    train()