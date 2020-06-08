import numpy as np
import pandas as pd
import os
import random
import math


dataset = pd.read_csv("emotion.data")

input_sentences = [text.split(" ") for text in dataset["text"].values.tolist()]
labels = dataset["emotions"].values.tolist()

word2id = dict()
label2id = dict()

max_words = 0

for sentence in input_sentences:
    for word in sentence:
        if word not in word2id:
            word2id[word] = len(word2id)
    if len(sentence) > max_words:
        max_words = len(sentence)
    

label2id = {l: i for i, l in enumerate(set(labels))}
id2label = {v: k for k, v in label2id.items()}

#https://keras.io/
import keras
#https://keras.io/preprocessing/sequence/
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

if os.path.exists('emotions.h5'):
    model=load_model('emotions.h5')

else:

    X = [[word2id[word] for word in sentence] for sentence in input_sentences]
    Y = [label2id[label] for label in labels]

    X = pad_sequences(X, max_words)

    Y = keras.utils.to_categorical(Y, num_classes=len(label2id), dtype='float32')


    embedding_dim = 100

    # Define input tensor
    sequence_input = keras.Input(shape=(max_words,), dtype='int32')

    # Word embedding layer
    embedded_inputs =keras.layers.Embedding(len(word2id) + 1, embedding_dim, input_length=max_words)(sequence_input)
if(jkalfjč)
    # Apply dropout to prevent overfitting
    embedded_inputs = keras.layers.Dropout(0.2)(embedded_inputs)

    # Apply Bidirectional LSTM over embedded inputs
    lstm_outs = keras.layers.wrappers.Bidirectional(
        keras.layers.LSTM(embedding_dim, return_sequences=True)
    )(embedded_inputs)

    # Apply dropout to LSTM outputs to prevent overfitting
    lstm_outs = keras.layers.Dropout(0.2)(lstm_outs)

    # Attention Mechanism - Generate attention vectors
    input_dim = int(lstm_outs.shape[2])
    permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
    attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
    attention_vector = keras.layers.Reshape((max_words,))(attention_vector)
    attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

    # Last layer: fully connected with softmax activation
    fc = keras.layers.Dense(embedding_dim, activation='relu')(attention_output)
    output = keras.layers.Dense(len(label2id), activation='softmax')(fc)

    # Finally building model
    model = keras.Model(inputs=[sequence_input], outputs=output)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

    # Train model 10 iterations
    model.fit(X, Y, epochs=2, batch_size=64, validation_split=0.1, shuffle=True)

    ##looking closer to model predictions and attentions
    # Re-create the model to get attention vectors as well as label prediction
    model_with_attentions = keras.Model(inputs=model.input,outputs=[model.output, model.get_layer('attention_vec').output])

    #save model
    model.save('emotions.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model

    # returns a compiled model
    # identical to the previous one
    model = load_model('emotions.h5')

model_with_attentions = keras.Model(inputs=model.input,outputs=[model.output, model.get_layer('attention_vec').output])


# Select random samples to illustrate
sample_text = random.choice(dataset["text"].values.tolist())
##sample_text=('i feel sick')
print(sample_text)

# Encode samples
tokenized_sample = sample_text.split(" ")
encoded_samples = [[word2id[word] for word in tokenized_sample]]

# Padding
encoded_samples = keras.preprocessing.sequence.pad_sequences(encoded_samples, maxlen=max_words)

# Make predictions
label_probs, attentions = model_with_attentions.predict(encoded_samples)
label_probs = {id2label[_id]: prob for (label, _id), prob in zip(label2id.items(),label_probs[0]*100)}

# Get word attentions using attenion vector
token_attention_dic = {}
max_score = 0.0
min_score = 0.0
for token, attention_score in zip(tokenized_sample, attentions[0][-len(tokenized_sample):]):
    token_attention_dic[token] = math.sqrt(attention_score)

emotions = [label for label, _ in label_probs.items()]
scores = [score for _, score in label_probs.items()]

print("Percent of every of 6 emotions in sentence: ")
print(label_probs)



