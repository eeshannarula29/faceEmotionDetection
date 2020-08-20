import os
import numpy as np

DIM1 = 48
DIM2 = 48
streams = 1

SHAPE_single = shape = shape_single = (DIM1,DIM2)
SHAPE_streamed = shape_streamed = input_shape = (DIM1,DIM2,streams)

shape_streamed_one = (1,DIM1,DIM2,streams)

def shape_for_nsamples(n):
    return (n,DIM1,DIM2,streams)

CATS = CAT = ['Anger',
              'Disgust',
              'Fear',
              'Happiness',
              'Sadness',
              'Surprise',
              'Contempt']


classes = len(CATS)

epochs = 5
lr = 0.0009
activation = 'softmax'
loss = 'categorical_crossentropy'
