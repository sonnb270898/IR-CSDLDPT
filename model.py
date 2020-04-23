import time
import cv2
import pickle
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras import applications
from keras import backend as K
import numpy as np
import h5py
import faiss
import json


def convnet_model_():
    vgg_model = applications.VGG16(weights=None, include_top=False, input_shape=(221, 221, 3))
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Lambda(lambda x_: K.l2_normalize(x,axis=1))(x)
#     x = Lambda(K.l2_normalize)(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model


def deep_rank_model():
    convnet_model = convnet_model_()

    first_input = Input(shape=(221, 221, 3))
    first_conv = Conv2D(96, kernel_size=(8, 8), strides=(16, 16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)

    second_input = Input(shape=(221, 221, 3))
    second_conv = Conv2D(96, kernel_size=(8, 8), strides=(32, 32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7, 7), strides=(4, 4), padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])
    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    emb = Dense(128)(emb)
    l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)

    final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    return final_model

def find_k_nn(train_emb,query_vec,k):
    dist_vec = -np.matmul(train_emb, query_vec.T)
    return np.argsort(dist_vec.flatten())[:k]


# load model to predict
deep_rank_model = deep_rank_model()
deep_rank_model.load_weights('transfer_learning/triplet_weight.hdf5')

with open('transfer_learning/label.pkl', 'rb') as f:
    Y = pickle.load(f)

with open('transfer_learning/data_path.pkl', 'rb') as f:
    X = pickle.load(f)
with open('transfer_learning/embd128.pkl', 'rb') as f:
    train_emb = pickle.load(f)
    train_emb = train_emb.reshape(train_emb.shape[0], train_emb.shape[2])

# def Predict(im):
#     im = im / 255
#     im = cv2.resize(im, (221, 221))

#     im = np.expand_dims(im, axis=0)
#     embd128 = deep_rank_model.predict([im, im, im])

#     A = find_k_nn(train_emb, embd128, 5)

#     A = A[:5]


#     res = []

#     for index in A:
#         str = X[index].split('/')
#         if len(str[1]) > 16:
#             res.append('data/trộn quần áo/' + str[-1])
#         elif str[1][0] == 'g':
#             res.append('data/giày/' + str[-1])
#         elif str[1][-1] == 'n':
#             res.append('data/quần/' + str[-1])
#         elif str[1][0] == 'v':
#             res.append('data/váy/' + str[-1])
#         elif len(str[1]) < 4:
#             res.append('data/áo/' + str[-1])
#         else:
#             res.append('data/đồ bộ/' + str[-1])

#     a1 = cv2.imread(res[0])
#     a2 = cv2.imread(res[1])
#     a3 = cv2.imread(res[2])
#     a4 = cv2.imread(res[3])
#     # a5 = cv2.imread(res[4])

#     numpy_horizontal_concat = np.concatenate((a1, a2, a3, a4), axis=1)
#     cv2.imshow('res', numpy_horizontal_concat)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

def generate_db_index():
    d = train_emb.shape[1]    # Dimension of vector
    index = faiss.IndexFlatIP(d)  # Build the index
    index.add(train_emb)  # add vectors to the index
    print('faiss indexing done...')
    return index

def predict_image(im):
    im = im / 255
    im = cv2.resize(im, (221, 221))

    im = np.expand_dims(im, axis=0)
    embd128 = deep_rank_model.predict([im, im, im])

    s_time = time.time()
    _, candidate_index = indexed_db.search(embd128,10)
    print('Total time to find nn: {:0.2f} ms'.format((time.time()-s_time)*1000))

    res = []

    for index in candidate_index.tolist()[0]:
        str = X[index].split('/')
        if len(str[1]) > 16:
            res.append('tron_quan_ao/' + str[-1])
        elif str[1][0] == 'g':
            res.append('giay/' + str[-1])
        elif str[1][-1] == 'n':
            res.append('quan/' + str[-1])
        elif str[1][0] == 'v':
            res.append('vay/' + str[-1])
        elif len(str[1]) < 4:
            res.append('ao/' + str[-1])
        else:
            res.append('do_bo/' + str[-1])

    return json.dumps(res[:4])

indexed_db = generate_db_index()