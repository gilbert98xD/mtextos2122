"""
## Minería de textos
Universidad de Alicante, curso 2021-2022

Esta documentación forma parte de la práctica "[Lectura y documentación de un sistema de
extracción de entidades](https://jaspock.github.io/mtextos2122/bloque2_practica.html)" y se 
basa en el código del curso [CS230](https://github.com/cs230-stanford/cs230-code-examples) 
de la Universidad de Stanford.

**Autores de los comentarios:** Gilbert Lurduy & Enrique Moreno

Este módulo define la red neuronal, la función de pérdida y la métrica de aciertos
para la evaluación del modelo. Se hace uso de la libería torch
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Definición de la clase red neuronal
    """

    def __init__(self, params): # EL CONSTRUCTOR, CUANDO CREAMOS UN NUEVO OBEJTO 
        # DE LA CLASE SE LLAMA? DEFINE LAS CAPAS DE TU RED NEURONAL

        super(Net, self).__init__()
        # llama al constructor de la clase 'Params', se construye su clase y
        # a continuación la clase hija 'Net'

        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        # se le da el tamaño del vocabulario y las dimensiones del embedding
        # a la capa de embedding


        # DE ABAJO LE LLEGA 
        # PROYECTA .. A OTR ADIMENSION PARAMS.LSTM_HIDDEN_DIM

        # SE PROYECTA A UNNUEVO PUNTO DEL ESPACIO
        # EL DE ENTRADA, Y EL PROYECTADO (CONTEXTUAL) NO TIENEN PORQUE TENER EL
        # MISMO TAMAÑO, OPR ESO SE PONE LAS DIMENSIONES

        # capa LSTM que recibe como parámetros las dimensiones del embedding
        # y el número de características del estado 'hidden'
        # batch_first = True -> hace que los tensores de entrada y salida se den
        # de forma batch,seq,feature
        self.lstm = nn.LSTM(params.embedding_dim,
                            params.lstm_hidden_dim, batch_first=True)



        # MATRIZ


        # PARAMS.NUMBER_OF_TAGS:
        # TANTOS NUMEROS COMO ETIQUETAS TENGA, ME DICE LA PROB DE QUE LA PALABRA SEA UN NOMBRE, LUGAR, TIEMPO,.. O CASI?
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

        # CONLSUION: TENEOMS TRES CAPAS, EMBEDDING, LSTM Y FC


    def forward(self, s): # METODO QUE SE INVOCA CUANDO, COMO SE VA PROCESANOD LA INFO EN UNA D ESAS CAPAS CUANDO ESTOY USANDO LA RED NEURNOAL
    # HACIA DELANTEÇ??

        s = self.embedding(s)

        s, _ = self.lstm(s)

        s = s.contiguous()

        s = s.view(-1, s.shape[2])

        s = self.fc(s)

        return F.log_softmax(s, dim=1)


def loss_fn(outputs, labels):

    labels = labels.view(-1)

    mask = (labels >= 0).float()

    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask))

    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens


def accuracy(outputs, labels):

    labels = labels.ravel()

    mask = (labels >= 0)

    outputs = np.argmax(outputs, axis=1)

    return np.sum(outputs == labels)/float(np.sum(mask))


metrics = {
    'accuracy': accuracy,
}
