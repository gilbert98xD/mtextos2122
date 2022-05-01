"""
## Minería de textos
Universidad de Alicante, curso 2021-2022

Esta documentación forma parte de la práctica "[Lectura y documentación de un sistema de
extracción de entidades](https://jaspock.github.io/mtextos2122/bloque2_practica.html)" y se 
basa en el código del curso [CS230](https://github.com/cs230-stanford/cs230-code-examples) 
de la Universidad de Stanford.

**Autores de los comentarios:** Gilbert Lurduy & Enrique Moreno

Este módulo define la red neuronal, la función de pérdida y la métrica de aciertos
para la evaluación del modelo. Se hace uso de la libería torch.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Definición de la clase red neuronal
    """

    def __init__(self, params):
        """
        Se define una red neuronal recurrente para la obtención de entidades
        nombradas de un texto. Se compone de tres capas: capa lineal de embedding, 
        capa LSTM y capa 'fully-connceted'.
        """


        """
        Llama al constructor de la clase 'Params', se construye su clase y  a 
        continuación la clase hija 'Net'
        """
        super(Net, self).__init__() 


        """
        Se le da el tamaño del vocabulario y las dimensiones del embedding 
        a la capa de embedding
        """
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)


        """
        Capa LSTM que recibe como parámetros las dimensiones del embedding
        y las dimensiones del estado 'hidden' que no tienen porqué coincidir
        batch_first = True -> hace que los tensores de entrada y salida se den
        de forma batch,seq,feature
        """
        self.lstm = nn.LSTM(params.embedding_dim,
                            params.lstm_hidden_dim, batch_first=True)


        # MATRIZ
        # PARAMS.NUMBER_OF_TAGS:
        # TANTOS NUMEROS COMO ETIQUETAS TENGA, 

        """
        Capa 'fully-connected', es la capa que da el output final, me dice la 
        probabilidad de que la palabra sea una ner (named entitty recognition) tag
        de cierto tipo (nombre, tiempo, lugar...)
        """
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

        """
        CONCLUSIÓN: Tenemos tres capas: la primera dada una palabra me da su embedding,
        la segunda ese embedding se lleva a otros espacio de embeddings que no tiene porque tener la misma dimension,
        y la tercera capa se lleva este nuevo embedding a otro espacio, que en este caso es el numero de etqieuta,
        probablidad de cada una de...?
        """



   
    def forward(self, s): 
        """
        Funcionalidad??
        """
    # METODO QUE SE INVOCA CUANDO, COMO SE VA PROCESANOD LA INFO EN UNA D ESAS
    # CAPAS CUANDO ESTOY USANDO LA RED NEURNOAL
    # HACIA DELANTEÇ??
        # variable input s, con dimensiones x

        """
        aplicamos una capa de embedding
        las dimensiones resultantes son(x,dimension de los embeddings)
        """
        s = self.embedding(s) 

        """
        Aplicación de una LSTM
        """
        s, _ = self.lstm(s) 

        """
        Se hace una copia del tensor en memoria
        """
        s = s.contiguous() 

        """
        Cambiamos la forma de la variable s de tal manera que cada fila tiene un token
        """
        s = s.view(-1, s.shape[2]) # cambiamos la forma de la variable s de tal manera que
        # cada fila tiene un token

        s = self.fc(s) # aplicación de capa 'fully-connected'

        return F.log_softmax(s, dim=1) # aplicamos una softmax seguida del logarimto log(softmax(argument)) 
        # en la dimensión indicada


def loss_fn(outputs, labels): 
    """
    método función de pérdida
    """
    labels = labels.view(-1) # aplana la variable

    mask = (labels >= 0).float() # para que coincidan los tamaños de las muestras se hace 'padding', que es
    # añadir ceros a las secuencias para la coincidencia. Estos token tienen -1 como etiqueta, por lo que
    # podemos usar una máscara que los excluya del cálculo de la función de pérdida
    # (ALMENOS YO LO HE ENTENDIDO ASÍ)

    labels = labels % outputs.shape[1] # conversión de las etiquetas en positivas (por los padding tokens)

    num_tokens = int(torch.sum(mask))

    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens
    # se devuelve la entropía cruzada de todos los tokens, menos los de padding, mediante el uso
    # de la variable 'mask' que hace de máscara, la cual hemos definido antes


def accuracy(outputs, labels):
    """
    Cálculo de la precisión a partir de las etiquetas y las salidas teniendo en cuenta los términos
    # de padding
    """
    labels = labels.ravel() # aplanamiento de la variable

    mask = (labels >= 0) # máscara similar al anterior método 'loss_fn'

    outputs = np.argmax(outputs, axis=1) # índices con los mayores valores, es decir, 
    # obtención de las clases más probables de cada token

    return np.sum(outputs == labels)/float(np.sum(mask)) # precisión/tasa de acierto


metrics = {
    'accuracy': accuracy,
}
