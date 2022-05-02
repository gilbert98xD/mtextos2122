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
    ### Clase 'Net'
    Definición de la clase red neuronal
    """

    def __init__(self, params):
        """
        ### Constructor
        Se define una red neuronal recurrente para la obtención de entidades
        nombradas de un texto. Se compone de tres capas: capa lineal de embedding, 
        capa LSTM y capa 'fully-connceted'.
        
        #### Parámetros:
            
        * 'params': parámetros con 'vocab_size', 'embedding_dim' y 'lstm_hidden_dim'
           
        #### Devuelve:
            
        * Tres capas para la red nuronal
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


        """
        Capa 'fully-connected', es la capa que da el output final, me dice la 
        probabilidad de que la palabra sea una ner (named entitty recognition) tag
        de cierto tipo (nombre, tiempo, lugar...)
        """
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

        """
        En resumen la primera capa, dada una palabra, me da su embedding, en la segunda ese embedding 
        se lleva a otros espacio de embeddings que no tiene porque tener la misma dimension, y la tercera
        capa se lleva este nuevo embedding a otro espacio, el número de etiqueta
        """



   
    def forward(self, s): 
        """
        ### Función 'forward'
        A partir de un batch input obtiene las probablidades logits de los tokens
        
        #### Parámetros:
            
        * 's': argumento con un 'lote' de oraciones organizados en filas
        y de dimensión tamaño del batch x longitud frase más larga. A las
        frases más cortas se le aplica padding.
            
        #### Devuelve:
        
        * probabilidades logits de los tokens
        """

        """
        aplicamos una capa de embedding
        las dimensiones resultantes son(x,dimension de los embeddings)
        """
        s = self.embedding(s) 

        """
        Aplicación de la LSTM
        """
        s, _ = self.lstm(s) 

        """
        Se hace una copia del tensor en memoria
        """
        s = s.contiguous() 

        """
        Cambiamos la forma de la variable s (es una matriz) de tal manera que cada fila tiene un token.
        Con el -1 le indicamos que calcule la dimensión automáticamente para obtener dos dimensiones. Y el
        s.shape[2] es lstm_hidden_dim. Se le pone el [2] porque el [0] es el tamaño de batch y el [1] es
        el máximo de la secuencia
        """
        s = s.view(-1, s.shape[2]) 

        """
        Última capa 'fully-connected'proyecta el nuevo embedding hacia un espacio con el número de etqiuetas
        """
        s = self.fc(s) 

        """
        No obstante, aun no tenemos probabilidades hay que aplicar una softmax. Por una mayor
        eficiencia se aplica un log(softmax) por lo que las probabilidades de 0 a 1 pasan a ser
        negativas. Cuanto más cerca estemos del cero más alta es la probabilidad.
        """
        return F.log_softmax(s, dim=1)
 

def loss_fn(outputs, labels): 
    """
    ### Función 'loss_fn'
    Método función de pérdida
    
    #### Parámetros:
        
    * 'outputs': resultados del modelo
    * 'labels': las etiqeutas para evaluar la pérdida
        
    #### Devuelve:
        
    * La entro`pía cruzada de todos los tokens, menos los de padding
    """


    """
    aplana la variable
    """
    labels = labels.view(-1) 

    """
    Los inputs de una red neuronal deben tener la misma forma y tamaño, para que esto sea así al pasar oraciones
    se hace 'padding', que añade ceros a las secuencias o corta oraciones largas. Estos token tienen -1 como etiqueta, 
    por lo que podemos usar una máscara que los excluya del cálculo de la función de pérdida.
    """
    mask = (labels >= 0).float() 

    """
    Conversión de las etiquetas en positivas (por los padding tokens)
    """
    labels = labels % outputs.shape[1] 

    num_tokens = int(torch.sum(mask))

    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens
    """
    Se devuelve la entropía cruzada de todos los tokens, menos los de padding, mediante el uso
    de la variable 'mask' que hace de máscara, la cual hemos definido antes
    """



def accuracy(outputs, labels):
    """
    ### Función 'accuracy'
    Cálculo de la precisión a partir de las etiquetas y las salidas teniendo en cuenta los términos
    de padding
    
    #### Parámetros:
        
    * 'outputs': resultados del modelo
    * 'labels': las etiqeutas para evaluar la pérdida
        
    #### Devuelve:
        
    * Tasa de acierto
    """

    """
    Aplanamiento de la variable
    """
    labels = labels.ravel() 

    """
    Máscara similar al anterior método 'loss_fn'
    """
    mask = (labels >= 0) 

    """
    Índices con los mayores valores, es decir, obtención de las clases más probables de cada token
    """
    outputs = np.argmax(outputs, axis=1) 

    return np.sum(outputs == labels)/float(np.sum(mask)) 


metrics = {
    'accuracy': accuracy,
}
