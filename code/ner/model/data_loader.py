"""
## Minería de textos
Universidad de Alicante, curso 2021-2022

Esta documentación forma parte de la práctica "[Lectura y documentación de un sistema de
extracción de entidades](https://jaspock.github.io/mtextos/bloque2_practica.html)" y se 
basa en el código del curso [CS230](https://github.com/cs230-stanford/cs230-code-examples) 
de la Universidad de Stanford.

**Autores de los comentarios:** Gilbert Lurduy & Enrique Moreno

Este módulo define una única clase (DataLoader) que se encarga de realizar diversas funciones con los datos.
"""


import random
import numpy as np
import os
import sys

import torch
from torch.autograd import Variable

import utils 

class DataLoader(object):
    """
    Guarda los parámetros del dataset, el vocabulario y las etiquetas junto a sus respectivos índices
    """
    
    def __init__(self, data_dir, params):

        """
        ### Función de inicialización
        Carga los parámetros del dataset, el vocabulario y sus etiquetas

        #### Parámetros:
        * `data_dir`: directorio donde se encuentra el dataset
        * `params`: parámetros del entrenamiento
        """

        # Cargar los parámetros del dataset
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        self.dataset_params = utils.Params(json_path)        
        
        """
        Una frase es representada por la secuencia de índices de las palabras en la frase. Por ejemplo,
        si el vocabulario es:

            { "ejemplo": 1,
              "Ésto": 2,
              "un": 3,
              "es": 4,
              ".": 5
            }

        entonces la frase "Ésto es un ejemplo." tendría la representación [2,4,3,1,5]. En el siguiente paso creamos el vocabulario.
        """

        # Crear el mapeo entre el vocabulario y los respectivos índices
        vocab_path = os.path.join(data_dir, 'words.txt')
        self.vocab = {}
        with open(vocab_path) as f:
            for i, l in enumerate(f.read().splitlines()):
                self.vocab[l] = i
        
        # Índices para las palabras UNKnown y los tokens de padding
        self.unk_ind = self.vocab[self.dataset_params.unk_word]
        self.pad_ind = self.vocab[self.dataset_params.pad_word]
        
        # Asociar cada etiqueta a un índice (de manera similar que con el vocabulario)
        tags_path = os.path.join(data_dir, 'tags.txt')
        self.tag_map = {}
        with open(tags_path) as f:
            for i, t in enumerate(f.read().splitlines()):
                self.tag_map[t] = i

        # Actualizar los parámetros en el fichero correspondiente
        params.update(json_path)

    def load_sentences_labels(self, sentences_file, labels_file, d):

        """
        ### Función `load_sentences_labels`
        Carga las frases y etiquetas de los archivos correspondientes. 
        También asocia los tokens (extraídos de las frases) y etiquetas, guardándolos en un diccionario.

        #### Parámetros:
        * `sentences_file`: archivo conteniendo las frases
        * `labels_file`: archivo conteniendo las etiquetas (de extracción de entidades) de las frases en **sentences_file**
        * `d`: diccionario donde guardar los datos cargados
        """

        sentences = []
        labels = []

        # Leer las frases del dataset y convertirlas en una secuencia de índices (obtenidos del vocabulario)
        with open(sentences_file) as f:
            for sentence in f.read().splitlines():
                # Si no se halla el token en el vocabulario, se añade el índice de palabra desconocida (UNK)
                s = [self.vocab[token] if token in self.vocab 
                     else self.unk_ind
                     for token in sentence.split(' ')]

                sentences.append(s)
        
        # Leer las etiquetas y crear una secuencia de índices
        with open(labels_file) as f:
            for sentence in f.read().splitlines():
                # Los índices se obtienen del diccionario asociativo entre etiquetas e índices
                l = [self.tag_map[label] for label in sentence.split(' ')]
                labels.append(l)        

        # Comprobar que cada token tiene una etiqueta
        assert len(labels) == len(sentences)
        for i in range(len(labels)):
            assert len(labels[i]) == len(sentences[i])

        # Guardar las frases, etiquetas y la cantidad de frases
        d['data'] = sentences
        d['labels'] = labels
        d['size'] = len(sentences)

    def load_data(self, types, data_dir):

        """
        ### Función `load_data`
        Carga los datos para cada partición de los datos presentes en `types`. 

        #### Parámetros:
        * `types`: contiene al menos una de las particiones **train**, **val** o **test**
        * `data_dir`: archivo conteniendo el dataset

        #### Devuelve:
        * un diccionario conteniendo las frases, etiquetas y la cantidad de frases para cada partición en **types**
        """

        data = {}
        
        for split in ['train', 'val', 'test']:
            if split in types:
                sentences_file = os.path.join(data_dir, split, "sentences.txt")
                labels_file = os.path.join(data_dir, split, "labels.txt")
                data[split] = {}
                self.load_sentences_labels(sentences_file, labels_file, data[split])

        return data

    def data_iterator(self, data, params, shuffle=False):

        """
        ### Función `data_iterator`
        Genera variables de PyTorch a partir de batches de frases  

        #### Parámetros:
        * `data`: diccionario conteniendo las frases, etiquetas y la cantidad de frases según la partición de datos
        * `params`: parámetros del entrenamiento
        * `shuffle`: booleano que determina si se mezclan los datos

        #### Genera:
        * `batch_data`: variable de PyTorch con los datos de las frases
        * `batch_labels`: variable de PyTorch con las etiquetas de las frases
        """
        
        # Mezclar los datos si así ha sido indicado
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        # Iterar de una pasada todos los datos, respetando el tamaño de los batches
        for i in range((data['size']+1)//params.batch_size):
            # Obtener frases y etiquetas para el batch actual
            batch_sentences = [data['data'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]

            # Calcular la longitud de la frase con mayor longitud
            batch_max_len = max([len(s) for s in batch_sentences])

            # Preparar arrays de NumPy para las frases y etiquetas
            """
            * Cantidad de filas: cantidad de frases en el batch
            * Cantidad de columnas: longitud de la frase más larga
            """

            # Datos rellenados inicialmente con el índice de los tokens de padding 
            # para rellenar el espacio vacío (si la frase no tiene la mayor longitud)
            batch_data = self.pad_ind*np.ones((len(batch_sentences), batch_max_len))
            # Etiquetas rellenadas inicialmente con -1 (para diferenciarlas de los índices de padding)
            batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

            # Rellenar arrays
            for j in range(len(batch_sentences)):
                cur_len = len(batch_sentences[j])
                batch_data[j][:cur_len] = batch_sentences[j]
                batch_labels[j][:cur_len] = batch_tags[j]

            # Transformar el contenido de los arrays (índices) a tipo Long, 
            # ya que la capa de embeddings de PyTorch requiere ese formato para sus inputs
            batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

            # Pasar los tensores a GPU si está disponible
            if params.cuda:
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

            # Convertir los tensores a variables de PyTorch
            batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

            # Obtener generadores de los datos y etiquetas
            yield batch_data, batch_labels
