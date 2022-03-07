"""
## Minería de textos
Universidad de Alicante, curso 2021-2022

Esta documentación forma parte de la práctica "[Lectura y documentación de un sistema de
extracción de entidades](https://jaspock.github.io/mtextos/bloque2_practica.html)" y se 
basa en el código del curso [CS230](https://github.com/cs230-stanford/cs230-code-examples) 
de la Universidad de Stanford.

**Autores de los comentarios:** Gilbert Lurduy & Enrique Moreno

Este módulo crea los siguientes ficheros a partir del contenido del directorio indicado 
como argumento: `words.txt` y `tags.txt`, que contienen aquellas palabras y etiquetas de
mayor o igual frecuencia que lo establecido en los argumentos del programa, respectivamente.
"""

import argparse
from collections import Counter
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--min_count_tag', default=1, help="Minimum count for tags in the dataset", type=int)
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")

PAD_WORD = '<pad>'
PAD_TAG = 'O'
UNK_WORD = 'UNK'

def save_vocab_to_txt_file(vocab, txt_path):
    """
    ### Función `save_vocab_to_txt_file`
    Crea un archivo de texto compuesto por un token por línea

    #### Parámetros:

    * `vocab`: proporciona los tokens que compondrán el archivo de texto
    * `txt_path`: ruta de escritura del archivo de texto de los tokens
    """
    with open(txt_path, "w") as f:  # Este tipo de comentario es ignorado por pycco.
        for token in vocab:
            f.write(token + '\n')
            

def save_dict_to_json(d, json_path):
    """
    ### Función `save_dict_to_json`
    Convierte un diccionario a JSON y guarda el JSON creado

    #### Parámetros:

    * `d`: diccionario a convertir a JSON
    * `json_path`: ruta del JSON a donde es convertido el diccionario
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        """
        La variable `d` es un diccionario compuesto por las parejas formadas por k (key) y v (value) provenientes del diccionario `d` (parámetro de entrada)
        """
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab):
    """
    ### Función `update_vocab`
    Actualiza la cuenta de tokens del dataset de entrada 

    #### Parámetros:

    * `txt_path`: ruta del dataset de entrada (un token por línea)
    * `vocab`: dataset del que se cuenta (y actualiza) la frecuencia de sus tokens
    
    #### Devuelve:
    
    * la cantidad de tokens en el dataset `vocab`
    """
    with open(txt_path) as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))

    return i + 1


if __name__ == '__main__':

    args = parser.parse_args()
    
    print("Building word vocabulary...")
    # Counter de palabras para contar la frecuencia de las palabras
    words = Counter()
    # Se calculan los tamaños de los datasets de frases de entrenamiento, validación y test, a la par que se actualiza el Counter de palabras
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train/sentences.txt'), words)
    size_dev_sentences = update_vocab(os.path.join(args.data_dir, 'val/sentences.txt'), words)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test/sentences.txt'), words)
    print("- done.")

    print("Building tag vocabulary...")
    # Counter de etiquetas para contar la frecuencia de las etiquetas
    tags = Counter()
    # Se calculan los tamaños de los datasets de labels de entrenamiento, validación y test, a la par que se actualiza el Counter de etiquetas
    size_train_tags = update_vocab(os.path.join(args.data_dir, 'train/labels.txt'), tags)
    size_dev_tags = update_vocab(os.path.join(args.data_dir, 'val/labels.txt'), tags)
    size_test_tags = update_vocab(os.path.join(args.data_dir, 'test/labels.txt'), tags)
    print("- done.")
    
    # Se comprueba que la cantidad de tokens y etiquetas en cada dataset sean iguales
    assert size_train_sentences == size_train_tags
    assert size_dev_sentences == size_dev_tags
    assert size_test_sentences == size_test_tags
    
    # Se mantienen aquellas palabras y etiquetas de mayor o igual frecuencia que lo establecido en los argumentos del programa
    words = [tok for tok, count in words.items() if count >= args.min_count_word]
    tags = [tok for tok, count in tags.items() if count >= args.min_count_tag]
    
    # Añadir tokens de padding
    if PAD_WORD not in words: words.append(PAD_WORD)
    if PAD_TAG not in tags: tags.append(PAD_TAG)
    
    # Añadir palabra que simboliza una palabra desconocida
    words.append(UNK_WORD)
    
    # Guardar las palabras y etiquetas en archivos distintos
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(words, os.path.join(args.data_dir, 'words.txt'))
    save_vocab_to_txt_file(tags, os.path.join(args.data_dir, 'tags.txt'))
    print("- done.")
    
    # Guardar propiedades del dataset en un JSON
    sizes = {
        'train_size': size_train_sentences,
        'dev_size': size_dev_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words),
        'number_of_tags': len(tags),
        'pad_word': PAD_WORD,
        'pad_tag': PAD_TAG,
        'unk_word': UNK_WORD
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))
    
    # Imprimir características del dataset
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))
