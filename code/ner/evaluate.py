"""
## Minería de textos
Universidad de Alicante, curso 2021-2022

Esta documentación forma parte de la práctica "[Lectura y documentación de un sistema de
extracción de entidades](https://jaspock.github.io/mtextos2122/bloque2_practica.html)" y se 
basa en el código del curso [CS230](https://github.com/cs230-stanford/cs230-code-examples) 
de la Universidad de Stanford.

**Autores de los comentarios:** Gilbert Lurduy & Enrique Moreno

Este módulo contiene una función para la evaluación de la red neuronal y el main del programa
para obtener la tasa de acierto en el conjunto de test para el modelo.
"""


import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.net as net
from model.data_loader import DataLoader



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")



def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps):
    """
    ### Función `evaluate`
    Método que evalua el modelo con 'num_steps' batches

    #### Parámetros:
    * `model`: red neuronal
    * `loss_fn`: función de pérdida (evalúa batch_output vs batch_labels)
    * `data_iterator`: generador de batches de datos y etiquetas
    * `metrics`: diccionario de funciones que calculan una medida entre los output y las verdaderas etiquetas para cada batch
    * `params`: parámetros de entrenamiento
    * `num_steps`: cantidad de batches sobre los que entrenar
    """
    
    """Evaluación del modelo"""
    model.eval()

    summ = []
    
    for _ in range(num_steps):
        
        """
        Obtenemos los datos y las etiquetas de cada batch; data_iterator nos da un generador
        """
        data_batch, labels_batch = next(data_iterator)  

        """
        Obtención de los ouputs
        """
        output_batch = model(data_batch)

        """
        Función de pérdida en este batch
        """
        loss = loss_fn(output_batch, labels_batch) 

        """
        Se obtienen los datos outputs de las variables de antes, los copiamos en la cpu y los convertimos en arrays 
        """
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        """Resumen""" 
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    """Media de las métricas obtenidas en la variable resumen 'summ' """
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean 


if __name__ == '__main__':

    args = parser.parse_args()
    """
    Establecimiento del camino (path)
    """
    json_path = os.path.join(args.model_dir, 'params.json')

    """
    Con la 'keyword' assert nos asegurames de que el archivo este en el path indicado, si no 
    saltará un error tipo AssertionError
    """
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    
    """
    Cargamos los parámetros con ayuda de la clase Params que definimos en el archivo utils.py
    """
    params = utils.Params(json_path)

    """
    Con este le pedimos que mejor el rendimiento mediante la GPU si es posible 
    """
    params.cuda = torch.cuda.is_available()     

    torch.manual_seed(230) """ Semilla para replicar el experimento con los mismos resultados"""
    if params.cuda: torch.cuda.manual_seed(230)
        
    """
    Llamamos a la función que establece los registros
    """
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log')) 

    """
    Registro de nivel info que nos indica que se está creando el dataset
    """
    logging.info("Creating the dataset...")

    """
    Carga de los datos de test
    """
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['test'], args.data_dir)
    test_data = data['test']

    params.test_size = test_data['size']
    test_data_iterator = data_loader.data_iterator(test_data, params)

    logging.info("- done.")

    """
    Creamos el modelo con la clase Net
    """
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    """
    Variables definidas de función de pérdida y métrica de tasa de acierto
    """
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    """
    Se indica en el registro que la evaluación comienza
    """
    logging.info("Starting evaluation")

    """
    Cargamos los pesos del archivo con la información del modelo guardada
    """
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    """
    Evaluación: se indica el número de pasos, se usa el método de evaluación 'evaluate', guardamos
    la tasa de acierto en el conjunto de test en un diccionario en el path 'save_path'
    """
    num_steps = (params.test_size + 1) // params.batch_size
    test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
