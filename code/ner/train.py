"""
## Minería de textos
Universidad de Alicante, curso 2021-2022

Esta documentación forma parte de la práctica "[Lectura y documentación de un sistema de
extracción de entidades](https://jaspock.github.io/mtextos/bloque2_practica.html)" y se 
basa en el código del curso [CS230](https://github.com/cs230-stanford/cs230-code-examples) 
de la Universidad de Stanford.

**Autores de los comentarios:** Gilbert Lurduy & Enrique Moreno

Este módulo se dedica a entrenar el modelo de extracción de entidades.
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

import utils
import model.net as net
from model.data_loader import DataLoader
from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):

    """
    ### Función `train`
    Entrena el modelo `model` con `num_steps` batches.

    #### Parámetros:
    * `model`: red neuronal
    * `optimizer`: optimizador para los parámetros del modelo
    * `loss_fn`: función de pérdida (evalúa batch a batch)
    * `data_iterator`: generador de batches de datos y etiquetas
    * `metrics`: diccionario de funciones que calculan una medida entre los output y las verdaderas etiquetas para cada batch
    * `params`: parámetros de entrenamiento
    * `num_steps`: cantidad de batches sobre los que entrenar
    """
    
    # Avisar al modelo de que se va a usar para entrenar
    model.train()

    # Lista para almacenar resumen de cada iteración del entrenamiento  
    summ = []
    # Media móvil
    loss_avg = utils.RunningAverage()

    # Mostrar barra de progreso
    t = trange(num_steps)

    # Iterar sobre cada batch
    for i in t:
        # Obtener el siguiente batch de entrenamiento
        train_batch, labels_batch = next(data_iterator)

        # Calcular el output del modelo y la función de pérdida
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)

        """
        Para cada batch en el proceso de entrenamiento queremos poner los gradientes a 0 antes de comenzar backpropagation,
        ya que PyTorch acumula los gradientes en subsiguientes pasadas. Si no se hace esto, el gradiente actual será una
        combinación del viejo y del actual, y por ello apuntaría en una dirección distinta a la del mínimo.
        """

        # Resetear los gradientes a 0 y realizar backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Actualizar los parámetros con los gradientes calculados
        optimizer.step()

        # Obtener resúmenes cada **save_summary_steps** iteraciones
        if i % params.save_summary_steps == 0:
            # Extraer el contenido del output y las etiquetas, pasarlos a la memoria CPU 
            # y convertirlos en arrays de NumPy
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()
            
            # Guardar medidas para este batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
        
        # Actualizar la media de pérdida actual
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
    
    # Calcular la media de todas las medidas para cada batch guardado
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):

    """
    ### Función `train_and_evaluate`
    Entrena el modelo `model` y evalúa cada epoch.

    #### Parámetros:
    * `model`: red neuronal
    * `train_data`: conjunto de entrenamiento con datos y etiquetas
    * `val_data`: conjunto de validación con datos y etiquetas
    * `optimizer`: optimizador para los parámetros del modelo
    * `loss_fn`: función de pérdida (evalúa por batch)
    * `metrics`: diccionario de funciones que calculan una medida entre los output y las verdaderas etiquetas para cada batch
    * `params`: parámetros de entrenamiento
    * `model_dir`: directorio conteniendo información sobre el modelo
    * `restore_file`: nombre del archivo desde el cual cargar un checkpoint del modelo
    """

    # Cargar checkpoint del modelo desde restore_file si así ha sido indicado
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    # Iterar **num_epochs** veces
    for epoch in range(params.num_epochs):

        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # Calcular cantidad de batches de entrenamiento en un epoch
        num_steps = (params.train_size + 1) // params.batch_size
        # Obtener generador de datos y etiquetas de las frases de entrenamiento
        train_data_iterator = data_loader.data_iterator(
            train_data, params, shuffle=True)
        # Entrenar el modelo
        train(model, optimizer, loss_fn, train_data_iterator,
              metrics, params, num_steps)

        # Calcular cantidad de batches de validación en un epoch
        num_steps = (params.val_size + 1) // params.batch_size
        # Obtener generador de datos y etiquetas de las frases de validación
        val_data_iterator = data_loader.data_iterator(
            val_data, params, shuffle=False)
        # Evaluar el modelo ya entrenado con el conjunto de validación
        val_metrics = evaluate(
            model, loss_fn, val_data_iterator, metrics, params, num_steps)

        # Determinar si en el epoch actual se ha obtenido el mayor accuracy
        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Guardar pesos del estado actual
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # Si el epoch actual ha obtenido el mayor accuracy, se guardan las medidas actuales
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Guardar las medidas actuales como las últimas registradas
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Cargar los parámetros
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Usar GPU si hay disponibilidad
    params.cuda = torch.cuda.is_available()

    # Establecer semilla para poder reproducir experimentos
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Preparar el logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading the datasets...")

    # Cargar los datos de entrenamiento y validación
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['train', 'val'], args.data_dir)
    train_data = data['train']
    val_data = data['val']

    # Tamaño de los conjuntos de datos
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")

    # Usar el modelo en la GPU si está disponible
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # Obtener función de perdida y medidas
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Entrenar el modelo
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
