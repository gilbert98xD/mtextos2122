import json
import logging
import os
import shutil

import torch


# ESTO EN MAYÚSCULASSS
# perimte leer un fichero de parametros, leemos un ficheor json que dice {"a":1,"b":2}
# la idea es convertir esto a un objeto de javascript para luego hacer x.a y x.b para que
# nos de 1 y 2

class Params():
    """
    Carga los parámetros contenidos en un archivo tipo json
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f) # carga del fichero json_path del se obtiene un objeto json
            self.__dict__.update(params) # actualización del diccionario built-in de atributos del objeto
            

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            # escribe/guarda el diccionario actualizado en el archhivo json_path
            # con una identación de 4 espacios

    def update(self, json_path): # método que hace la misma función que el constructor
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property # decorador
    def dict(self): 
        return self.__dict__ # nos devuelve el diccionario de atributos del objeto creado con esta clase



class RunningAverage():
    """
    Obtención de la cantidad 'running average' de cualquier variable
    """

    def __init__(self): # el constructor inicializa los pasos
        self.steps = 0 
        self.total = 0

    def update(self, val): 
        self.total += val # actualizaciones del valor total y de los pasos
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps) # obtención de la media de los valores final




def set_logger(log_path):
    """
    ESTABLECE DOS LOGGERS PARA IR ENVIANDO LA INFORACION,POR UN LADO FICHEORS Y LUEGO FORMA ESTÁNDARD
    """

    logger = logging.getLogger() # se crea un objeto logger
    logger.setLevel(logging.INFO) # se establece el nivel del logger. Se guardaran los logs
    # de tipo INFO o superior (debug,notset)

    if not logger.handlers: # si no hay un handler se creará uno; éste
    # especifica a qué tipo de archivo se envía el registro (log)

        file_handler = logging.FileHandler(log_path) # handler que guarda el registro
        # en archivos del disco
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler() # guarda el registro en consola
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        # la línea de código inmediatamente superior sólo especifica el formato
        # del log 
        logger.addHandler(stream_handler) # adición del stream handler al logger


def save_dict_to_json(d, json_path):
    """
    Se guarda el diccionario d en formato json y sus valores se pasan a floats
    """
 
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()} # conversión a floats de los items de d
        json.dump(d, f, indent=4) # se guarda d en f (json_path)


# UN CHECKPOIINT ES UN FICEHRO QUE CONITNEE UN MODELO DE PYTORCH
# LOS PARAMETROS DE EMBEDDING, NUMERO CAPAS, SDG

# LOS CHECKPOINTS SON DICCIONARIOS CON TODOS LOS VALORES REALMENTE
# ESE DICCIONARIO ES STATE
# LO GRABA EN UN FORMATO BINARIO? GUARDARLO COMO TEXTO SERÍA MUY PESADO
# EMPAQUETA EN BINARIO Y LO COMPRIME


def save_checkpoint(state, is_best, checkpoint):
    """
    Guarda los 'checkpoints' (fichero con todos los parámetros del modelo pytorch)

    - state: diccionario del modelo
    - is_best:  booleano que determina si el modelo es el mejor que se ha obtenido hasta entonces
    - checkpoint: string indicando dónde guardar
    """

    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint): # si no existe el directorio checkpoint, entonces lo crea
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint) # creación del directorio
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath) # guarda las características del modelo en 'filepath'
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar')) 
        # en el caso de que el modelo sea el mejor, se copia su
        #  checkpoint grabado de filepath en 'best.pth.tar'



def load_checkpoint(checkpoint, model, optimizer=None):
   
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint) # ES UNDICIONARI, STATE-:DCIT TIENE LOS PESOS DE LA RED NEURONAL
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer: # OPTIMIZADOR DEL LEARNING RATE?
        optimizer.load_state_dict(checkpoint['optim_dict']) # GUARDA

    return checkpoint
    