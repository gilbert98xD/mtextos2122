import json
import logging
import os
import shutil

import torch

"""
Creación de una clase que, a paritir de un archivo tipo json, 
carga los parámetros 
"""
# ESTO EN MAYÚSCULASSS
# perimte leer un fichero de parametros, leemos un ficheor json que dice {"a":1,"b":2}
# la idea es convertir esto a un objeto de javascript para luego hacer x.a y x.b para que
# nos de 1 y 2

class Params():

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params) # actualización del diccionario
            # cualquier OBJETO DE JAVASCRIPT TIEEN UN OBJECTO __dict__ DICCIONARIO DE ATRIBUTOS DEL OBJETO
            # LE PASAMOS UNOS ATRIBUTOS params y LO QUE HACE ES ME...?

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            # estas dos líneas de código guardan el diccionario en formato json
            # la identación es de 4 espacios

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self): # ASI SE PUEDE METER AUTOMICAITMENTE A TODOS LOS ATRIBUTOS?
        return self.__dict__


"""
Clase para tener el 'running average' de cualquier variable
"""

# LE AÑADES VALAORES Y EN CUALQUIE RMOMENTO NOS PUEDE DAR LA MEDIA DE LOS VALORES AÑADIDOS

class RunningAverage():

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val # actualizaciones del valor total y de los pasos
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)



# ESTABLECE DOS LOGGERS PARA IR ENVIANDO LA INFORACION,POR UN LADO FICHEORS Y LUEGO FORMA ESTÁNDARD
# 


def set_logger(log_path):

    logger = logging.getLogger() # se crea un objeto logger
    logger.setLevel(logging.INFO) # se establece el nivel del logger. Se guardaran los logs
    # de tipo INFO o superior (debug,notset)

    if not logger.handlers: # si no hay un handler, creamos un uno, este
    # espeicifica a que tipo de archivo se envia el registro(log)


        file_handler = logging.FileHandler(log_path) # handler que guarda el registro
        # en archivos del disco
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler() # guarda el registro en consola
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        # la línea de código inmediatamente superior sólo especifica el formato
        # del log 
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """
    Se guarda el diccionario d en formato json y sus valores se pasan a floats
    """
 
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()} # conversión a floats de los items de d
        json.dump(d, f, indent=4)


# UN CHECKPOIINT ES UN FICEHRO QUE CONITNEE UN MODELO DE PYTORCH
# LOS PARAMETROS DE EMBEDDING, NUMERO CAPAS, SDG

# LOS CHECKPOINTS SON DICCIONARIOS CON TODOS LOS VALORES REALMENTE
# ESE DICCIONARIO ES STATE
# LO GRABA EN UN FORMATO BINARIO? GUARDARLO COMO TEXTO SERÍA MUY PESADO
# EMPAQUETA EN BINARIO Y LO COMPRIME


def save_checkpoint(state, is_best, checkpoint):
    
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        # COPIA EL CHECKPOINT QUE YA HABÍOAS GRTABADO, COPIALO EN BEST, 
        # PORQUE PRA CREAR EL BINARIO COMPRIMRI OTRA VEZ PUES TARDA MÁS


# SE SUELE GUARDAR EL ULTMI MODELO YEL MEJOR


#
def load_checkpoint(checkpoint, model, optimizer=None):
   
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint) # ES UNDICIONARI, STATE-:DCIT TIENE LOS PESOS DE LA RED NEURONAL
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer: # OPTIMIZADOR DEL LEARNING RATE?
        optimizer.load_state_dict(checkpoint['optim_dict']) # GUARDA

    return checkpoint
    