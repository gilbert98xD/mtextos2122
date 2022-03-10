import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# AQUI DEFINIMOS UNA RED NEURONAL

##NN.MODULE ES U MODELO NEURONAL BASICO, LUEGO PUEDO PARTICULARIZAR O HEREDAR DE EL
## NN.MODULE ES UNA CLASE BASE?

class Net(nn.Module):

    def __init__(self, params): # EL CONSTRUCTOR, CUANDO CREAMOS UN NUEVO OBEJTO DE LA CLASE SE LLAMA?
    # DEFINE LAS CAPAS DE TU RED NEURONAL

        super(Net, self).__init__() # LLAMA AL CONSTRUCTOR DEL PADRE
        # SE LE PIDE AL PDRE QUE CONSTRUYA SU CLASE Y LUEGO YA EL RESTO DE ESTA CLASE

        # COMPONENTES QUE TNEDRÁ LA RED NUERONAL, 
        # MATRIZ DE EMBEDDINGS

        # SE LE DA EL TAMAÑO, EL TAMAÑO DEL VOCABULARIO, PARAMS.VOCAB_SIZE
        # LAS DIMENSIONES LAS DECIDO YO, 

        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        #


        # DE ABAJO LE LLEGA 
        # PROYECTA .. A OTR ADIMENSION PARAMS.LSTM_HIDDEN_DIM

        # SE PROYECTA A UNNUEVO PUNTO DEL ESPACIO
        # EL DE ENTRADA, Y EL PROYECTADO (CONTEXTUAL) NO TIENEN PORQUE TENER EL MISMO TAMAÑO, 
        # OPR ESO SE PONE LAS DIMENSIONES

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
