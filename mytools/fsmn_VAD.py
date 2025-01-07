from funasr import AutoModel
import numpy as np


model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")


def Fsmn_VAD(wpath,outPath,model):

    print('[INFO]: Start computing VAD...')
    vad_time = model.generate(input=wpath)[0]

    np.savetxt(outPath,vad_time['value'],fmt='%d')
    return vad_time['value']

