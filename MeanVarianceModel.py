from UQModel import UQModel

class MeanVarianceModel(UQModel):

    def __init__(self, name, params):
        super().__init__(name, params)
