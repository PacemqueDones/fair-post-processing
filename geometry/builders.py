from .geometry import FairGeometry


class FairGeometryBuilder:
    def __init__(
        self,
        distance_builder,
        laplacian_builder,
    ):
        self.distance_builder = distance_builder
        self.laplacian_builder = laplacian_builder

    def build(self, X):
        D_X = self.distance_builder.build(X)

        W, L = self.laplacian_builder.build(X)

        return FairGeometry(
            D_X=D_X,
            W=W,
            L=L
        )