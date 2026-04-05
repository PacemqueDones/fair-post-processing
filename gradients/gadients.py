class GradientDiagnostics:
    def __init__(self, objectives):
        self.objectives = objectives

    def collect(self, losses, params):
        ...
        return {
            "grad_norms": grad_norms,
            "grads_per_obj": grads_per_obj,
            "cosine_similarity": cosine_dict
        }