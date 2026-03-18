from fairpp.postprocessor import FairPostProcessor
from fairpp.model import ThresholdModel
from fairpp.objectives import CrossEntropyObjective
from fairpp.selectors import TopsisSelector

model = ThresholdModel(num_classes=2)

post = FairPostProcessor(
    model=model,
    objectives=[CrossEntropyObjective()],
    selector=TopsisSelector()
)

post.fit(probs_train, y_train, sensitive_attr)

preds = post.predict(probs_test)