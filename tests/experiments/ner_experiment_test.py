from regra.common.regra_unit_test import RegraTestCase
from transport.experiments.ner.ner_experiment import NerExperiment
import luigi


class NERExperimentTest(RegraTestCase):
    def ner_experiment_test(self):
        task = NerExperiment(mode=self.mode, config_file=self.config_file)
        luigi.build([task], local_scheduler=True)
