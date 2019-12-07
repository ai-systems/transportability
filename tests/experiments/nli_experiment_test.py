from regra.common.regra_unit_test import RegraTestCase
from transport.experiments.trainer.nli_trainer import NLIExperiment
import luigi


class SNLIExperimentTest(RegraTestCase):
    def test_snli(self):
        task = NLIExperiment(mode=self.mode, config_file=self.config_file)
        luigi.build([task])
