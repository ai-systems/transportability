from regra.common.regra_unit_test import RegraTestCase
from transport.datasets.snli import SnliDataset
import luigi
from torch.utils.data import DataLoader


class SNLIDatasetTest(RegraTestCase):
    def test_snli(self):
        task = SnliDataset(mode=self.mode, config_file=self.config_file)
        luigi.build([task])
        torch_dataset = task.get_dataset()

        params = {
            'shuffle': True,
            'batch_size': 8,
            'num_workers': 8
        }

        dataloader = DataLoader(torch_dataset, **params)
        for data in dataloader:
            print(data)
