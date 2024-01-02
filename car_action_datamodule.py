from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import car_action_dataset
import utils
from data_processor import DataProcessor


class CarActionDataModule(LightningDataModule):
    """Datamodule for car action dataset.

    
    """
    def __init__(self,training_path,test_path):#),training_data:dict, valid_data:dict, test_data:dict):
        """Init function for car action datamodule

        Args:
            training_data (List[tuple]): List of tuple (image,label)
            valid_data (List[tuple]): List of tuple (image,label)
            test_data (List[tuple]): List of tuple (image,label)
        """
        super().__init__()
        
        #self.training_data = training_data
        #self.valid_data = valid_data
        #self.test_data = test_data
        self.training_path = training_path
        self.test_path = test_path

    def setup(self, stage: str):
        data_processor = DataProcessor(self.training_path,self.test_path,0.3,0)
        if stage == "fit":
            self.train_dataset = car_action_dataset.CarActionDataset(list(zip(data_processor.x_train,data_processor.y_train)))
            self.valid_dataset = car_action_dataset.CarActionDataset(list(zip(data_processor.x_eval,data_processor.y_eval)))

        if stage == "validate":
            self.valid_dataset = car_action_dataset.CarActionDataset(list(zip(data_processor.x_eval,data_processor.y_eval)))
        
        if stage == "test":
            self.test_dataset = car_action_dataset.CarActionDataset(list(zip(data_processor.test_samples,data_processor.test_labels)))

    def train_dataloader(self):
        
        return DataLoader(
            self.train_dataset,
            batch_size = utils.BATCH_SIZE,
            num_workers = utils.NUM_WORKERS,
            shuffle = True,
            #collate_fn=utils.collate_fn
        ) 
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size = utils.BATCH_SIZE,
            num_workers = utils.NUM_WORKERS,
            shuffle = False,
            #collate_fn=utils.collate_fn
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = utils.BATCH_SIZE,
            num_workers = utils.NUM_WORKERS,
            shuffle = False,
            #collate_fn=utils.collate_fn
        )