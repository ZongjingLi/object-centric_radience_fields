from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from moic.utils import load_json
from PIL import Image
import numpy as np
import os

class Clevr4(Dataset):
    def __init__(self,split = "train"):
        super().__init__()
        assert split in ["train","test","validate"],print("Unknown dataset split type: {}".format(split))

    def len(self):return 1

    def __getitem__(self,index):
        return index

class Sprite3(Dataset):
    def __init__(self,split = "train"):
        super().__init__()
        assert split in ["train","test","validate"],print("Unknown dataset split type: {}".format(split))
    
        self.split = split
        self.root_dir = "datasets/sprites3"
        self.files = os.listdir(os.path.join(self.root_dir,split))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.questions = load_json("datasets/sprites3/train_sprite3_qa.json")
        
    def __len__(self): return len(self.files)

    def __getitem__(self,index):
        qa_sample = self.questions[index]
        # open the image of the file
        idx = np.random.choice(range(len(qa_sample)))
        
        image = Image.open(os.path.join(self.root_dir,self.split,"{}_{}.png".format(self.split,qa_sample[idx]["image"])))
        image = image.convert("RGB").resize([64,64])
        image = self.img_transform(image)

        
        sample = {"image":image,
                "question":qa_sample[idx]["question"],
                "program" :qa_sample[idx]["program"],
                "answer"  :qa_sample[idx]["answer"],}
        return sample