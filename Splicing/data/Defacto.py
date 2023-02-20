# -*- coding: utf-8 -*-
"""
defacto-splicing Dataset utills 
데이터세트 정의 클래스, 데이터세트 불러오기 함수 구현
_crop_size = (256,256)
_grid_crop = True
_blocks = ('RGB', 'DCTvol', 'qtable')
tamp_list = None
DCT_channels = 1
"""
import os,sys
from .AbstractDataset import AbstractDataset
import torch
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
from torch.utils.data import DataLoader,random_split
import PIL.Image as Image
import cv2 as cv
import shutil as sh
import random

def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
fix_seed(42)

# https://www.kaggle.com/code/alerium/defacto-test
class DefactoDataset(AbstractDataset):
    def __init__(self, im_root_dir,label_root_dir, num, img_size, mode='test' , transform=None, 
                # crop_size=(512,512), grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1 \
                    crop_size=(512,512), grid_crop=True, blocks=('RGB','DCTvol', 'qtable'), DCT_channels=1 \
                    ):
        
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self.im_root_dir = im_root_dir
        self.label_root_dir = label_root_dir
        self.transform = transform
        self.num = num
        self.mode = mode
        self.img_size = img_size
        self.df = pd.DataFrame({
            'image_path':[os.path.join(self.im_root_dir,_) for _ in sorted(os.listdir(self.im_root_dir))[:self.num]],
            'mask_path':[os.path.join(self.label_root_dir,_) for _ in sorted(os.listdir(self.label_root_dir))[:self.num]]
                         })
        # self.df = pd.DataFrame({
        #     'image_path':[_ for _ in sorted(os.listdir(self.im_root_dir))],
        #     'mask_path':[_ for _ in sorted(os.listdir(self.label_root_dir))]
        #                  })

        self.name = self.df['image_path'].iloc[:self.num]
        self.label = self.df['mask_path'].iloc[:self.num] 

    def __len__(self):
        return len(self.name)

    def _resize(self , sample):
        image, mask = sample[0], sample[1]
        n = self.img_size
        image = TF.resize(image , size=(n,n),interpolation=transforms.InterpolationMode.BICUBIC)
        mask = TF.resize(mask , size=(n,n) ,interpolation=transforms.InterpolationMode.BICUBIC)

        return image , mask
    
    def __getitem__(self,idx):
        mask = np.array(Image.open(self.df['mask_path'].iloc[idx]).convert("L"))
        mask[mask > 0] = 1
        # x,y,z = self._create_tensor(self.df['image_path'].iloc[idx], mask)
        # image = x[:3]

        return self._create_tensor(self.df['image_path'].iloc[idx], mask)#{'artifact':x, 'landmarks': y,'qtable': z}

    
    def _getitem_test(self, idx):
        image = torch.FloatTensor(cv.imread(self.name[idx]))
        label = torch.FloatTensor(cv.imread(self.label[idx],cv.IMREAD_GRAYSCALE))

        if self.transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size,self.img_size),interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
            ])

        if self.mode in ['train','eval']:
            x,y  = self._resize((image.permute(2,0,1), label.unsqueeze(0))) # (C,W,H)로 마춰줘야 함
            x = self.transform(x)
            y = y.ge(0.5).float()
        else: # test : 네트워크에 통과되지 않고 바로 plot 가능한 image 반환
            x = image.permute(2,0,1)
            y = label.unsqueeze(0)/255.0
            x,y = self._resize((x/255.0,y))
            y = y.ge(0.5).float() # element-wise로 값을 비교해 크거나 같으면 True를, 작으면 False를 반환한다.
            y = y.permute(1,2,0)

        if True :# label == 'forgery'
            label = torch.tensor(1,dtype=torch.long)
        else:
            label[0] = 1.0

        return {'image': x, 'landmarks': y,'label':label}
        
    def get_info(self):
        s = f"crop_size={self._crop_size}, grid_crop={self._grid_crop}, blocks={self._blocks}, mode={self.mode}\n"
        return s

def load_dataset(total_nums,img_size,batch_size,dir_img,dir_mask):
    # ImageNet 표준으로 정규화 <- 일반적인 관행

    dataset = DefactoDataset(dir_img,
                dir_mask,
                total_nums,
                img_size,
                'train',None)

    # dataset_size = len(dataset)
    train_size =  10000 #int(dataset_size * 0.8)
    validation_size = 764 #int(dataset_size * 0.2)

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print("train images len : ",train_dataset.__len__())
    print("validation images len : ",validation_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    
    return train_dataloader,val_dataloader

def test(model,device,index,mode,img_size,dir_img,dir_mask):
    model.eval()
    # transformi = transforms.Compose([
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
                # ])
    testdata = DefactoDataset(dir_img,
               dir_mask,
               3000,
               img_size,
               mode,None)

    data =testdata.__getitem__(index) 
    img = data['image']
    mask = data['landmarks']


    inp = torch.Tensor([img.numpy()]).to(device)
    with torch.no_grad():
        pred = model(inp) # normalize none

    return pred, img, mask

def test_dct(model,device,index,mode,img_size,dir_img,dir_mask):
    model.eval()
    testdata = DefactoDataset(dir_img,
               dir_mask,
               3000,
               img_size,
               mode,None)

    data =  testdata.__getitem__(index)
    jpg_artifact =data['artifact']
    mask = data['landmarks']
    qtable = data['qtable']
    img = data['image']

    inp = jpg_artifact.unsqueeze(0).to(device)
    qtable = qtable.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp,qtable) # normalize none

    

    return pred, img, mask

def load_dataset_for_casia(total_nums,img_size,batch_size,dir_img,dir_mask):
    # ImageNet 표준으로 정규화 <- 일반적인 관행

    dataset = DefactoDataset(dir_img,
                dir_mask,
                total_nums,
                img_size,
                'train',None)

    # dataset_size = len(dataset)
    train_size =  2300 #int(dataset_size * 0.8)
    validation_size = 200 #int(dataset_size * 0.2)
    test_size = 559
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size,test_size])
    print("train images len : ",train_dataset.__len__())
    print("validation images len : ",validation_dataset.__len__())
    print("test images len : ",test_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    return train_dataloader,val_dataloader,test_dataloader


# tif to jpg in tif dir
if __name__ == '__main__':
    tif_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\Tp"
    out_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\Tp_jpg"
    mask_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\Groundtruth"
    mask_out_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\Groundtruth_jpg"
    os.makedirs(out_dir,exist_ok=True)
    os.makedirs(mask_out_dir,exist_ok=True)

    source = os.listdir(tif_dir)
    l = len(source)
    # print('total iamges : ',l)

    # for i,infile in enumerate(source,1):
    #     try:
    #         if infile[-3:] == "tif":
    #             outfile = infile[:-3] + "jpg"
    #             im = Image.open(os.path.join(tif_dir,infile))
    #             out = im.convert("RGB")
    #             out.save(os.path.join(out_dir,outfile), "JPEG", quality=100)
    #         if  i %100 ==0: 
    #             print(f"{i} images processed. {i/l:.3}  done. ")
    #     except:
    #         print(i)
       

    print(len(os.listdir(tif_dir)))
    print(len(os.listdir(mask_dir)))
    """
    10765
    10765
    """
    images = sorted(os.listdir(tif_dir))
    
    maskes = sorted(os.listdir(mask_dir))

    for i,data in enumerate(zip(images,maskes),1):
        infile,mask = data
        try:
            if infile[-3:] == "tif":
                # image
                outfile = infile[:-3] + "jpg"
                im = Image.open(os.path.join(tif_dir,infile))
                out = im.convert("RGB")
                out.save(os.path.join(out_dir,outfile), "JPEG", quality=100)

                # mask
                outfile = mask[:-3] + "jpg"
                im = Image.open(os.path.join(mask_dir,mask))
                out = im.convert("L")
                out.save(os.path.join(mask_out_dir,outfile), "JPEG", quality=100)
            if  i %100 ==0: 
                print(f"{i} images processed. {i/l:.3}  done. ")
        except:
            print(i)

    mask_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\Groundtruth_jpg"
    out_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\TP_jpg"

    testdata = DefactoDataset(out_dir,
            mask_dir,
            -1,
            (512,512),
            "tr0in",None)

    def match_test():
        name = testdata.name
        label = testdata.label
        for data in zip(name,label):
            n,l = data   
            if not (n[:-4] in l):
                print("doesn't match pare")
                print(n)
                print(l)
                return 0
        return 1
    print(match_test())
    # def mask_move(isMatch:bool,mask_dir,new_mask_path):
    #     os.makedirs( new_mask_path,exist_ok=True)
    #     for img_name in os.listdir(out_dir):
    #         sh.move(os.path.join(mask_dir,img_name),
    #         os.path.join(new_mask_path,img_name))


    # # mask_move(-1,mask_dir,new_mask_path)
    # with open("./CASIA_list.txt",'r') as f:
    #     text = f.read()
    # text = text.replace(',png','')
    # with open("./CASIA_list2.txt",'w') as f:
    #     f.write(text)
