from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pickle
import torch
from torch.utils.data import random_split
import random
from random import sample
from tqdm import tqdm
import numpy as np
from ddecomposition.decomposition import DynamicDecomposition
import torch.nn as nn
from main_model import CSDI_Physio
from ddecomposition.preprocess import getData
import os

class TrainData(Dataset):

    def __init__(self, file_path, test_path, window_length=100,split=4,mask_ratio=0.5, device='cuda:0'):   
        self.device = device
                
        dataset_name = os.path.basename(file_path).split('_')[0]  
        data_dir = os.path.dirname(file_path) + '/'
        
        print(f"加载数据集: {dataset_name} 从路径: {data_dir}")
        
        try:
            processed_data = getData(
                path=data_dir,
                dataset=dataset_name,
                period=1440,  
                train_rate=1  
            )
            
            self.train_data = processed_data['train_data']
            self.train_time = processed_data['train_time']
            self.train_stable = processed_data['train_stable']
            print(f"成功使用D3R预处理: train_data shape: {self.train_data.shape}")
            print(f"成功使用D3R预处理: train_time shape: {self.train_time.shape}")
            print(f"成功使用D3R预处理: train_time dims: {self.train_time.ndim}")
            
        except Exception as e:
            print(f"D3R预处理失败: {e}")
            print("回退到IMDiffusion原始加载逻辑...")
            
            self.data = np.load(file_path)
            length = self.data.shape[0]
            self.test_data = np.load(test_path)
            self.data = np.concatenate([self.data, self.test_data])
            self.data = torch.Tensor(self.data)
            self.data = self.data[:length, :] * 20
            
          
            self.train_data = self.data
            self.train_time = np.zeros((self.data.shape[0], 5))
            self.train_stable = self.data  
            print(f"回退到原始逻辑: train_time shape: {self.train_time.shape}")
            print(f"回退到原始逻辑: train_time dims: {self.train_time.ndim}")
        
        
        if not isinstance(self.train_data, torch.Tensor):
            self.train_data = torch.Tensor(self.train_data)
        if not isinstance(self.train_time, torch.Tensor):
            self.train_time = torch.Tensor(self.train_time)
        if not isinstance(self.train_stable, torch.Tensor):
            self.train_stable = torch.Tensor(self.train_stable)
    
        self.window_length = window_length
        self.begin_indexes = list(range(0, len(self.train_data) - window_length))
        self.split = split
        self.mask_ratio = mask_ratio
    
        feature_dim = self.train_data.shape[1]
        time_dim = self.train_time.shape[1]
        
        self.decomposition = DynamicDecomposition(
            window_size=window_length,
            model_dim=512,
            ff_dim=2048,
            atten_dim=64,
            feature_num=feature_dim,
            time_num=time_dim,
            block_num=2,
            head_num=8,
            dropout=0.6,
            d=30
        ).to(device)

  
        model_path = f"decomposer_results/{dataset_name}/best_decomposer.pth"
        try:
            self.decomposition.load_state_dict(torch.load(model_path))
            print(f"成功加载预训练分解模型: {model_path}")
        except Exception as e:
            print(f"加载预训练模型失败: {e}，使用未训练模型")
        
        # 分解结果缓存
        self.decomposition_cache = {}
        print("TrainData初始化完成")


    def get_mask(self, observed_data, observed_mask, strategy_type):
        mask = torch.zeros_like(observed_mask)
        length = observed_mask.shape[0]
        skip = length // self.split
        
    
        segment_features = []
        segments_data = []
        segment_indices = []  # 存储每段的索引信息
        
       
        for split_index, begin_index in enumerate(list(range(0, length, skip))):
            end_index = min(begin_index + skip, length)
            segment_data = observed_data[begin_index:end_index]
            segments_data.append(segment_data)
            segment_indices.append((split_index, begin_index, end_index))

            var_score = torch.var(segment_data, dim=0).sum()
            

            gradient_score = torch.mean(torch.abs(torch.diff(segment_data, dim=0)))
            
            if split_index > 0:
                prev_segment = segments_data[-2]
 
                continuity = torch.mean(torch.abs(segment_data[0] - prev_segment[-1]))
                prev_trend = torch.mean(torch.diff(prev_segment, dim=0))
                curr_trend = torch.mean(torch.diff(segment_data, dim=0))
                trend_continuity = torch.abs(curr_trend - prev_trend)
                continuity_score = continuity + trend_continuity
            else:
                continuity_score = torch.tensor(0.0)
            
         
            score = var_score + gradient_score + continuity_score
            segment_features.append(score)
        
        segment_features = torch.stack(segment_features)
        sorted_scores, sorted_indices = torch.sort(segment_features)
        
        
        for rank, idx in enumerate(sorted_indices):
            split_index, begin_index, end_index = segment_indices[idx]
           
        total_segments = len(segment_features)
        lowest_30_percent = int(total_segments * 0.3) 
        
    
        selected_low_indices = sorted_indices[:lowest_30_percent]
        
        if strategy_type == 0:
            selected_indices = selected_low_indices

            for idx in selected_indices:
                split_index, begin_index, end_index = segment_indices[idx.item()]
                mask[begin_index:end_index, :] = 1
                
        else:
            selected_indices = selected_low_indices
            
            for idx in selected_indices:
                split_index, begin_index, end_index = segment_indices[idx.item()]
                mask[begin_index:end_index, :] = 1

        return mask
    

    def __len__(self):
        return len(self.begin_indexes)

    def __getitem__(self, item):
        if random.random() < 0.5:
            strategy_type = 0
        else:
            strategy_type = 1
        
        data_window = self.train_data[
            self.begin_indexes[item]:
            self.begin_indexes[item] + self.window_length
        ]
        time_window = self.train_time[
            self.begin_indexes[item]:
            self.begin_indexes[item] + self.window_length
        ]
        
        cache_key = self.begin_indexes[item]
        if cache_key not in self.decomposition_cache:
            batch_data = data_window.unsqueeze(0).to(self.device)
            batch_time = time_window.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                try:
                    stable, trend = self.decomposition(batch_data, batch_time)
                    stable = stable.squeeze(0).cpu()
                    trend = trend.squeeze(0).cpu()
                except Exception as e:
                    print(f"D3R分解出错: {e}")
                    stable = data_window
                    trend = torch.zeros_like(data_window)
            
            # 存入缓存
            self.decomposition_cache[cache_key] = (stable, trend)
        else:
            stable, trend = self.decomposition_cache[cache_key]
        
        preprocess_stable = self.train_stable[
            self.begin_indexes[item]:
            self.begin_indexes[item] + self.window_length
        ]
        
        observed_mask = torch.ones_like(data_window - trend)
        gt_mask = self.get_mask(data_window - trend, observed_mask, strategy_type)
        timepoints = np.arange(self.window_length)
        
        return {
            "observed_data": data_window - trend,
            "original_data": data_window,
            "trend": trend,
            "preprocess_stable": preprocess_stable,
            "stable": stable,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": timepoints,
            "strategy_type": strategy_type,
            "timestamp_features": time_window
        }

class TestData(Dataset):

    def __init__(self, file_path, label_path, train_path, window_length=100, 
                get_label=False, window_split=1, strategy=1, split=4, mask_list=[], device='cuda:0'):
        self.strategy = strategy
        self.get_label = get_label
        self.mask_list = mask_list
        self.device = device
        
        from ddecomposition.preprocess import getData
        import os
        
        dataset_name = os.path.basename(file_path).split('_')[0]
        data_dir = os.path.dirname(file_path) + '/'
        
        print(f"加载测试数据集: {dataset_name} 从路径: {data_dir}")
        
        try:
            processed_data = getData(
                path=data_dir,
                dataset=dataset_name,
                period=1440,
                train_rate=1
            )
            
            self.test_data = processed_data['test_data']
            self.test_time = processed_data['test_time']
            self.test_stable = processed_data['test_stable']
            self.test_label = processed_data['test_label']
           
        except Exception as e:
            print(f"D3R测试数据预处理失败: {e}")
            print("回退到IMDiffusion原始测试数据加载逻辑...")
            
            self.data = np.load(file_path)
            length = self.data.shape[0]
            try:
                self.train_data = np.load(train_path)
            except:
                print("train data get wrong!")
                self.train_data = np.zeros((1, self.data.shape[1]))
            
            try:
                self.label = np.load(label_path)
            except:
                print("label get wrong!")
                self.label = np.zeros(self.data.shape[0])
            
            self.label = torch.LongTensor(self.label)
            self.data = np.concatenate([self.data, self.train_data])
            self.data = torch.Tensor(self.data)
            self.data = self.data[:length, :] * 20
            
            self.test_data = self.data
            self.test_time = np.zeros((self.data.shape[0], 5))
            self.test_stable = self.data
            self.test_label = self.label
            print(f"回退到原始测试逻辑: test_time shape: {self.test_time.shape}")
            print(f"回退到原始测试逻辑: test_time dims: {self.test_time.ndim}")
        
        # 转换为tensor
        if not isinstance(self.test_data, torch.Tensor):
            self.test_data = torch.Tensor(self.test_data)
        if not isinstance(self.test_time, torch.Tensor):
            self.test_time = torch.Tensor(self.test_time)
        if not isinstance(self.test_stable, torch.Tensor):
            self.test_stable = torch.Tensor(self.test_stable)
        if not isinstance(self.test_label, torch.Tensor):
            self.test_label = torch.LongTensor(self.test_label)
        
        self.window_length = window_length
        self.begin_indexes = list(range(0, len(self.test_data) - window_length, window_length // window_split))
        self.split = split
        
        from ddecomposition.decomposition import DynamicDecomposition
        feature_dim = self.test_data.shape[1]
        time_dim = self.test_time.shape[1]
        
        self.decomposition = DynamicDecomposition(
            window_size=window_length,
            model_dim=512,
            ff_dim=2048,
            atten_dim=64,
            feature_num=feature_dim,
            time_num=time_dim,
            block_num=2,
            head_num=8,
            dropout=0.6,
            d=30
        ).to(device)

        model_path = f"decomposer_results/{dataset_name}/best_decomposer.pth"
        try:
            self.decomposition.load_state_dict(torch.load(model_path))
            print(f"成功加载预训练分解模型: {model_path}")
        except Exception as e:
            print(f"加载预训练模型失败: {e}，使用未训练模型")
        
        # 分解结果缓存
        self.decomposition_cache = {}
        print("TestData初始化完成")

    def __len__(self):
        return len(self.begin_indexes)

    def get_mask(self, observed_data, observed_mask):
        mask = torch.zeros_like(observed_mask)
        length = observed_mask.shape[0]
        skip = length // self.split
        
        segment_features = []
        segments_data = []
        segment_indices = []
        
        for split_index, begin_index in enumerate(list(range(0, length, skip))):
            end_index = min(begin_index + skip, length)
            segment_data = observed_data[begin_index:end_index]
            segments_data.append(segment_data)
            segment_indices.append((split_index, begin_index, end_index))
            
            var_score = torch.var(segment_data, dim=0).sum()
            
            # 2. 计算梯度特征
            gradient_score = torch.mean(torch.abs(torch.diff(segment_data, dim=0)))
            
            # 3. 计算连续性特征
            if split_index > 0:
                prev_segment = segments_data[-2]
                # 计算与前一段的连续性
                continuity = -torch.mean(torch.abs(segment_data[0] - prev_segment[-1]))
                # 计算趋势的连续性
                prev_trend = torch.mean(torch.diff(prev_segment, dim=0))
                curr_trend = torch.mean(torch.diff(segment_data, dim=0))
                trend_continuity = -torch.abs(curr_trend - prev_trend)
                continuity_score = continuity + trend_continuity
            else:
                continuity_score = torch.tensor(0.0)
            
            score = var_score + gradient_score + continuity_score
            segment_features.append(score)
        
        segment_features = torch.stack(segment_features)
        sorted_scores, sorted_indices = torch.sort(segment_features)
        
     
        total_segments = len(segment_features)
        lowest_30_percent = int(total_segments * 0.3)  # 取最低30%的段
        
 
        selected_low_indices = sorted_indices[:lowest_30_percent]
        if self.strategy == 0:
        
            selected_indices = selected_low_indices
            
           
            for idx in selected_indices:
                split_index, begin_index, end_index = segment_indices[idx.item()]
                mask[begin_index:end_index, :] = 1
                
        else:
     
            selected_indices = selected_low_indices
            
         
            for idx in selected_indices:
                split_index, begin_index, end_index = segment_indices[idx.item()]
                mask[begin_index:end_index, :] = 1


        return mask

    def __getitem__(self, item):
        # 获取窗口数据
        data_window = self.test_data[
            self.begin_indexes[item]:
            self.begin_indexes[item] + self.window_length
        ]
        time_window = self.test_time[
            self.begin_indexes[item]:
            self.begin_indexes[item] + self.window_length
        ]
        

        label_window = self.test_label[
            self.begin_indexes[item]:
            self.begin_indexes[item] + self.window_length
        ]
        
        # 获取预处理的稳定分量
        preprocess_stable = self.test_stable[
            self.begin_indexes[item]:
            self.begin_indexes[item] + self.window_length
        ]
        
        
        cache_key = self.begin_indexes[item]
        if cache_key not in self.decomposition_cache:
            
            batch_data = data_window.unsqueeze(0).to(self.device)
            batch_time = time_window.unsqueeze(0).to(self.device)
            
            
            with torch.no_grad():
                try:
                    stable, trend = self.decomposition(batch_data, batch_time)
                    stable = stable.squeeze(0).cpu()
                    trend = trend.squeeze(0).cpu()
                except Exception as e:
                    print(f"测试数据D3R分解出错: {e}")
                    stable = data_window
                    trend = torch.zeros_like(data_window)
            
           
            self.decomposition_cache[cache_key] = (stable, trend)
        else:
            stable, trend = self.decomposition_cache[cache_key]
        
    
        observed_mask = torch.ones_like(data_window - trend)
        gt_mask = self.get_mask(data_window - trend, observed_mask)
        timepoints = np.arange(self.window_length)
        
        if self.get_label:
            return {
                "observed_data": data_window - trend,  
                "original_data": data_window,
                "trend": trend,
                "stable": stable,
                "preprocess_stable": preprocess_stable,
                "observed_mask": observed_mask,
                "gt_mask": gt_mask,
                "timepoints": timepoints,
                "label": label_window,
                'strategy_type': self.strategy,
                "timestamp_features": time_window  
            }
        else:
            return {
                "observed_data": data_window - trend,  
                "original_data": data_window,
                "trend": trend,
                "stable": stable,
                "preprocess_stable": preprocess_stable,
                "observed_mask": observed_mask,
                "gt_mask": gt_mask,
                "timepoints": timepoints,
                'strategy_type': self.strategy,
                "timestamp_features": time_window  
            }

def get_mask(observed_mask, mask_ratio):
    mask = torch.zeros_like(observed_mask)

    original_mask_shape = mask.shape

    mask = mask.reshape(-1)
    total_index_list = list(range(len(mask)))

    selected_number = int(len(total_index_list) * mask_ratio)

    selected_index = sample(total_index_list, selected_number)

    selected_index = torch.LongTensor(selected_index)

    mask[selected_index] = 1

    mask = mask.reshape(original_mask_shape)

    return mask

def get_dataloader(train_path, test_path, label_path,batch_size = 32,window_split=1,split=4,mask_ratio=0.5):
    train_data = TrainData(train_path,test_path,split=split,mask_ratio=mask_ratio)
    train_data, valid_data = random_split(
        train_data, [len(train_data) - int(0.05 * len(train_data)) , int(0.05 * len(train_data)) ]
    )

    temp_dict = train_data.__getitem__(0)
    observed_mask = temp_dict['observed_mask']

    mask_list = []

    for i in tqdm(range(0,100)):
        mask_list.append(get_mask(observed_mask,mask_ratio=mask_ratio))


    test_data_strategy_1 = TestData(test_path, label_path, train_path,window_split=window_split,strategy=0,split=split,mask_list=mask_list)
    test_data_strategy_2 = TestData(test_path, label_path, train_path, window_split=window_split, strategy=1,split=split,mask_list=mask_list)

    train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data,batch_size=batch_size,shuffle=True)

    test_loader1 = DataLoader(test_data_strategy_1,batch_size=batch_size)
    test_loader2 = DataLoader(test_data_strategy_2,batch_size=batch_size)

    return train_loader, valid_loader, test_loader1, test_loader2


if __name__ == "__main__":
    print("测试D3R+IMDiffusion预处理集成...")
    
    # 创建训练数据集
    train_data = TrainData(
        file_path="data/Machine/SMD_train.npy",
        test_path="data/Machine/SMD_test.npy",
        window_length=100,
        split=4,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    

    sample = train_data[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"observed_data shape: {sample['observed_data'].shape}")
    print(f"original_data shape: {sample['original_data'].shape}")
    print(f"trend shape: {sample['trend'].shape}")
    
  
    import matplotlib.pyplot as plt
    import numpy as np
    
   
    feature_idx = 0
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(sample['original_data'][:, feature_idx].numpy())
    plt.title('Original Data')
    
    plt.subplot(3, 1, 2)
    plt.plot(sample['trend'][:, feature_idx].numpy())
    plt.title('Trend Component')
    
    plt.subplot(3, 1, 3)
    plt.plot(sample['observed_data'][:, feature_idx].numpy())
    plt.title('Stable Component')
    
    plt.tight_layout()
    plt.savefig('decomposition_test.png')
    plt.close()
    
    print(f"分解结果已保存到 decomposition_test.png")
    
   
    train_loader, valid_loader, test_loader1, test_loader2 = get_dataloader(
        train_path="data/Machine/SMD_train.npy",
        test_path="data/Machine/SMD_test.npy",
        label_path="data/Machine/SMD_test_label.npy",
        batch_size=4,
        split=4
    )
    
    print(f"训练加载器批次数: {len(train_loader)}")
    print(f"验证加载器批次数: {len(valid_loader)}")
    print(f"测试加载器1批次数: {len(test_loader1)}")
    print(f"测试加载器2批次数: {len(test_loader2)}")
    
    print("预处理测试完成!")
