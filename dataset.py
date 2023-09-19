from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    
    label_index = 0
    numerical_indices = [i for i in range(1, 14)]
    categorical_indices = [i for i in range(14, 40)]
    
    def __init__(
        self, 
        df: pd.DataFrame,
        dataset_type: str = "train",
        categorical_layers: dict[int, dict] = None,
        catalog_nums: list[int] = None,
        numerical_meta: dict[int, list[int, int]] = None
    ) -> None:
        
        df = df.fillna(np.NaN)
        self.df = df
        
        if dataset_type != "train":
            assert categorical_layers is not None
            assert catalog_nums is not None
            assert numerical_meta is not None
            
            self.categorical_layers = categorical_layers
            self.catalog_nums = catalog_nums
            self.numerical_meta = numerical_meta
            
        else:
            ## Numerical features
            self.numerical_meta = {}        ## idx -> (min, max)
            for _idx in self.numerical_indices:
                min_value = self.df[_idx].min()
                max_value = self.df[_idx].max()
                
                self.numerical_meta[_idx] = [float(min_value), float(max_value)]
            
            ## Categorical features
            self.categorical_layers = {}    ## idx -> categorical layer
            self.catalog_nums = []          ## total catalog nums
            for _idx in self.categorical_indices:
                catalogs = self.df[_idx]\
                            .value_counts(dropna=True)\
                            .index.values
                catalog_num = len(
                    self.df[_idx]\
                        .value_counts(dropna=True)
                )
                
                categorical_layer = self._make_lookup_layer(
                    catalogs = catalogs
                )
                self.categorical_layers[_idx] = categorical_layer
                self.catalog_nums.append(catalog_num)
    
    def export_data(self):
        return self.categorical_layers, self.catalog_nums, self.numerical_meta
    
    def _make_lookup_layer(
        self, 
        catalogs: list[str]
    ) -> dict[str, int]:
        idx = 1
        lookup_layer = dict()
        
        for catalog in catalogs:
            if lookup_layer.get(catalog, None) is not None:
                continue
            else:
                lookup_layer[catalog] = idx
                idx += 1
        
        return lookup_layer
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        ## 0 -> label, 1~13 -> numerical, 14 ~ 39 -> categorical
        row_dict = row.to_dict()
        label = row_dict[0]
        
        ## Get Numerical Value
        numeric_values = []
        numeric_indicators = []
        
        for _idx in self.numerical_indices:
            if np.isnan(row_dict[_idx]):
                numeric_indicators.append(0)
                numeric_values.append(0)
            else:
                _value = row_dict[_idx]
                _min_value = self.numerical_meta[_idx][0]
                _max_value = self.numerical_meta[_idx][1]
                
                _normalized_value = (_value - _min_value) / (_max_value - _min_value)
                
                numeric_indicators.append(1)
                numeric_values.append(
                    _normalized_value
                )
        
        ## Get Categorical value
        categoric_values = []
        categoric_indicators = []
        for _idx in self.categorical_indices:
            catagorical_input = row_dict[_idx]
            lookup_value = self.categorical_layers[_idx].get(
                catagorical_input, 0
            )   ## 0 for padding, invalid inputs
            
            if np.isnan(lookup_value):
                categoric_indicators.append(0)
                categoric_values.append(0)
            else:
                categoric_indicators.append(1)
                categoric_values.append(lookup_value)
            
        return (
            label, 
            np.array(numeric_values, np.float32),
            np.array(categoric_values, np.int32),
            np.array(numeric_indicators, np.int32),
            np.array(categoric_indicators, np.int32)
        )
    
    def __len__(self):
        return len(self.df)
        