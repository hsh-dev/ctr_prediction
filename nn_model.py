import torch
import torch.nn as nn

class NumericEmbeddingLayer(nn.Module):
    
    def __init__(
        self,
        bucket_nums: int = 4,
        embedding_dim: int = 4,
        global_temperature: float = 1e-1,
    ) -> None:
        super().__init__()
        
        self.bucket_nums = bucket_nums
        self.embedding_dim = embedding_dim
        self.global_temperature = global_temperature    ## not adaptive
        
        ## Meta Embedding   shape : bucket nums x embedding dim
        self.meta_embedding = nn.Linear(
            self.bucket_nums, 
            self.embedding_dim, 
            bias=False,
        )

        ## numeric value -> discret vector
        self.discret_w = nn.Linear(
            in_features=1, 
            out_features=self.bucket_nums, 
            bias=False
        )        ## 1 x bucket
        self.discret_leaky_relu = nn.LeakyReLU() 
        self.discret_kernel = nn.Linear(
            self.bucket_nums, 
            self.bucket_nums, 
            bias=True,
        )       ## bucket x bucket

    def forward(self, x):
        ### x     ## shape : batch size x 1
        x_discret = self.discret_kernel(
            self.discret_leaky_relu(self.discret_w(x))
        )    ## shape : batch size x bucket nums

        ## softmax temperature
        temperature = torch.tile(
            torch.Tensor([self.global_temperature]), 
            dims=(x.shape[0], 1)
        )   ## shape : batch size x 1

        ## calculate softmax
        x_discret = x_discret / temperature
        x_discret = torch.softmax(x_discret, dim = 1)   ## shape : batch size x bucket nums

        ## Get Embedding
        x_emb = self.meta_embedding(x_discret)  ## shape : batch size x embedding dim

        return x_emb
        
        ## calculate adaptive temperature
        # global_statistics_feature = torch.Tensor([max_value, min_value, mean_value])
        # temp_dim = 4
        # temp_w = nn.Linear(4, temp_dim)         ## 4 (statistics feature(3), input feature(1)) x temp dim
        # temp_leaky_relu = nn.LeakyReLU()
        # temp_kernel = nn.Linear(temp_dim, 1, bias=False)    ## temp dim x 1
        # temp_sigmoid = nn.Sigmoid()

        # statistics_input = torch.tile(
        #     global_statistics_feature, 
        #     dims = (x_input.shape[0], 1)
        # )   ## batch size x 3
        # temp_input = torch.concat([statistics_input, x_input], dim=1)   ## [batch size x 3], [batch size x 1]
        # temp_output = temp_leaky_relu(temp_w(temp_input))
        # adaptive_temp = temp_sigmoid(temp_kernel(temp_output))        


class FMLayer(nn.Module):
    
    def __init__(
        self,
        feature_nums: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        
        self.feature_nums = feature_nums
        self.embedding_dim = embedding_dim
        
        self.order_1_kernel = nn.Linear(self.feature_nums, 1)
        self.order_1_final_kernel = nn.Linear(self.embedding_dim, 1)
        
        
    def forward(
        self, 
        x_input_embeddings, 
        x_indicator
    ):
        # x_input_embeddings    ## batch size x feature nums x embedding dim  
        # x_indicator           ## batch size x feature nums
        
        _batch_size = x_input_embeddings.shape[0]
        
        ### order - 1
        x_input_trans = torch.transpose(
            x_input_embeddings, 1, 2
        )   ## batch size x embedding dim  x feature nums
        order_1_output = torch.squeeze(
            self.order_1_kernel(x_input_trans),
            2
        )   ## batch size x embedding dim
        order_1_output = self.order_1_final_kernel(order_1_output)   ## batch size  x 1

        ### order - 2
        order_2_output = torch.matmul(
            x_input_embeddings,
            x_input_trans
        )   ## batch size x feature nums x feature nums

        order_2_mask = torch.tile(
            torch.diag(
                torch.ones(self.feature_nums),
            ),
            dims = (_batch_size, 1, 1)
        )   
        order_2_mask = torch.ones_like(order_2_mask) - order_2_mask 
        ## zero mask for self inner product  ## batch size  x feature nums x feature nums

        indicator_matrix = torch.matmul(
            torch.unsqueeze(x_indicator, 2),
            torch.unsqueeze(x_indicator, 1)
        )   ## batch size x feature nums x feature nums

        indicator_matrix = indicator_matrix * order_2_mask
        order_2_output = order_2_output * indicator_matrix
        order_2_output = torch.sum(order_2_output, (1, 2)) / 2  ## batch size

        ### FM output
        fm_output = order_1_output + torch.unsqueeze(order_2_output, 1)

        return fm_output


class DeepLayer(nn.Module):

    def __init__(
        self,
        feature_nums: int,
        embedding_dim: int,
        dropout_ratio: float = 0.6,
        hidden_dim: int = 4,
        hidden_layer_nums: int = 3,
    ) -> None:
        super().__init__()
        
        self.feature_nums = feature_nums
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hidden_layer_nums = hidden_layer_nums
        
        self.feature_aggregate_layer = nn.Linear(
            in_features=self.feature_nums,
            out_features=1
        )
        
        self.dense_input_layer = nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.hidden_dim
        )
        self.activation_input_layer = nn.ReLU()
        
        self.dense_layers = []
        self.activation_layers = []
        self.dropout_layers = []
        for i in range(0, hidden_layer_nums):
            _dense_layer = nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim
            )                                               ## batch size x hidden dim
            _activation_layer = nn.ReLU()
            _dropout_layer = nn.Dropout(dropout_ratio)
            
            self.dense_layers.append(_dense_layer)
            self.activation_layers.append(_activation_layer)
            self.dropout_layers.append(_dropout_layer)
        
        self.final_layer = nn.Linear(
            in_features=self.hidden_dim,
            out_features=1
        )
        
    def forward(
        self, 
        x_input_embeddings, 
        x_indicator
    ):
        x_input_embeddings = x_input_embeddings * x_indicator.unsqueeze(dim=2)
        x_input_trans = torch.transpose(x_input_embeddings, 1, 2)   ## batch size x embedding dim x feature nums

        ### 1. Feature Aggregation
        dense_input = torch.squeeze(
            self.feature_aggregate_layer(x_input_trans),
            2
        )   ## batch size x embedding dim

        ### 2. Dense Input
        dense_output = self.dense_input_layer(dense_input)    ## batch size x hidden dim
        dense_output = self.activation_input_layer(dense_output) ## batch size x hidden dim

        ### 3. Dense Layers
        for i in range(self.hidden_layer_nums):
            dense_output = self.dense_layers[i](dense_output)
            dense_output = self.activation_layers[i](dense_output) ## batch size x hidden dim
            dense_output = self.dropout_layers[i](dense_output)

        ### 4. Final Layer
        dense_output = self.final_layer(dense_output)
        
        return dense_output
    

class DeepFM(nn.Module):
    
    def __init__(
        self,
        catalog_nums: list[int],
        numerical_len: list[str],
        categorical_len: list[str],
        embedding_dim: int = 4,
        hidden_dim: int = 4,
        bucket_nums: int = 4,
        hidden_layer_nums: int = 3,
        dropout_ratio: float = 0.6,
    ) -> None:
        super().__init__()

        assert len(catalog_nums) == categorical_len

        self.categorical_layers = [] 
        for _catalog_num in catalog_nums:        
            categorical_embedding_layer = nn.Embedding(
                num_embeddings=_catalog_num + 1,
                embedding_dim=embedding_dim,
                padding_idx=0
            )
            self.categorical_layers.append(categorical_embedding_layer)
        
        self.numerical_layers = [] 
        for i in range(numerical_len):
            numerical_embedding_layer = NumericEmbeddingLayer(
                bucket_nums=bucket_nums,
                embedding_dim=embedding_dim,
                global_temperature=0.1
            )
            self.numerical_layers.append(numerical_embedding_layer)

        self.x_numerical_indices = [i for i in range(numerical_len)]
        self.x_categorical_indices = [i for i in range(categorical_len)]


        self.fm_layer = FMLayer(
            feature_nums=numerical_len + categorical_len,
            embedding_dim=embedding_dim
        )
        
        self.deep_layer = DeepLayer(
            feature_nums=numerical_len + categorical_len,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            hidden_layer_nums=hidden_layer_nums,
            dropout_ratio=dropout_ratio,
        )

    def forward(
        self, 
        x_numeric, 
        x_categoric,
        x_numeric_indicator,
        x_categoric_indicator,
    ):
        ## x_numeric (batch size, numeric feature)                  float64
        ## x_categoric (batch size, categorical feature)            int64
        ## x_numeric_indicator (batch size, numeric feature)        int64
        ## x_categoric_indicator (batch size, categorical feature)  int64
        
        ### 1. Gather Embeddings
        embeddings = []
        
        for idx in self.x_numerical_indices:
            numeric_input = x_numeric[:, idx]
            numeric_input = torch.reshape(
                numeric_input, 
                shape=(numeric_input.shape[0], 1)
            )   ## batch size x 1
            
            numeric_emb = self.numerical_layers[idx](numeric_input)
            embeddings.append(numeric_emb)
            
        for idx in self.x_categorical_indices:
            categorical_input = x_categoric[:, idx]

            categorical_emb = self.categorical_layers[idx](categorical_input)
            embeddings.append(categorical_emb)
                    
        embeddings = torch.stack(embeddings)    ## feature nums x batch size x embedding dim
        input_embeddings = torch.transpose(
            embeddings, 0, 1
        )  ## batch size  x feature nums x embedding dim
        
        ### 2. Apply to Layers 
        indicator = torch.concat(
            [x_numeric_indicator, x_categoric_indicator],
            dim=1 
        ) 
        
        fm_output = self.fm_layer(
            input_embeddings, 
            indicator
        )   ## batch size x 1
                
        deep_output = self.deep_layer(
            input_embeddings,
            indicator
        )   ## batch size x 1
        
        output_sum = fm_output + deep_output    ## batch size x 1
        
        ### 3. Predict
        logits = torch.sigmoid(output_sum)
        
        return logits