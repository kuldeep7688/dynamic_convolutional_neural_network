import torch
import math
import torch.nn as nn


class DCNNCell(nn.Module):
    def __init__(
        self,
        cell_number=1,
        sent_length=7,
        conv_kernel_size=(3, 1),
        conv_input_channels=1,
        conv_output_channels=2,
        conv_stride=(1, 1),
        k_max_number=5,
        folding_kernel_size=(1, 2),
        folding_stride=(1,1)
    ):
        super().__init__()
        self.cell_number=cell_number 
        self.sent_length=sent_length
        self.conv_kernel_size=conv_kernel_size
        self.conv_input_channels=conv_input_channels
        self.conv_output_channels=conv_output_channels
        self.conv_stride=conv_stride
        self.k_max_number=k_max_number
        self.folding_kernel_size=folding_kernel_size
        self.folding_stride=folding_stride
        
        # calculating padding size
        self.pad_0_direction = math.ceil(self.conv_kernel_size[0]  - 1)
        self.pad_1_direction = math.ceil(self.conv_kernel_size[1] - 1)
        
        # 2d convolution
        self.conv_layer = nn.Conv2d(
            in_channels=self.conv_input_channels,
            out_channels=self.conv_output_channels,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=(self.pad_0_direction, self.pad_1_direction)
        )
        
        # if cell is last then initialising folding
        if cell_number == -1:
            self.fold = nn.AvgPool2d(kernel_size=self.folding_kernel_size, stride=self.folding_stride)
            
    def forward(self, inp):
        
        # [batch_size, input_channels, sent_length_in, embedding_dim]
        conved = self.conv_layer(inp)
        
        # [batch_size, out_channels, sent_length_out, embedding_dim]
        if self.cell_number == -1:
            conved = self.fold(conved)
        
        # [batch_size, out_channels, sent_length, embedding_dim/2]
        k_maxed = torch.tanh(torch.topk(conved, self.k_max_number, dim=2, largest=True)[0])
        
        # [batch_size, out_channels, k_maxed_number, embedding_dim/2]
        return k_maxed

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class DCNN_SST(nn.Module):
    def __init__(
        self,
        parameter_dict
    ):
        super().__init__()
        self.parameter_dict = parameter_dict
        
        self.embedding = nn.Embedding(
            embedding_dim=self.parameter_dict["embedding_dim"],
            num_embeddings=self.parameter_dict["vocab_length"]
        )
        self.dcnn_first_cell = DCNNCell(
            cell_number=-1,
            sent_length=self.parameter_dict["cell_one_parameter_dict"]["sent_length"],
            conv_kernel_size=self.parameter_dict["cell_one_parameter_dict"]["conv_kernel_size"],
            conv_input_channels=self.parameter_dict["cell_one_parameter_dict"]["conv_input_channels"],
            conv_output_channels=self.parameter_dict["cell_one_parameter_dict"]["conv_output_channels"],
            conv_stride=self.parameter_dict["cell_one_parameter_dict"]["conv_stride"],
            k_max_number=self.parameter_dict["cell_one_parameter_dict"]["k_max_number"],
            folding_kernel_size=self.parameter_dict["cell_one_parameter_dict"]["folding_kernel_size"],
            folding_stride=self.parameter_dict["cell_one_parameter_dict"]["folding_stride"],
        )
        self.dcnn_last_cell = DCNNCell(
            cell_number=-1,
            sent_length=self.parameter_dict["cell_two_parameter_dict"]["sent_length"],
            conv_kernel_size=self.parameter_dict["cell_two_parameter_dict"]["conv_kernel_size"],
            conv_input_channels=self.parameter_dict["cell_two_parameter_dict"]["conv_input_channels"],
            conv_output_channels=self.parameter_dict["cell_two_parameter_dict"]["conv_output_channels"],
            conv_stride=self.parameter_dict["cell_two_parameter_dict"]["conv_stride"],
            k_max_number=self.parameter_dict["cell_two_parameter_dict"]["k_max_number"],
            folding_kernel_size=self.parameter_dict["cell_two_parameter_dict"]["folding_kernel_size"],
            folding_stride=self.parameter_dict["cell_two_parameter_dict"]["folding_stride"],
        )
        self.fc_layer_input = self.parameter_dict["cell_two_parameter_dict"]["k_max_number"] *\
            self.parameter_dict["cell_two_parameter_dict"]["conv_output_channels"] *\
            math.floor(self.parameter_dict["embedding_dim"]/4)
            
        self.dropout = nn.Dropout(self.parameter_dict["dropout_rate"])
        self.flatten = Flatten()
        self.fc = nn.Linear(self.fc_layer_input, self.parameter_dict["output_dim"])
    
    def forward(self, inp):
        # [batch_size, sent_length]
        embedded = self.embedding(inp)
        # [batch_size, sent_length, embedding_dim]
        # adding single channel dimension
        embedded = embedded.unsqueeze(1)
        # print(embedded.shape)
        # [batch_size, 1(initial_input_channel), sent_length, embedding_dim]
        out = self.dcnn_first_cell(embedded)
        # print(out.shape)
        # [batch_size, first_cell_output_channels, first_cell_k_maxed_number, embedding_dim]
        out = self.dcnn_last_cell(out)
        # print(out.shape)
        # [batch_size, last_cell_output_channels, last_cell_k_maxed_number, embedding_dim/2]
        out = self.dropout(self.flatten(out))
        # print(flat.shape)
        #[batch_size, last_cell_output_channels * last_cell_k_maxed_number * embedding_dim/2]
        out = self.fc(out)
        # print(fc.shape)
        return out

