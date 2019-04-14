SST2_DATASET_PARAMETERS = {
    "cell_one_parameter_dict" : {
        "sent_length": 19,
        "conv_kernel_size": (7, 1),
        "conv_input_channels": 1,
        "conv_output_channels": 6,
        "conv_stride": (1, 1),
        "k_max_number": 10,
        "folding_kernel_size": (1, 2),
        "folding_stride": (1, 2)
    },
    "cell_two_parameter_dict" : {
        "sent_length": None,
        "conv_kernel_size": (5, 1),
        "conv_input_channels": 6,
        "conv_output_channels": 14,
        "conv_stride": (1, 1),
        "k_max_number": 4,
        "folding_kernel_size": (1, 2),
        "folding_stride": (1, 2)
    },
    "dropout_rate": 0.5,
    "embedding_dim": None,
    "vocab_length": None,
    "output_dim": 2
}
SST2_DATASET_PARAMETERS["cell_two_parameter_dict"]["sent_length"] = SST2_DATASET_PARAMETERS["cell_one_parameter_dict"]["k_max_number"]

SST1_DATASET_PARAMETERS = {
    "cell_one_parameter_dict" : {
        "sent_length": 19,
        "conv_kernel_size": (10, 1),
        "conv_input_channels": 1,
        "conv_output_channels": 6,
        "conv_stride": (1, 1),
        "k_max_number": 9,
        "folding_kernel_size": (1, 2),
        "folding_stride": (1, 2)
    },
    "cell_two_parameter_dict" : {
        "sent_length": None,
        "conv_kernel_size": (7, 1),
        "conv_input_channels": 6,
        "conv_output_channels": 12,
        "conv_stride": (1, 1),
        "k_max_number": 5,
        "folding_kernel_size": (1, 2),
        "folding_stride": (1, 2)
    },
    "dropout_rate": 0.5,
    "embedding_dim": None,
    "vocab_length": None,
    "output_dim": 5
}
SST1_DATASET_PARAMETERS["cell_two_parameter_dict"]["sent_length"] = SST1_DATASET_PARAMETERS["cell_one_parameter_dict"]["k_max_number"]

TREC_DATASET_PARAMETERS = {
    "cell_one_parameter_dict": {
        "sent_length": 12,
        "conv_kernel_size": (8, 1),
        "conv_input_channels": 1,
        "conv_output_channels": 5,
        "conv_stride": (1, 1),
        "k_max_number": 8,
        "folding_kernel_size": (1, 2),
        "folding_stride": (1, 2)
    },
    "dropout_rate": 0.5,
    "embedding_dim": None,
    "vocab_length": None,
    "output_dim": 6
}