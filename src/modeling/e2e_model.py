from src.modeling.modeling import (
    PixelBertForPreTraining,
    PixelBertForSequenceClassification,
    PixelBertForMultipleChoice
    )
from src.modeling.grid_feat import GridFeatBackbone
from torch import nn
from src.datasets.data_utils import repeat_tensor_rows
from src.utils.load_save import load_state_dict_with_mismatch
from src.modeling.resnet50gcb_extra import ResNet50Extra
from src.modeling.resnet31gcb_extra import ResNet31Extra


class PixelBert(nn.Module):
    def __init__(self, config, input_format="BGR",
                 detectron2_model_cfg=None,
                 transformer_cls=PixelBertForPreTraining):
        super(PixelBert, self).__init__()
        self.config = config
        self.detectron2_model_cfg = detectron2_model_cfg
        assert detectron2_model_cfg is not None
        cnn_cls = GridFeatBackbone
        print(f"cnn_cls {cnn_cls}")
        self.cnn = cnn_cls(
            detectron2_model_cfg=detectron2_model_cfg,
            config=config, input_format=input_format)
        self.transformer = transformer_cls(config)
        

    def forward(self, batch):
        # used to make visual feature copies
        repeat_counts = batch["n_examples_list"]
        del batch["n_examples_list"]
        # print('cnn input shape: ',batch["visual_inputs"].shape)
        visual_features = self.cnn(batch["visual_inputs"].squeeze(dim=1))
        batch["visual_inputs"] = repeat_tensor_rows(
            visual_features, repeat_counts)
        
        outputs = self.transformer(**batch)  # dict
        return outputs

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False


class PixelBertWithResNet50GCB(nn.Module):
    def __init__(self, config, input_format="BGR",
                 transformer_cls=PixelBertForPreTraining):
        super(PixelBertWithResNet50GCB, self).__init__()
        self.config = config
        cnn_cls = ResNet50Extra
        print(f"cnn_cls {cnn_cls}")
        gcb_config = dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, True, True, True],
        )
        self.cnn = cnn_cls(layers=[3,4,6,3], input_dim=3, gcb_config=gcb_config, input_format=input_format)
        self.transformer = transformer_cls(config)
        

    def forward(self, batch):
        # used to make visual feature copies
        repeat_counts = batch["n_examples_list"]
        del batch["n_examples_list"]
        # print('cnn input shape: ',batch["visual_inputs"].shape)
        visual_features = self.cnn(batch["visual_inputs"].squeeze(dim=1))[-1]
        batch["visual_inputs"] = repeat_tensor_rows(
            visual_features, repeat_counts)
        
        outputs = self.transformer(**batch)  # dict
        return outputs

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False


class PixelBertWithResNet50(nn.Module):
    def __init__(self, config, input_format="BGR",
                 transformer_cls=PixelBertForPreTraining):
        super(PixelBertWithResNet50, self).__init__()
        self.config = config
        cnn_cls = ResNet50Extra
        print(f"cnn_cls {cnn_cls}")
        gcb_config = dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, False, False, False],
        )
        self.cnn = cnn_cls(layers=[3,4,6,3], input_dim=3, gcb_config=gcb_config, input_format=input_format)
        self.transformer = transformer_cls(config)
        

    def forward(self, batch):
        # used to make visual feature copies
        repeat_counts = batch["n_examples_list"]
        del batch["n_examples_list"]
        # print('cnn input shape: ',batch["visual_inputs"].shape)
        visual_features = self.cnn(batch["visual_inputs"].squeeze(dim=1))[-1]
        batch["visual_inputs"] = repeat_tensor_rows(
            visual_features, repeat_counts)
        
        outputs = self.transformer(**batch)  # dict
        return outputs

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False



class PixelBertWithResNet31GCB(nn.Module):
    def __init__(self, config, input_format="BGR",
                 transformer_cls=PixelBertForPreTraining):
        super(PixelBertWithResNet31GCB, self).__init__()
        self.config = config
        cnn_cls = ResNet31Extra
        print(f"cnn_cls {cnn_cls}")
        gcb_config = dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, True, True, True],
        )
        self.cnn = cnn_cls(layers=[1,2,5,3], input_dim=3, gcb_config=gcb_config, input_format=input_format)
        self.transformer = transformer_cls(config)
        

    def forward(self, batch):
        # used to make visual feature copies
        repeat_counts = batch["n_examples_list"]
        del batch["n_examples_list"]
        # print('cnn input shape: ',batch["visual_inputs"].shape)
        visual_features = self.cnn(batch["visual_inputs"].squeeze(dim=1))[-1]
        batch["visual_inputs"] = repeat_tensor_rows(
            visual_features, repeat_counts)
        
        outputs = self.transformer(**batch)  # dict
        return outputs

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False