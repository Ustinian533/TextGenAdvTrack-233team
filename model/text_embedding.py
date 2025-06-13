import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel,RobertaModel
from peft import get_peft_config, LoraConfig, TaskType,PeftModel

class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name,output_hidden_states=False,lora=False,infer=False,use_pooling='average',lora_r=128,lora_alpha=256,lora_dropout=0):
        super(TextEmbeddingModel, self).__init__()
        self.model_name = model_name
        if output_hidden_states:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if lora:
            self.peft_config = LoraConfig(peft_type=TaskType.FEATURE_EXTRACTION,inference_mode=infer, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            self.model = PeftModel(self.model, self.peft_config)
            self.model.print_trainable_parameters()
        # if 'bge' in self.model_name.lower() or 'mxbai' in self.model_name.lower():
        #     use_pooling = 'cls'

        self.use_pooling = use_pooling

    def pooling(self, model_output, attention_mask,hidden_states=False):
        if hidden_states:
            if self.use_pooling == "average":
                model_output.masked_fill(~attention_mask[None,..., None].bool(), 0.0)
                emb = model_output.sum(dim=2) / attention_mask.sum(dim=1)[..., None]
            elif self.use_pooling == "max":
                emb = model_output.masked_fill(~attention_mask[None, ..., None].bool(), float('-inf'))
                emb, _ = emb.max(dim=2)
            elif self.use_pooling == "cls":
                emb = model_output[:,:, 0]
            else:
                raise ValueError("Pooling method not supported")
            emb = emb.permute(1, 0, 2)
        else:
            if self.use_pooling == "average":
                model_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
                emb = model_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            elif self.use_pooling == "max":
                emb = model_output.masked_fill(~attention_mask[..., None].bool(), float('-inf'))
                emb, _ = emb.max(dim=1)
            elif self.use_pooling == "cls":
                emb = model_output[:, 0]
            else:
                raise ValueError("Pooling method not supported")
        return emb
    
    def forward(self, encoded_batch,hidden_states=False,retrun_all_emb=False):
        if "t5" in self.model_name.lower():
            # print(self.model.config.pad_token_id)
            input_ids = encoded_batch['input_ids']
            decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
            model_output = self.model(**encoded_batch, 
                                  decoder_input_ids=decoder_input_ids)
        else:
            model_output = self.model(**encoded_batch)
        
        
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        if isinstance(model_output, dict):
            if hidden_states:
                model_output = model_output["hidden_states"]
                model_output = torch.stack(model_output, dim=0)
            else:
                model_output = model_output["last_hidden_state"]
            

        # print(self.model.config)
        
        
        emb = self.pooling(model_output, encoded_batch['attention_mask'],hidden_states)
        # emb = torch.nn.functional.normalize(emb, dim=-1)
        if retrun_all_emb:
            return emb,model_output
        return emb

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size,num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class TextClassificationModel(nn.Module):
    def __init__(self, opt,dim=2):
        super(TextClassificationModel, self).__init__()
        self.model = TextEmbeddingModel(opt.model_name,lora=True,use_pooling=opt.pooling,\
                                        lora_r=opt.lora_r,lora_alpha=opt.lora_alpha,infer=True)
        self.root_classfier = nn.Linear(opt.embedding_dim, dim)

    def forward(self, encoded_batch):
        q = self.model(encoded_batch)
        out = self.root_classfier(q)
        return out


if __name__ == '__main__':
    model_name = "answerdotai/ModernBERT-large"
    model = TextEmbeddingModel(model_name,output_hidden_states=True,lora=True).cuda()
    text=['zhangshan1111111111111111111111','lisi']
    encoded_batch = model.tokenizer.batch_encode_plus(
                        text,
                        return_tensors="pt",
                        max_length=512,
                        padding=True,
                        truncation=True,
                    )
    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
    emb = model(encoded_batch,hidden_states=True)[:,1:,:]
    print(emb.shape)

    # #计算模型参数量
    # num_params = sum(p.numel() for p in model.parameters())
    