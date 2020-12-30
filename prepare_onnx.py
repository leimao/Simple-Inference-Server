import os
import torch
from typing import Tuple
from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering

def get_bert_qa_model(model_name="deepset/bert-base-cased-squad2", cache_dir="./cache") -> Tuple[BertConfig, BertForQuestionAnswering, BertTokenizer]:

    # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForQuestionAnswering
    config = BertConfig.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = BertForQuestionAnswering.from_pretrained(model_name, config=config, cache_dir=cache_dir)

    return config, model, tokenizer

def main() -> None:

    model_name = "deepset/bert-base-cased-squad2"
    cache_dir = "./cache"
    output_dir = "./saved_models"
    torch_model_name = "bert-base-cased-squad2_model.pt"
    onnx_model_name = "bert-base-cased-squad2_model.onnx"
    torch_model_config_name = "bert-base-cased-squad2_config.pt"
    tokenizer_name = "bert-base-cased-squad2_tokenizer.pt"
    onnx_opset_version = 11
    max_seq_length = 512

    config, model, tokenizer = get_bert_qa_model(model_name=model_name, cache_dir=cache_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.save_pretrained(os.path.join(output_dir, torch_model_name))
    tokenizer.save_pretrained(os.path.join(output_dir, tokenizer_name))
    config.save_pretrained(os.path.join(output_dir, torch_model_config_name))

    inputs = {
        'input_ids':      torch.zeros((1, max_seq_length)).long(),
        'attention_mask': torch.zeros((1, max_seq_length)).long(),
        'token_type_ids': torch.zeros((1, max_seq_length)).long(),
    }

    model.eval()

    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(
            model,                                            # model being run
            args=tuple(inputs.values()),                      # model input (or a tuple for multiple inputs)
            f=os.path.join(output_dir, onnx_model_name),                              # where to save the model (can be a file or file-like object)
            opset_version=onnx_opset_version,                      # the ONNX version to export the model to
            do_constant_folding=True,                         # whether to execute constant folding for optimization
            input_names=[
                'input_ids',                         # the model's input names
                'input_mask', 
                'segment_ids'
            ],
            output_names=['start', 'end'],                    # the model's output names
            dynamic_axes={
                'input_ids': symbolic_names,        # variable length axes
                'input_mask' : symbolic_names,
                'segment_ids' : symbolic_names,
                'start' : symbolic_names,
                'end' : symbolic_names
            }
        )
        print("Model exported at ", os.path.join(output_dir, onnx_model_name))

if __name__ == "__main__":
    
    main()