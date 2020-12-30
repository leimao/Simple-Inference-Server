from typing import Optional, Dict, List
import onnxruntime
import numpy as np
import torch
from transformers import BertTokenizer, BertForQuestionAnswering


class QaInferenceSession(object):

    def __init__(self, model_filepath: str, tokenizer_filepath: str) -> None:

        self.model_filepath = model_filepath
        self.tokenizer_filepath = tokenizer_filepath

    def run(self, question, text):

        raise NotImplementedError

class QaTorchInferenceSession(QaInferenceSession):

    def __init__(self, model_filepath: str, tokenizer_filepath: str, device: Optional[str] = "cuda:0") -> None:

        super(QaTorchInferenceSession, self).__init__(model_filepath=model_filepath, tokenizer_filepath=tokenizer_filepath)

        self.model = BertForQuestionAnswering.from_pretrained(self.model_filepath).eval()
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_filepath)
        self.device = device
        self.model.to(self.device)

    def prepare_qa_inputs(self, question, text, device=None) -> Dict[str, torch.Tensor]:

        inputs = self.tokenizer(question, text, return_tensors="pt")
        if self.device is not None:
            inputs_cuda = dict()
            for input_name in inputs.keys():
                inputs_cuda[input_name] = inputs[input_name].to(self.device)
            inputs = inputs_cuda
        
        return inputs

    def run(self, question, text):

        inputs = self.prepare_qa_inputs(question=question, text=text)
        all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].cpu().numpy()[0])
        outputs = self.model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        answer_start_idx = torch.argmax(start_scores, 1)[0]
        answer_end_idx = torch.argmax(end_scores, 1)[0] + 1

        answer = " ".join(all_tokens[answer_start_idx : answer_end_idx])

        return answer

class QaOnnxInferenceSession(QaInferenceSession):

    def __init__(self, model_filepath: str, tokenizer_filepath: str, num_intra_op_num_threads: int = 1, execution_providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]) -> None:

        super(QaOnnxInferenceSession, self).__init__(model_filepath=model_filepath, tokenizer_filepath=tokenizer_filepath)

        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = num_intra_op_num_threads
        # Set graph optimization level
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # To enable model serialization after graph optimization set this
        # sess_options.optimized_model_filepath = "<model_output_path/optimized_model.onnx>"

        self.num_intra_op_num_threads = num_intra_op_num_threads
        self.execution_providers = execution_providers

        self.session = onnxruntime.InferenceSession(self.model_filepath, sess_options)
        self.session.set_providers(execution_providers)
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_filepath)

    def run(self, question, text):

        inputs = self.tokenizer(question, text, return_tensors="np")
        all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        ort_inputs = {
            "input_ids":  inputs["input_ids"],
            "input_mask": inputs["attention_mask"],
            "segment_ids": inputs["token_type_ids"],
        }

        outputs = self.session.run(["start", "end"], ort_inputs)

        start_scores = outputs[0]
        end_scores = outputs[1]

        answer_start_idx = np.argmax(start_scores, 1)[0]
        answer_end_idx = np.argmax(end_scores, 1)[0] + 1

        answer = " ".join(all_tokens[answer_start_idx : answer_end_idx])

        return answer

if __name__ == "__main__":

    onnx_model_filepath = "./saved_models/bert-base-cased-squad2_model.onnx"
    torch_model_filepath = "./saved_models/bert-base-cased-squad2_model.pt"
    tokenizer_filepath = "./saved_models/bert-base-cased-squad2_tokenizer.pt"

    onnx_inference_session = QaOnnxInferenceSession(model_filepath=onnx_model_filepath, tokenizer_filepath=tokenizer_filepath)
    torch_inference_session = QaTorchInferenceSession(model_filepath=torch_model_filepath, tokenizer_filepath=tokenizer_filepath)

    question = "What publication printed that the wealthiest 1% have more money than those in the bottom 90%?"

    text = "According to PolitiFact the top 400 richest Americans \"have more wealth than half of all Americans combined.\" According to the New York Times on July 22, 2014, the \"richest 1 percent in the United States now own more wealth than the bottom 90 percent\". Inherited wealth may help explain why many Americans who have become rich may have had a \"substantial head start\". In September 2012, according to the Institute for Policy Studies, \"over 60 percent\" of the Forbes richest 400 Americans \"grew up in substantial privilege\"."

    onnx_answer = onnx_inference_session.run(question=question, text=text)
    torch_answer = torch_inference_session.run(question=question, text=text)

    print(onnx_answer)
    print(torch_answer)