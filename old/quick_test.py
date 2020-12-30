# qa.py

import os
import time
import torch
from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering

def measure_inference_latency(model, inputs, num_samples=100):

    start_time = time.time()
    for _ in range(num_samples):
        _ = model(**inputs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave

def get_bert_qa_model(model_name="deepset/bert-base-cased-squad2", cache_dir="./saved_models"):

    # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForQuestionAnswering
    config = BertConfig.from_pretrained(model_name, cache_dir=cache_dir)
    print(config)
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = BertForQuestionAnswering.from_pretrained(model_name, config=config, cache_dir=cache_dir)

    return model, tokenizer

def prepare_qa_inputs(question, text, tokenizer, device=None):

    inputs = tokenizer(question, text, return_tensors="pt")
    if device is not None:
        inputs_cuda = dict()
        for input_name in inputs.keys():
            inputs_cuda[input_name] = inputs[input_name].to(device)
        inputs = inputs_cuda
    
    return inputs

def move_inputs_to_device(inputs, device=None):

    inputs_cuda = dict()
    for input_name in inputs.keys():
        inputs_cuda[input_name] = inputs[input_name].to(device)

    return inputs_cuda

def run_qa(model, tokenizer, question, text, device=None):

    inputs = prepare_qa_inputs(question=question, text=text, tokenizer=tokenizer)

    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])

    if device is not None:
        inputs = move_inputs_to_device(inputs, device=device)
        model = model.to(device)

    outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_start_idx = torch.argmax(start_scores, 1)[0]
    answer_end_idx = torch.argmax(end_scores, 1)[0] + 1

    answer = " ".join(all_tokens[answer_start_idx : answer_end_idx])

    return answer

def get_model_size(model, temp_dir="/tmp"):

    model_dir = os.path.join(temp_dir, "temp")
    torch.save(model.state_dict(), model_dir)
    # model.save_pretrained(model_dir)
    size = os.path.getsize(model_dir)
    os.remove(model_dir)
    
    return size

def main():

    cuda_device = torch.device("cuda:0")
    num_samples = 100

    model, tokenizer = get_bert_qa_model(model_name="deepset/bert-base-cased-squad2")
    model.eval()
    # https://pytorch.org/docs/stable/torch.quantization.html?highlight=torch%20quantization%20quantize_dynamic#torch.quantization.quantize_dynamic
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    print("=" * 75)
    print("Model Sizes")
    print("=" * 75)

    model_size = get_model_size(model=model)
    quantized_model_size = get_model_size(model=quantized_model)

    print("FP32 Model Size: {:.2f} MB".format(model_size / (2 ** 20)))
    print("INT8 Model Size: {:.2f} MB".format(quantized_model_size / (2 ** 20)))

    question = "What publication printed that the wealthiest 1% have more money than those in the bottom 90%?"

    text = "According to PolitiFact the top 400 richest Americans \"have more wealth than half of all Americans combined.\" According to the New York Times on July 22, 2014, the \"richest 1 percent in the United States now own more wealth than the bottom 90 percent\". Inherited wealth may help explain why many Americans who have become rich may have had a \"substantial head start\". In September 2012, according to the Institute for Policy Studies, \"over 60 percent\" of the Forbes richest 400 Americans \"grew up in substantial privilege\"."

    inputs = prepare_qa_inputs(question=question, text=text, tokenizer=tokenizer)
    answer = run_qa(model=model, tokenizer=tokenizer, question=question, text=text)
    answer_quantized = run_qa(model=quantized_model, tokenizer=tokenizer, question=question, text=text)

    print("=" * 75)
    print("BERT QA Example")
    print("=" * 75)

    print("Text: ")
    print(text)
    print("Question: ")
    print(question)
    print("Model Answer: ")
    print(answer)
    print("Dynamic Quantized Model Answer: ")
    print(answer_quantized)

    print("=" * 75)
    print("BERT QA Inference Latencies")
    print("=" * 75)

    model_latency = measure_inference_latency(model=model, inputs=inputs, num_samples=num_samples)
    print("CPU Inference Latency: {:.2f} ms / sample".format(model_latency * 1000))

    quantized_model_latency = measure_inference_latency(model=quantized_model, inputs=inputs, num_samples=num_samples)
    print("Dynamic Quantized CPU Inference Latency: {:.2f} ms / sample".format(quantized_model_latency * 1000))

    inputs_cuda = move_inputs_to_device(inputs, device=cuda_device)
    model.to(cuda_device)
    model_cuda_latency = measure_inference_latency(model=model, inputs=inputs_cuda, num_samples=num_samples)
    print("CUDA Inference Latency: {:.2f} ms / sample".format(model_cuda_latency * 1000))

    # No CUDA backend for dynamic quantization in PyTorch 1.7.0
    # quantized_model_cuda = quantized_model.to(cuda_device)
    # quantized_model_cuda_latency = measure_inference_latency(model=quantized_model_cuda, inputs=inputs_cuda, num_samples=num_samples)
    # print("Dynamic Quantized GPU Inference Latency: {:.2f} ms / sample".format(quantized_model_cuda_latency * 1000))

if __name__ == "__main__":

    main()