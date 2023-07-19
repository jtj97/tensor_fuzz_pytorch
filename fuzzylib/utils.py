import numpy as np
import random
import torch
import torch.nn.functional as F
from PIL import Image
import os

def numpy_to_pic(numpy_array, path):
    numpy_array = numpy_array.reshape(28, 28)
    numpy_array = numpy_array * 255
    image = Image.fromarray(numpy_array)
    image = image.convert('L')
    image.save(path)

def save_results(corpus_element, path):
    base_name = os.path.splitext(path)[0]
    suffix = os.path.splitext(path)[1]
    numpy_to_pic(corpus_element.data[0].numpy(), f'{base_name}_FP32{suffix}')
    fp16_numpy = corpus_element.data[0].numpy()
    fp16_numpy = fp16_numpy.astype(np.float16).astype(np.float32)
    numpy_to_pic(fp16_numpy, f'{base_name}_FP16{suffix}')
    
def metadata_function(metadata_batches):
    """Gets the metadata."""
    logit_32_batch = metadata_batches[0]
    logit_16_batch = metadata_batches[1]
    metadata_list = []
    for idx in range(logit_16_batch.shape[0]):
        metadata_list.append((logit_32_batch[idx], logit_16_batch[idx]))
    return metadata_list

def objective_function(corpus_element):
    """Checks if the element is misclassified."""
    logits_32 = corpus_element.metadata[0]
    logits_16 = corpus_element.metadata[1]
    prediction_16 = np.argmax(logits_16)
    prediction_32 = np.argmax(logits_32)

    if prediction_16 == prediction_32:
        return False
    
    print(f"Objective function satisfied: 32: {prediction_32}, 16: {prediction_16}")
    return True

def coverage_function(coverage_batch):
    coverage_list = []
    elt = coverage_batch[0]
    coverage_list.append(elt)
    return coverage_list

def coverage_function_batched(coverage_batch):
    coverage_list = []
    for idx in range(coverage_batch.shape[1]):
        elt = coverage_batch[0][idx] # logit_32
        coverage_list.append(elt)
    return coverage_list


def mutation_function(corpus_element, mutations_count, a_min, a_max, sigma = 0.2):
    data = corpus_element.data[0].numpy()
    data_batch = np.tile(data, [mutations_count, 1, 1])
        
    noise = np.random.normal(size=data_batch.shape, scale=sigma)
    np.clip(noise, -1, 1, out = noise)
    #print(np.average(np.abs(noise)))
    mutated_data_batch = noise + data_batch
    
    np.clip(mutated_data_batch, a_min, a_max, out = mutated_data_batch)
    
    return mutated_data_batch

def fetch_function(model, model_fp16, input):
    #input = input.to(torch.device('cuda:0'))
    scores_fp32, coverage_fp32 = model(input)
    scores_fp16, coverage_fp16 = model_fp16(input)
    
    scores_softmax_fp32 = F.log_softmax(scores_fp32, dim=-1)
    scores_softmax_fp16 = F.log_softmax(scores_fp16, dim=-1)
    
    metadata_batches = np.concatenate((scores_softmax_fp32.cpu().numpy(), scores_softmax_fp16.float().cpu().numpy()), axis = 0)
    coverage_batches = np.concatenate((coverage_fp32.cpu().numpy(), coverage_fp16.float().cpu().numpy()), axis = 0)
    
    return coverage_batches, metadata_batches

def fetch_function_batched(model, model_fp16, input_batch): # need fix
    # (mutations, 1, 28*28)
    metadata_batches = None
    coverage_batches = None
    for i in range(input_batch.size()[0]):
        input = input_batch[i]
        #input = input.to(torch.device('cuda:0'))
        scores_fp32, coverage_fp32 = model(input)
        scores_fp16, coverage_fp16 = model_fp16(input)
        
        scores_softmax_fp32 = F.log_softmax(scores_fp32.view(1, 1, -1), dim=-1)
        scores_softmax_fp16 = F.log_softmax(scores_fp16.view(1, 1, -1), dim=-1)
        coverage_fp32 = coverage_fp32.view(1, 1, -1)
        coverage_fp16 = coverage_fp16.view(1, 1, -1)
        
        # 2, mutations, 10(classes)
        if metadata_batches is not None:
            tmp = np.concatenate((coverage_fp32.cpu().numpy(), coverage_fp16.cpu().numpy()), axis = 0)
            tmp_softmax = np.concatenate((scores_softmax_fp32.cpu().numpy(), scores_softmax_fp16.cpu().numpy()), axis = 0)
            metadata_batches = np.concatenate((metadata_batches, tmp_softmax), axis = 1)
            coverage_batches = np.concatenate((coverage_batches, tmp), axis = 1)
        else:
            metadata_batches = np.concatenate((scores_softmax_fp32.cpu().numpy(), scores_softmax_fp16.cpu().numpy()), axis = 0)
            coverage_batches = np.concatenate((coverage_fp32.cpu().numpy(), coverage_fp16.cpu().numpy()), axis = 0)
    return coverage_batches, metadata_batches


def generate_fetch_function(model, model_fp16):
    def func(input):
        return fetch_function(model, model_fp16, input)
    return func

def generate_fetch_function_batched(model, model_fp16):
    def func(input):
        return fetch_function_batched(model, model_fp16, input)
    return func


def recent_sample_function(input_corpus):
    corpus = input_corpus.corpus
    reservoir = corpus[-5:] + [random.choice(corpus)]
    choice = random.choice(reservoir)
    return choice

def double_check(model, model_fp16, corpus_element, iterations = 10):
    # double check
    input = corpus_element.data[0] # class 'torch.Tensor'
    for _ in range(iterations):
        _, metadata_batch = fetch_function(model, model_fp16, input)
        fp32_scores = metadata_batch[0]
        fp16_scores = metadata_batch[1]
        prediction_16 = np.argmax(fp32_scores)
        prediction_32 = np.argmax(fp16_scores)
        if prediction_16 == prediction_32:
            return False # double check failed. spurious disagreement
    
    return True
