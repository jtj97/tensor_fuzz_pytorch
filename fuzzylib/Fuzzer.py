import fuzzylib.corpus as corpus
import torch
import random
import numpy as np
from fuzzylib.utils import double_check

random.seed(0)
np.random.seed(0)

class Fuzzer:
    def __init__(
        self,
        corpus,
        coverage_function,
        metadata_function,
        objective_function,
        mutation_function,
        fetch_function,
        algo_num,
        a_min_list, 
        a_max_list,
        mutations_per_corpus_item,
        sigma,
        coverage_mode,
        max_items,
        model,
        model_fp16,
        seed
    ):
        self.corpus = corpus
        self.coverage_function = coverage_function
        self.metadata_function = metadata_function
        self.objective_function = objective_function
        self.mutation_function = mutation_function
        self.fetch_function = fetch_function
        self.algo_num = algo_num
        self.a_min_list = a_min_list
        self.a_max_list = a_max_list
        self.mutations_per_corpus_item = mutations_per_corpus_item
        self.sigma = sigma
        self.coverage_mode = coverage_mode
        self.max_items = max_items
        self.model = model
        self.model_fp16 = model_fp16
        self.seed = seed
    
    def loop(self, iterations):
        res = []
        last_corpus_num = 0
        iter_threshold = 5
        for iteration in range(iterations):
            if iteration % 100 == 0:
                print(f"fuzzing iteration: {iteration}")
            if iteration % iter_threshold == 0:
                # adjust threshold
                added_ratio = float(len(self.corpus.corpus) - last_corpus_num) / float(self.mutations_per_corpus_item * iter_threshold)
                last_corpus_num = len(self.corpus.corpus)
                if added_ratio >= 0.30:
                    self.corpus.updater.threshold *= 1.5
                elif added_ratio <= 0.20:
                    self.corpus.updater.threshold *= 0.75
            parent = self.corpus.sample_input()

            # Get a mutated batch for each input tensor
            mutated_data_batches = self.mutation_function(parent, self.mutations_per_corpus_item, 
                                                          self.a_min_list,
                                                          self.a_max_list,
                                                          self.sigma,
                                                          self.seed)
            mutated_data_batches_tensor = torch.Tensor(mutated_data_batches)
            
            # Grab the coverage and metadata
            coverage_batches, metadata_batches = self.fetch_function(mutated_data_batches_tensor, self.algo_num)

            # Get the coverage - one from each batch element
            mutated_coverage_list = self.coverage_function(coverage_batches)

            # Get the metadata objects - one from each batch element
            mutated_metadata_list = self.metadata_function(metadata_batches) # MUTATIONS_PER_CORPUS_ITEM, 2, 9
            
            # Check for new coverage and create new corpus elements if necessary.
            for idx in range(len(mutated_coverage_list)):
                new_element = corpus.CorpusElement(
#                    [batch for batch in mutated_data_batches_tensor],
                    [mutated_data_batches_tensor[idx]],
                    mutated_metadata_list[idx],
                    mutated_coverage_list[idx],
                    parent,
                )
                if self.coverage_mode != 0:
                    old_corpus_size = len(self.corpus.corpus)
                    self.corpus.maybe_add_to_corpus(new_element)
                    if self.objective_function(new_element): # added a new element:
                        res.append(new_element)
                else:
                    self.corpus.mutations_processed += 1
                    self.corpus.corpus.append(new_element)
                    if self.objective_function(new_element):
                        res.append(new_element)
                
                if len(res) >= self.max_items:
                    return res
        return res
