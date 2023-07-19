import numpy as np
import pyflann
import time

_BUFFER_SIZE = 64

class CorpusElement(object):
    def __init__(self, data, metadata, coverage, parent):
        """Inits the object.
        Args:
          data: a list of Tensor representing the mutated data.
          metadata: arbitrary python object to be used by the fuzzer for e.g.
            computing the objective function during the fuzzing loop. For our test, we use the list [FP32result, FP16result] as metadata
          coverage: an arbitrary hashable python object that guides fuzzing process.
          parent: a reference to the CorpusElement this element is a mutation of.
          iteration: the fuzzing iteration (number of CorpusElements sampled to
            mutate) that this CorpusElement was created at.
        Returns:
          Initialized object.
        """
        self.data = data
        self.metadata = metadata
        self.parent = parent
        self.coverage = coverage
    
    def oldest_ancestor(self):
        """Returns the least recently created ancestor of this corpus item."""
        current_element = self
        generations = 0
        while current_element.parent is not None:
            current_element = current_element.parent
            generations += 1
        return current_element, generations
    
class Updater(object):
    """Class holding the state of the update function."""
    
    def __init__(self, threshold, algorithm, verbose):
        """Inits the object.
        Args:
          threshold: Float distance at which coverage is considered new.
          algorithm: Algorithm used to get approximate neighbors.
        Returns:
          Initialized object.
        """
        self.flann = pyflann.FLANN()
        self.threshold = threshold
        self.algorithm = algorithm
        self.corpus_buffer = []
        self.lookup_array = []
        self.verbose = verbose
    
    def build_index_and_flush_buffer(self, corpus_object):
        """Builds the nearest neighbor index and flushes buffer of examples.
        This method first empties the buffer of examples that have not yet
        been added to the nearest neighbor index.
        Then it rebuilds that index using the contents of the whole corpus.
        Args:
          corpus_object: InputCorpus object.
        """
        self.corpus_buffer[:] = []
        self.lookup_array = np.array(
            [element.coverage for element in corpus_object.corpus]
        )
        self.flann.build_index(self.lookup_array, algorithm=self.algorithm)

    def update_function(self, corpus_object, element):
        """Checks if coverage is new and updates corpus if so.
        The updater maintains both a corpus_buffer and a lookup_array.
        When the corpus_buffer reaches a certain size, we empty it out
        and rebuild the nearest neighbor index.
        Whenever we check for neighbors, we get exact neighbors from the
        buffer and approximate neighbors from the index.
        This stops us from building the index too frequently.
        FLANN supports incremental additions to the index, but they require
        periodic rebalancing anyway, and so far this method seems to be
        working OK.
        Args:
          corpus_object: InputCorpus object.
          element: CorpusElement object to maybe be added to the corpus.
        """
        if corpus_object.corpus is None:
            corpus_object.corpus = [element]
            self.build_index_and_flush_buffer(corpus_object)
        else:
            element.coverage = np.squeeze(element.coverage)
            _, approx_distances = self.flann.nn_index(
                np.array([element.coverage]), 1, algorithm=self.algorithm
            )
            exact_distances = [
                np.sum(np.square(element.coverage - buffer_elt))
                for buffer_elt in self.corpus_buffer
            ]
            nearest_distance = min(exact_distances + approx_distances.tolist())
            if nearest_distance > self.threshold:
              if self.verbose > 0:
                print(f"corpus_size {len(corpus_object.corpus)} mutations_processed {corpus_object.mutations_processed}",)
              if self.verbose > 1:
                print(f"coverage: {element.coverage}, metadata: {element.metadata}",)
              corpus_object.corpus.append(element)
              self.corpus_buffer.append(element.coverage)
              if len(self.corpus_buffer) >= _BUFFER_SIZE:
                  self.build_index_and_flush_buffer(corpus_object)

class InputCorpus(object):
    """Class that holds inputs and associated coverage."""
    
    def __init__(self, seed_corpus, sample_function, threshold, algorithm, verbose):
        """Init the class.
        Args:
          seed_corpus: a list of Tensors, one for each input tensor in the
            fuzzing process.
          sample_function: a function that looks at the whole current corpus and
            samples the next element to mutate in the fuzzing loop.
        Returns:
          Initialized object.
        """
        self.mutations_processed = 0
        self.corpus = None
        self.sample_function = sample_function
        self.start_time = time.time()
        self.current_time = time.time()
        self.log_time = time.time()
        self.updater = Updater(threshold, algorithm, verbose)

        for corpus_element in seed_corpus:
            self.maybe_add_to_corpus(corpus_element)
            
    def maybe_add_to_corpus(self, element):
        """Adds item to corpus if it exercises new coverage."""
        self.updater.update_function(self, element)
        self.mutations_processed += 1
        current_time = time.time()
        if current_time - self.log_time > 10:
            self.log_time = current_time
            print(
                f"mutations_per_second: {float(self.mutations_processed) / (current_time - self.start_time)}",
            )
            
    def sample_input(self):
        """Grabs new input from corpus according to sample_function."""
        choice = self.sample_function(self)
        return choice
    
def seed_corpus_from_tensor(tensor, coverage_function, metadata_function, fetch_function):
    seed_corpus = []
    input = tensor
    coverage_data, metadata = fetch_function(input)
    coverage_list = coverage_function(coverage_data)
    metadata_list = metadata_function(metadata)
    new_element = CorpusElement(
        [input], metadata_list[0], coverage_list[0], None
    )
    seed_corpus.append(new_element)
    return seed_corpus
