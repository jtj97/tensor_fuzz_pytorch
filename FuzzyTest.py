import torch
import argparse
import fuzzylib.utils, fuzzylib.Fuzzer, fuzzylib.corpus
from torchvision import datasets, transforms
import random
import numpy as np
import time
import os

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", default='./data/mnist/ffn_4layer_not_norm.pth', type=str, help="path of model checkpoint")
    parser.add_argument("-i", "--iterations", default=50000, type=int, help="iterations")
    parser.add_argument("-m", "--mutations", default=100, type=int, help="mutations per item")
    parser.add_argument("-s", "--scaling", default=1.0, type=float, help="scaling factor of a_min and a_max list")
    parser.add_argument("-t", "--threshold", default=0.1, type=float, help="distance below which we consider something new coverage")
    parser.add_argument("-verbose", "--verbose", default=1, type=int, help="verbose level")
    parser.add_argument("-amin", "--amin", default=0, type=float, help="a min list to clip mutation elements E.g. 1,2,3,4,5")
    parser.add_argument("-amax", "--amax", default=1, type=float, help="a max list to clip mutation elements E.g. 1,2,3,4,5")
    parser.add_argument("-sig", "--sigma", default=0.01, type=float, help="white noise sigma")
    parser.add_argument("-cov", "--coverage_mode", default=1, type=int, help="use coverage guided fuzzing(0:not use, 1:use)")
    parser.add_argument("-mi", "--max_items", default=50, type=int, help="use coverage guided fuzzing")
    return parser.parse_args()

def print_parameters(args):
    print(f"iterations: {args.iterations}")
    print(f"mutations : {args.mutations}")
    print(f"scaling   : {args.scaling}")
    print(f"threshold : {args.threshold}")
    print(f"sigma     : {args.sigma}")

# larger model to test
def main():
    args = parseArgs()
    
    random.seed(42)
    np.random.seed(42)
    
    print_parameters(args)
    
    device = torch.device('cuda:0')
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=False, 
                       transform=transforms.Compose( [transforms.ToTensor(),
#                                                      transforms.Normalize((0.1307,), (0.3081,))
                                                      ] 
                                                    )
                       ),
        batch_size=1, shuffle=True)

    model = torch.load(args.checkpoint)
    model_fp16  = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.float16)
    model.eval()
    model_fp16.eval()
    #model_fp16 = torch.load(args.checkpoint).half()

    #model.to(device)
    #model_fp16.to(device)
    
    # function:
    coverage_function = fuzzylib.utils.coverage_function
    coverage_function_batched = fuzzylib.utils.coverage_function_batched
    metadata_function = fuzzylib.utils.metadata_function
    objective_function = fuzzylib.utils.objective_function
    mutation_function = fuzzylib.utils.mutation_function
    fetch_function = fuzzylib.utils.generate_fetch_function(model, model_fp16)
    fetch_function_batched = fuzzylib.utils.generate_fetch_function_batched(model, model_fp16)
    sample_function = fuzzylib.utils.recent_sample_function
    
    with torch.no_grad():
        input = test_loader.dataset[0][0]
        input = input.view(input.size()[0], -1)

        seed_corpus = fuzzylib.corpus.seed_corpus_from_tensor(input, coverage_function, metadata_function, fetch_function)
        
        corpus = fuzzylib.corpus.InputCorpus(seed_corpus, sample_function, args.threshold, "kdtree", args.verbose)
        
        fuzzer = fuzzylib.Fuzzer.Fuzzer(
                corpus,
                coverage_function_batched,
                metadata_function,
                objective_function,
                mutation_function,
                fetch_function_batched,
                args.amin, 
                args.amax,
                args.mutations,
                args.sigma,
                args.coverage_mode,
                args.max_items,
                model,
                model_fp16
            )

        results = fuzzer.loop(args.iterations)
        if results:
            print(f"Fuzzing succeeded. Generated {len(results)} tensors caused surfaces disagreements between FP32 and FP16 models")
            tmp_str = "coverage guided" if args.coverage_mode != 0 else "coverage not guided"
            print(f"corpus_size({tmp_str}) {len(corpus.corpus)} \nmutations_processed({tmp_str}): {corpus.mutations_processed}")
            for i in range(len(results)):
                result = results[i]
                if args.verbose > 1:
                    print(f"mutated tensor: {result.data}\n")
                    print(f"Generations to make satisfying element: {result.oldest_ancestor()[1]}.")
                if args.verbose > 2:
                    fuzzylib.utils.save_results(result, f"images/{i}.jpg")
        else:
            print("Fuzzing failed to satisfy objective function.")

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
        
if __name__ == '__main__':
    setup_seed()
    begin = time.time()
    main()
    end = time.time()
    print(f"Fuzzy test time: {end - begin} s")
