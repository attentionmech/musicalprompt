#!/usr/bin/env python3
"""
Random Walk LoopyFinder: hunts for "sticky" token cycles in a causal LM's transition graph
using probabilistic random walks instead of exhaustive search.
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict


class RandomWalkLoopyFinder:
    def __init__(
        self,
        model_name: str = "gpt2",
        top_k: int = 10,
        device: str = None,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.vocab_size = self.tokenizer.vocab_size
        self.top_k = top_k
        
    def get_next_token(self, token_id: int):
        """Sample next token based on probabilities."""
        input_ids = torch.tensor([[token_id]], device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0, -1]
            probs = F.softmax(logits, dim=-1)
        
        # Get top-k tokens and probabilities
        topk_probs, topk_ids = torch.topk(probs, self.top_k)
        
        # Convert to numpy for sampling
        topk_probs = topk_probs.cpu().numpy()
        topk_ids = topk_ids.cpu().numpy()
        
        # Normalize probabilities to sum to 1
        topk_probs = topk_probs / topk_probs.sum()
        
        # Sample next token
        next_token_idx = np.random.choice(len(topk_ids), p=topk_probs)
        next_token = topk_ids[next_token_idx]
        next_prob = topk_probs[next_token_idx]
        
        return next_token, next_prob
    
    def random_walk(self, start_token: int, max_steps: int = 1000):
        """Perform a random walk from start_token and detect cycles."""
        path = []
        token_positions = {}  # Keep track of where each token appears in the path
        
        current_token = start_token
        
        for step in range(max_steps):
            # Add current token to path
            path.append(current_token)
            
            # Check if we've seen this token before
            if current_token in token_positions:
                # Found a potential cycle
                previous_pos = token_positions[current_token]
                cycle = path[previous_pos:]
                return cycle
            
            # Record position of current token
            token_positions[current_token] = len(path) - 1
            
            # Sample next token
            next_token, _ = self.get_next_token(current_token)
            current_token = next_token
        
        # If we reach here, no cycle was detected within max_steps
        return []
    
    def calculate_cycle_stickiness(self, cycle):
        """Calculate how sticky a cycle is."""
        if not cycle:
            return 0.0
        
        # Calculate probability of staying in the cycle
        in_cycle_prob = 0.0
        total_transitions = len(cycle)
        
        for i in range(len(cycle)):
            current = cycle[i]
            next_idx = (i + 1) % len(cycle)
            target = cycle[next_idx]
            
            # Get all possible transitions from current token
            input_ids = torch.tensor([[current]], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids).logits[0, -1]
                probs = F.softmax(logits, dim=-1)
            
            # Get probability of transitioning to the target
            target_prob = probs[target].item()
            in_cycle_prob += target_prob
        
        # Average probability of staying in cycle
        avg_stickiness = in_cycle_prob / total_transitions
        return avg_stickiness
    
    def find_cycles(self, num_walks: int = 1000, walk_length: int = 100, top_n: int = 50, start_tokens: list = None):
        """Find cycles using multiple random walks."""
        cycles = defaultdict(float)
        
        # If no start tokens provided, sample randomly from vocab
        if not start_tokens:
            start_tokens = random.sample(range(self.vocab_size), min(num_walks, self.vocab_size))
        
        # Ensure we have enough start tokens
        if len(start_tokens) < num_walks:
            additional = random.sample(range(self.vocab_size), num_walks - len(start_tokens))
            start_tokens.extend(additional)
        
        # Perform random walks
        for i in tqdm(range(num_walks), desc="Performing random walks"):
            start_token = start_tokens[i % len(start_tokens)]
            cycle = self.random_walk(start_token, max_steps=walk_length)
            
            if cycle:
                # Convert cycle to a hashable tuple
                cycle_tuple = tuple(cycle)
                
                # Calculate stickiness if we haven't seen this cycle before
                if cycle_tuple not in cycles:
                    stickiness = self.calculate_cycle_stickiness(cycle)
                    cycles[cycle_tuple] = stickiness
        
        # Sort cycles by stickiness
        sorted_cycles = sorted(cycles.items(), key=lambda x: -x[1])
        
        # Convert to readable format
        results = []
        for cycle_tuple, stickiness in sorted_cycles[:top_n]:
            tokens = self.tokenizer.convert_ids_to_tokens(cycle_tuple)
            results.append((" ".join(tokens), stickiness))
        
        return results
    
    def find_cycles_with_sampling(self, num_samples: int = 1000, top_n: int = 50, temperature: float = 1.0):
        """Find cycles by sampling many starting points and performing walks."""
        # Strategy: 
        # 1. Sample from potentially interesting tokens (e.g., high-frequency tokens)
        # 2. Perform weighted random walks (using temperature to control randomness)
        # 3. Track and analyze discovered cycles
        
        # Get token frequency info (approximate by sampling model outputs)
        token_counts = self._sample_token_frequencies(n_samples=100)
        
        # Select start tokens with probability proportional to their frequency
        start_token_ids = self._sample_weighted(
            list(token_counts.keys()),
            [count**0.5 for count in token_counts.values()],  # Square root to flatten distribution
            num_samples
        )
        
        return self.find_cycles(num_walks=num_samples, start_tokens=start_token_ids, top_n=top_n)
    
    def _sample_token_frequencies(self, n_samples=100, seq_length=20):
        """Sample from the model to get token frequency information."""
        token_counts = defaultdict(int)
        
        # Start with common tokens
        start_tokens = [self.tokenizer.bos_token_id or 0]
        
        for _ in tqdm(range(n_samples), desc="Sampling token frequencies"):
            # Generate a sequence
            input_ids = torch.tensor([start_tokens], device=self.device)
            
            for _ in range(seq_length):
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    next_token_logits = outputs.logits[0, -1, :]
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    
                    # Sample from distribution
                    next_token = torch.multinomial(next_token_probs, 1).item()
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
                    
                    # Count token
                    token_counts[next_token] += 1
        
        return token_counts
    
    def _sample_weighted(self, items, weights, n):
        """Sample n items with probability proportional to weights."""
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()  # Normalize
        return np.random.choice(items, size=n, p=weights, replace=True).tolist()


def main():
    parser = argparse.ArgumentParser(description="Find sticky token cycles in a causal LM using random walks.")
    parser.add_argument("--model", type=str, default="openai-community/gpt2", help="HF model to scan (default: gpt2)")
    parser.add_argument("--top_k", type=int, default=10, help="number of tokens to consider at each step")
    parser.add_argument("--top_n", type=int, default=50, help="how many top cycles to print")
    parser.add_argument("--walks", type=int, default=1000, help="number of random walks to perform")
    parser.add_argument("--walk_length", type=int, default=100, help="maximum steps per random walk")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--sampling", action="store_true", help="use intelligent sampling of start tokens")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    finder = RandomWalkLoopyFinder(
        model_name=args.model,
        top_k=args.top_k,
    )
    
    if args.sampling:
        print(f"Finding sticky cycles using intelligent sampling ({args.walks} samples)...")
        results = finder.find_cycles_with_sampling(
            num_samples=args.walks,
            top_n=args.top_n
        )
    else:
        print(f"Finding sticky cycles using random walks ({args.walks} walks)...")
        results = finder.find_cycles(
            num_walks=args.walks,
            walk_length=args.walk_length,
            top_n=args.top_n
        )
    
    print(f"\nTop {min(len(results), args.top_n)} sticky cycles:")
    for seq, score in results:
        print(f"[{score:.4f}] {seq}")


if __name__ == "__main__":
    main()
