# GraphGen Complete Technical Guide

## Architecture & Algorithms

### **Core Workflow**
GraphGen follows a 4-step pipeline:
1. **Read & Split** → 2. **Extract KG** → 3. **Quiz & Judge** → 4. **Generate QA**

### **1. Knowledge Graph Construction**

**Algorithm**: Multi-turn LLM extraction with iterative refinement
- Uses structured prompts to extract entities and relationships from text chunks
- Implements a **loop mechanism** (max 3 iterations) where the LLM continues extracting until it says "no" to "Are there more entities/relationships?"
- **Merging strategy**: Combines duplicate entities/relationships using LLM-based similarity detection
- **Storage**: NetworkX graph with rich metadata (descriptions, source chunks, loss scores)

### **2. Knowledge Gap Detection (ECE Method)**

**Algorithm**: Expected Calibration Error for knowledge prioritization
- **Quiz Generation**: Creates positive/negative statement pairs from graph descriptions
- **Judgment**: Uses trainee model to predict "yes/no" with probability scores
- **Loss Calculation**: `yes_no_loss_entropy()` computes `-log(p)` where p is correct answer probability
- **Prioritization**: Higher loss = knowledge gap = higher generation priority

### **3. Graph Partitioning Strategies**

#### **ECE Method** (for atomic/aggregated/multi-hop)
```python
# Core algorithm: Breadth-first expansion with loss-based sampling
def _get_level_n_edges_by_max_width():
    start_nodes = {src, tgt}  # bidirectional if enabled
    while max_depth > 0 and max_extra_edges > 0:
        candidate_edges = get_neighbors(start_nodes)
        if len(candidates) >= max_extra_edges:
            # Sort by loss: max_loss, min_loss, or random
            selected = sort_by_loss(candidates)[:max_extra_edges]
            break
        # Continue expanding
        start_nodes = get_new_nodes(candidate_edges)
```

**Key Parameters**:
- `bidirectional`: Expand from both source and target nodes
- `edge_sampling`: `max_loss` (prioritize knowledge gaps), `min_loss`, `random`
- `max_depth`: How many hops to traverse
- `max_extra_edges`: Batch size limit
- `loss_strategy`: `only_edge` vs `both` (include node losses)

#### **Leiden Method** (for CoT)
```python
# Community detection using igraph + leidenalg
partition = find_partition(graph, ModularityVertexPartition, seed=42)
# Split large communities if max_size specified
if community_size > max_size:
    split_into_chunks(community, max_size)
```

### **4. QA Generation Modes**

#### **Atomic Mode**
- **Input**: Single edges + immediate neighbors
- **Output**: Simple factual QA pairs
- **Template**: Basic question generation from entity-relationship context

#### **Aggregated Mode** 
- **Input**: Multi-hop subgraphs (deeper traversal)
- **Output**: Complex QA requiring multiple facts
- **Algorithm**: Same ECE traversal but with larger `max_extra_edges`

#### **Multi-hop Mode**
- **Input**: Path-based subgraphs (shallower but wider)
- **Output**: Reasoning chains across multiple entities
- **Algorithm**: ECE with optimized depth/width balance

#### **Chain-of-Thought (CoT) Mode**
- **Input**: Leiden communities (semantically coherent clusters)
- **Output**: Reasoning templates + step-by-step answers
- **Algorithm**: 
  1. Community detection finds related entity clusters
  2. LLM designs reasoning template for each community
  3. LLM generates CoT answer following the template

### **5. Key Algorithms**

#### **Loss-Based Prioritization**
```python
def yes_no_loss_entropy(predictions, ground_truth):
    losses = []
    for pred, gt in zip(predictions, ground_truth):
        if pred.text == gt:
            losses.append(-math.log(pred.prob))  # Correct but uncertain
        else:
            losses.append(-math.log(1 - pred.prob))  # Wrong
    return sum(losses) / len(losses)
```

#### **Graph Traversal Strategy**
- **Breadth-first expansion** from seed edges
- **Loss-weighted sampling** prioritizes knowledge gaps
- **Token budget management** prevents context overflow
- **Bidirectional traversal** captures richer context

### **6. Technical Implementation**

**Concurrency**: Extensive use of `asyncio` with semaphores for rate limiting
**Storage**: Modular design with NetworkX graphs + JSON key-value stores
**LLM Integration**: OpenAI-compatible API with token counting and top-k sampling
**Prompt Engineering**: Language-aware templates (English/Chinese) with structured output parsing

### **7. Innovation Points**

1. **Knowledge-driven synthesis**: Uses graph structure to identify what to generate
2. **Loss-based prioritization**: Targets model knowledge gaps using calibration metrics  
3. **Multi-modal generation**: Different algorithms for different reasoning types
4. **Iterative extraction**: Self-improving KG construction through LLM loops
5. **Community-aware CoT**: Uses graph communities for coherent reasoning chains

The system is essentially a **knowledge graph-guided synthetic data generator** that uses probabilistic loss metrics to identify and target the most valuable knowledge gaps for LLM training.

## Previous Session Summary

### **Molecular Network QA Generation Setup**
* GraphGen repository setup for molecular network QA generation with 18,950 nodes and 55,711 edges
* Custom molecular network integration using optimized_molecular_network.graphml file
* LiteLLM proxy configuration for AWS Bedrock models (8 models: 6 Llama + 2 Claude)
* Graph preparation requirements: adding description attributes to nodes and edges
* Four QA generation modes: atomic, chain-of-thought (CoT), aggregated, and multi-hop
* SMILES codes implementation as node identifiers instead of arbitrary numbers
* Data format options: Alpaca, ShareGPT, and ChatML for different fine-tuning needs
* Partition method requirements: leiden for CoT, ECE for aggregated/multi-hop modes
* Troubleshooting LiteLLM parameter compatibility issues with Bedrock

### **AWS Bedrock Integration**
* LiteLLM configuration with drop_params: true for Bedrock compatibility
* ECE partition method parameters: bidirectional, edge_sampling, expand_method, isolated_node_strategy, max_depth, max_extra_edges, max_tokens, loss_strategy
* Leiden partition method parameters: max_size, use_lcc, random_seed
* Data format structures for Alpaca (instruction/input/output), ShareGPT (conversations), and ChatML (messages)
* GraphGen command syntax: python3 -m graphgen.generate --config_file config.yaml --output_dir output_dir/
* Graph preparation code for adding molecular descriptions to nodes and edges
* SMILES-to-node-ID mapping implementation using NetworkX

### **Key Technical Insights**
* GraphGen requires both node and edge description attributes for proper QA generation
* Different QA modes have specific partition method requirements that cannot be interchanged
* LiteLLM drop_params setting is critical for AWS Bedrock compatibility to avoid logprobs errors
* SMILES codes as node identifiers produce more meaningful QA pairs than arbitrary numbers
* ChatML format is optimal for modern Llama fine-tuning applications
* ECE method requires all 8 parameters to function properly for aggregated and multi-hop modes
* Molecular network successfully generates chemistry-specific QA pairs suitable for domain fine-tuning

### **Production Configuration**
**Working LiteLLM + AWS Bedrock Setup:**
- **8 total models** accessible via OpenAI-compatible API
- **6 Llama models**: llama-3-8b, llama-3-70b, llama-3-2-3b, llama-3-1-70b, llama-3-2-90b, llama-3-3-70b
- **2 Claude models**: claude-3-7-sonnet, claude-sonnet-4
- **Cross-region inference** working for all newer models
- **Direct access** for standard models
- **Cost optimization** - Direct AWS pricing vs. third-party markup
- **Enterprise compliance** - AWS security and governance
