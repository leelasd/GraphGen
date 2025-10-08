#!/usr/bin/env python3
"""
Chemistry Curator using Real LogD Data
Generates rich explanations and reasoning for actual SMILES-LogD pairs.
"""

import pandas as pd
from typing import Dict
from datasets import Dataset
from bespokelabs import curator

class LogDExplainer(curator.LLM):
    """Generates detailed explanations for real SMILES-LogD pairs."""
    
    def prompt(self, input: Dict) -> str:
        smiles = input['smiles']
        logd = input['logd']
        
        return f"""You are an expert medicinal chemist. Given this molecule:

SMILES: {smiles}
Experimental LogD: {logd}

Provide a detailed explanation of why this molecule has this LogD value. Include:
1. Structural analysis of key functional groups
2. Hydrophobic vs hydrophilic contributions  
3. Ionization effects at physiological pH
4. Molecular size and flexibility factors
5. Comparison to similar compounds

Format your response as:
Explanation: [detailed reasoning]
Key Groups: [list main functional groups]
Prediction Confidence: [how confident you'd be predicting this value]"""

    def parse(self, input: Dict, response: str) -> Dict:
        lines = response.strip().split('\n')
        explanation = ""
        key_groups = ""
        confidence = ""
        
        for line in lines:
            if line.startswith('Explanation:'):
                explanation = line.split(':', 1)[1].strip()
            elif line.startswith('Key Groups:'):
                key_groups = line.split(':', 1)[1].strip()
            elif line.startswith('Prediction Confidence:'):
                confidence = line.split(':', 1)[1].strip()
        
        return {
            "task": "logd_explanation",
            "smiles": input['smiles'],
            "experimental_logd": input['logd'],
            "explanation": explanation,
            "key_groups": key_groups,
            "confidence": confidence,
            "instruction": f"Explain why the molecule with SMILES {input['smiles']} has LogD = {input['logd']}",
            "output": f"This molecule has LogD = {input['logd']} because: {explanation}"
        }

class LogDComparator(curator.LLM):
    """Generates comparative analyses between molecules."""
    
    def prompt(self, input: Dict) -> str:
        mol1_smiles = input['mol1_smiles']
        mol1_logd = input['mol1_logd']
        mol2_smiles = input['mol2_smiles'] 
        mol2_logd = input['mol2_logd']
        
        return f"""Compare these two molecules and explain their LogD difference:

Molecule 1: {mol1_smiles} (LogD = {mol1_logd})
Molecule 2: {mol2_smiles} (LogD = {mol2_logd})

Analyze:
1. Structural differences between the molecules
2. How these differences affect lipophilicity
3. Which functional groups drive the LogD difference
4. Predict which would have better drug-like properties

Format as:
Comparison: [detailed analysis]
Key Difference: [main structural factor]
Drug Design Insight: [practical implication]"""

    def parse(self, input: Dict, response: str) -> Dict:
        lines = response.strip().split('\n')
        comparison = ""
        key_diff = ""
        insight = ""
        
        for line in lines:
            if line.startswith('Comparison:'):
                comparison = line.split(':', 1)[1].strip()
            elif line.startswith('Key Difference:'):
                key_diff = line.split(':', 1)[1].strip()
            elif line.startswith('Drug Design Insight:'):
                insight = line.split(':', 1)[1].strip()
        
        return {
            "task": "logd_comparison",
            "mol1_smiles": input['mol1_smiles'],
            "mol1_logd": input['mol1_logd'],
            "mol2_smiles": input['mol2_smiles'],
            "mol2_logd": input['mol2_logd'],
            "comparison": comparison,
            "key_difference": key_diff,
            "insight": insight,
            "instruction": f"Compare LogD values: {input['mol1_smiles']} (LogD={input['mol1_logd']}) vs {input['mol2_smiles']} (LogD={input['mol2_logd']})",
            "output": f"Comparison: {comparison}. Key difference: {key_diff}. Insight: {insight}"
        }

class LogDPredictor(curator.LLM):
    """Generates prediction reasoning for SMILES."""
    
    def prompt(self, input: Dict) -> str:
        smiles = input['smiles']
        
        return f"""You are predicting LogD for this molecule: {smiles}

Walk through your prediction process step-by-step:
1. Identify all functional groups
2. Estimate hydrophobic contributions
3. Estimate hydrophilic contributions  
4. Consider ionization at pH 7.4
5. Make final LogD prediction with reasoning

Format as:
Functional Groups: [list groups]
Hydrophobic Score: [estimate]
Hydrophilic Score: [estimate]
pH Effects: [ionization analysis]
Predicted LogD: [your prediction]
Reasoning: [step-by-step logic]"""

    def parse(self, input: Dict, response: str) -> Dict:
        lines = response.strip().split('\n')
        groups = ""
        hydrophobic = ""
        hydrophilic = ""
        ph_effects = ""
        predicted = ""
        reasoning = ""
        
        for line in lines:
            if line.startswith('Functional Groups:'):
                groups = line.split(':', 1)[1].strip()
            elif line.startswith('Hydrophobic Score:'):
                hydrophobic = line.split(':', 1)[1].strip()
            elif line.startswith('Hydrophilic Score:'):
                hydrophilic = line.split(':', 1)[1].strip()
            elif line.startswith('pH Effects:'):
                ph_effects = line.split(':', 1)[1].strip()
            elif line.startswith('Predicted LogD:'):
                predicted = line.split(':', 1)[1].strip()
            elif line.startswith('Reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
        
        return {
            "task": "logd_prediction",
            "smiles": input['smiles'],
            "actual_logd": input.get('logd', 'unknown'),
            "functional_groups": groups,
            "predicted_logd": predicted,
            "reasoning": reasoning,
            "instruction": f"Predict LogD for {input['smiles']} and explain your reasoning step-by-step",
            "output": f"Predicted LogD: {predicted}. Reasoning: {reasoning}"
        }

def load_logd_data():
    """Load top 20 rows from CSV."""
    df = pd.read_csv('/Users/ldodda/Documents/Codes/GraphGen/test_logd.csv')
    df = df.head(20)
    
    # Clean and prepare data
    data = []
    for _, row in df.iterrows():
        data.append({
            'smiles': row['canonical_smiles'],
            'logd': float(row['standard_value']),
            'compound_id': row['compound_chembl_id']
        })
    
    return data

def create_comparison_pairs(data):
    """Create pairs for comparison analysis."""
    pairs = []
    for i in range(0, len(data)-1, 2):
        pairs.append({
            'mol1_smiles': data[i]['smiles'],
            'mol1_logd': data[i]['logd'],
            'mol2_smiles': data[i+1]['smiles'],
            'mol2_logd': data[i+1]['logd']
        })
    return pairs

def main():
    """Generate chemistry training data using real LogD values."""
    
    print("🧪 Loading real LogD data...")
    logd_data = load_logd_data()
    print(f"📊 Loaded {len(logd_data)} SMILES-LogD pairs")
    
    # Model configuration
    model_config = {
        "model_name": "llama-3-3-70b",
        "backend": "openai", 
        "backend_params": {
            "base_url": "http://localhost:4000/v1",
            "api_key": "bedrock-graphgen-2024",
            "max_requests_per_minute": 20,
            "max_tokens_per_minute": 30000,
            "require_all_responses": False
        }
    }
    
    all_results = []
    
    # 1. Generate explanations for each molecule
    print("\n📝 Generating LogD explanations...")
    explainer = LogDExplainer(**model_config)
    explanations = explainer(Dataset.from_list(logd_data))
    all_results.extend(explanations.dataset.to_list())
    print(f"✅ Generated {len(explanations.dataset)} explanations")
    
    # 2. Generate predictions (hide actual LogD)
    print("\n🔮 Generating prediction reasoning...")
    predictor = LogDPredictor(**model_config)
    prediction_data = [{'smiles': item['smiles']} for item in logd_data[:10]]  # First 10
    predictions = predictor(Dataset.from_list(prediction_data))
    all_results.extend(predictions.dataset.to_list())
    print(f"✅ Generated {len(predictions.dataset)} predictions")
    
    # 3. Generate comparisons
    print("\n⚖️ Generating molecular comparisons...")
    comparison_pairs = create_comparison_pairs(logd_data[:10])  # First 10 for 5 pairs
    comparator = LogDComparator(**model_config)
    comparisons = comparator(Dataset.from_list(comparison_pairs))
    all_results.extend(comparisons.dataset.to_list())
    print(f"✅ Generated {len(comparisons.dataset)} comparisons")
    
    # Save results
    output_file = "/Users/ldodda/Documents/Codes/GraphGen/chemistry_real_data_training.jsonl"
    import json
    with open(output_file, 'w') as f:
        for item in all_results:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Generated {len(all_results)} total training examples")
    print(f"💾 Saved to: {output_file}")
    
    # Show samples
    print("\n📋 Sample Outputs:")
    
    # Explanation sample
    explanations = [x for x in all_results if x['task'] == 'logd_explanation']
    if explanations:
        sample = explanations[0]
        print(f"\n📝 Explanation Sample:")
        print(f"SMILES: {sample['smiles']}")
        print(f"LogD: {sample['experimental_logd']}")
        print(f"Explanation: {sample['explanation'][:150]}...")
    
    # Prediction sample  
    predictions = [x for x in all_results if x['task'] == 'logd_prediction']
    if predictions:
        sample = predictions[0]
        print(f"\n🔮 Prediction Sample:")
        print(f"Input: {sample['instruction'][:100]}...")
        print(f"Output: {sample['output'][:150]}...")

if __name__ == "__main__":
    main()
