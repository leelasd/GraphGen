"""
TxLlama Drug Discovery Dataset Generators using Curator + LiteLLM

This module implements specialized generators for creating synthetic datasets
to train Llama models for drug discovery tasks, following the TxGemma approach.

Key capabilities:
1. Molecular property prediction (LogD, solubility, toxicity, etc.)
2. Molecular generation and optimization
3. Chemical reasoning and explanation
4. SMILES manipulation and analysis
5. Structure-activity relationship (SAR) reasoning
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datasets import Dataset
from bespokelabs import curator


# ============================================================================
# Pydantic Models for Structured Outputs
# ============================================================================

class MolecularProperty(BaseModel):
    """Molecular property prediction with confidence and reasoning."""
    property_name: str = Field(description="Name of the molecular property")
    predicted_value: float = Field(description="Predicted numerical value")
    confidence: str = Field(description="Confidence level: high, medium, low")
    reasoning: str = Field(description="Chemical reasoning for the prediction")
    key_structural_features: List[str] = Field(description="Important structural features affecting the property")


class MolecularGeneration(BaseModel):
    """Generated molecule with properties and explanation."""
    smiles: str = Field(description="Generated SMILES string")
    rationale: str = Field(description="Explanation of design choices")
    predicted_properties: Dict[str, float] = Field(description="Expected property values")
    synthetic_accessibility: str = Field(description="Ease of synthesis: easy, moderate, difficult")


class ChemicalReasoning(BaseModel):
    """Chemical reasoning response with step-by-step analysis."""
    analysis_steps: List[str] = Field(description="Step-by-step chemical analysis")
    conclusion: str = Field(description="Final conclusion or answer")
    confidence: str = Field(description="Confidence in the reasoning")
    relevant_concepts: List[str] = Field(description="Key chemical concepts used")


class MolecularOptimization(BaseModel):
    """Molecular optimization suggestions."""
    original_smiles: str = Field(description="Original molecule SMILES")
    optimized_molecules: List[Dict[str, Union[str, float]]] = Field(
        description="List of optimized molecules with SMILES and predicted improvements"
    )
    optimization_strategy: str = Field(description="Strategy used for optimization")
    trade_offs: str = Field(description="Discussion of property trade-offs")


# ============================================================================
# Core Dataset Generators
# ============================================================================

class PropertyPredictor(curator.LLM):
    """Generates property prediction tasks with reasoning."""
    
    response_format = MolecularProperty
    
    def prompt(self, input: Dict) -> str:
        smiles = input['canonical_smiles']
        property_type = input.get('standard_type', 'LogD')
        
        return f"""You are an expert medicinal chemist. Analyze the following molecule and predict its {property_type}.

Molecule (SMILES): {smiles}

Provide a detailed prediction including:
1. The predicted {property_type} value
2. Your confidence level
3. Chemical reasoning based on structural features
4. Key structural features that influence this property

Consider factors like:
- Lipophilicity and hydrophilicity
- Molecular weight and size
- Functional groups present
- Aromatic systems
- Hydrogen bonding potential
- Charge distribution"""

    def parse(self, input: Dict, response: MolecularProperty) -> Dict:
        return {
            "smiles": input['canonical_smiles'],
            "actual_value": input.get('standard_value'),
            "predicted_value": response.predicted_value,
            "property_type": input.get('standard_type', 'LogD'),
            "confidence": response.confidence,
            "reasoning": response.reasoning,
            "key_features": response.key_structural_features,
            "compound_id": input.get('compound_chembl_id', '')
        }


class MolecularGenerator(curator.LLM):
    """Generates molecules with specific properties."""
    
    response_format = MolecularGeneration
    
    def prompt(self, input: Dict) -> str:
        target_property = input['target_property']
        target_value = input['target_value']
        constraints = input.get('constraints', 'drug-like properties')
        
        return f"""You are an expert medicinal chemist specializing in molecular design. 

Design a molecule with the following specifications:
- Target {target_property}: {target_value}
- Additional constraints: {constraints}

Requirements:
1. Provide a valid SMILES string
2. Explain your design rationale
3. Predict key molecular properties
4. Assess synthetic accessibility

Consider:
- Lipinski's Rule of Five
- ADMET properties
- Synthetic feasibility
- Structure-activity relationships"""

    def parse(self, input: Dict, response: MolecularGeneration) -> Dict:
        return {
            "target_property": input['target_property'],
            "target_value": input['target_value'],
            "generated_smiles": response.smiles,
            "design_rationale": response.rationale,
            "predicted_properties": response.predicted_properties,
            "synthetic_accessibility": response.synthetic_accessibility,
            "constraints": input.get('constraints', '')
        }


class ChemicalReasoningGenerator(curator.LLM):
    """Generates chemical reasoning tasks and explanations."""
    
    response_format = ChemicalReasoning
    
    def prompt(self, input: Dict) -> str:
        question = input['question']
        context = input.get('context', '')
        
        prompt = f"""You are an expert medicinal chemist. Answer the following question with detailed chemical reasoning.

Question: {question}"""
        
        if context:
            prompt += f"\n\nContext: {context}"
            
        prompt += """

Provide:
1. Step-by-step analysis
2. Clear conclusion
3. Confidence assessment
4. Relevant chemical concepts

Use your knowledge of:
- Organic chemistry principles
- Pharmacokinetics and pharmacodynamics
- Structure-activity relationships
- Medicinal chemistry best practices"""

        return prompt

    def parse(self, input: Dict, response: ChemicalReasoning) -> Dict:
        return {
            "question": input['question'],
            "context": input.get('context', ''),
            "analysis_steps": response.analysis_steps,
            "conclusion": response.conclusion,
            "confidence": response.confidence,
            "concepts": response.relevant_concepts
        }


class MolecularOptimizer(curator.LLM):
    """Generates molecular optimization tasks."""
    
    response_format = MolecularOptimization
    
    def prompt(self, input: Dict) -> str:
        smiles = input['smiles']
        optimization_goal = input['optimization_goal']
        current_value = input.get('current_value', 'unknown')
        
        return f"""You are an expert medicinal chemist specializing in lead optimization.

Starting molecule (SMILES): {smiles}
Current {optimization_goal}: {current_value}
Optimization goal: Improve {optimization_goal}

Provide:
1. 3-5 optimized molecular structures (SMILES)
2. Predicted improvements for each
3. Optimization strategy explanation
4. Discussion of potential trade-offs

Consider:
- Structure-activity relationships
- Metabolic stability
- Selectivity
- Synthetic accessibility
- Off-target effects"""

    def parse(self, input: Dict, response: MolecularOptimization) -> Dict:
        return {
            "original_smiles": input['smiles'],
            "optimization_goal": input['optimization_goal'],
            "current_value": input.get('current_value'),
            "optimized_molecules": response.optimized_molecules,
            "strategy": response.optimization_strategy,
            "trade_offs": response.trade_offs
        }


# ============================================================================
# Dataset Creation Functions
# ============================================================================

def create_property_prediction_dataset(
    csv_path: str,
    model_name: str = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    sample_size: Optional[int] = None
) -> Dataset:
    """Create property prediction dataset from molecular data CSV."""
    
    # Load and prepare data
    df = pd.read_csv(csv_path)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)))
    
    # Convert to dataset format
    dataset = Dataset.from_pandas(df)
    
    # Initialize generator
    predictor = PropertyPredictor(
        model_name=model_name,
        backend="litellm",
        backend_params={
            "max_requests_per_minute": 500,  # Conservative for Bedrock
            "max_tokens_per_minute": 1_000_000,
            "aws_region_name": "us-east-1"
        }
    )
    
    # Generate predictions with reasoning
    results = predictor(dataset)
    return results


def create_molecular_generation_dataset(
    property_targets: List[Dict],
    model_name: str = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
) -> Dataset:
    """Create molecular generation dataset."""
    
    # Prepare generation tasks
    generation_data = []
    for target in property_targets:
        generation_data.append({
            "target_property": target["property"],
            "target_value": target["value"],
            "constraints": target.get("constraints", "drug-like, orally bioavailable")
        })
    
    dataset = Dataset.from_list(generation_data)
    
    # Initialize generator
    generator = MolecularGenerator(
        model_name=model_name,
        backend="litellm"
    )
    
    results = generator(dataset)
    return results


def create_chemical_reasoning_dataset(
    questions: List[Dict],
    model_name: str = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
) -> Dataset:
    """Create chemical reasoning dataset."""
    
    dataset = Dataset.from_list(questions)
    
    # Initialize reasoning generator
    reasoner = ChemicalReasoningGenerator(
        model_name=model_name,
        backend="litellm"
    )
    
    results = reasoner(dataset)
    return results


def create_optimization_dataset(
    molecules_data: List[Dict],
    model_name: str = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
) -> Dataset:
    """Create molecular optimization dataset."""
    
    dataset = Dataset.from_list(molecules_data)
    
    # Initialize optimizer
    optimizer = MolecularOptimizer(
        model_name=model_name,
        backend="litellm"
    )
    
    results = optimizer(dataset)
    return results


# ============================================================================
# Example Usage and Demo Functions
# ============================================================================

def demo_property_prediction(csv_path: str):
    """Demo property prediction generation."""
    print("🧪 Generating Property Prediction Dataset...")
    
    dataset = create_property_prediction_dataset(
        csv_path=csv_path,
        sample_size=5  # Small sample for demo
    )
    
    print("\n📊 Sample Results:")
    df = dataset.to_pandas()
    for _, row in df.head(3).iterrows():
        print(f"\nSMILES: {row['smiles']}")
        print(f"Actual {row['property_type']}: {row['actual_value']}")
        print(f"Predicted {row['property_type']}: {row['predicted_value']}")
        print(f"Confidence: {row['confidence']}")
        print(f"Reasoning: {row['reasoning'][:200]}...")


def demo_molecular_generation():
    """Demo molecular generation."""
    print("🔬 Generating Molecular Design Dataset...")
    
    targets = [
        {"property": "LogD", "value": "2.0-3.0", "constraints": "CNS penetrant, MW < 400"},
        {"property": "solubility", "value": "> 100 μM", "constraints": "oral bioavailability"},
        {"property": "hERG IC50", "value": "> 10 μM", "constraints": "cardiac safety"}
    ]
    
    dataset = create_molecular_generation_dataset(targets)
    
    print("\n🎯 Sample Generated Molecules:")
    df = dataset.to_pandas()
    for _, row in df.iterrows():
        print(f"\nTarget: {row['target_property']} = {row['target_value']}")
        print(f"Generated SMILES: {row['generated_smiles']}")
        print(f"Rationale: {row['design_rationale'][:200]}...")


def demo_chemical_reasoning():
    """Demo chemical reasoning generation."""
    print("🧠 Generating Chemical Reasoning Dataset...")
    
    questions = [
        {
            "question": "Why do benzimidazole derivatives often show good oral bioavailability?",
            "context": "Consider molecular properties and pharmacokinetics"
        },
        {
            "question": "How does adding a fluorine atom typically affect drug metabolism?",
            "context": "Focus on CYP enzyme interactions and metabolic stability"
        }
    ]
    
    dataset = create_chemical_reasoning_dataset(questions)
    
    print("\n💭 Sample Reasoning:")
    df = dataset.to_pandas()
    for _, row in df.iterrows():
        print(f"\nQuestion: {row['question']}")
        print(f"Conclusion: {row['conclusion']}")
        print(f"Key Concepts: {', '.join(row['concepts'])}")


if __name__ == "__main__":
    # Demo with your LogD dataset
    csv_path = "/Users/ldodda/Documents/Codes/GraphGen/test_logd.csv"
    
    print("🚀 TxLlama Drug Discovery Dataset Generation Demo")
    print("=" * 60)
    
    # Run demos
    demo_property_prediction(csv_path)
    print("\n" + "=" * 60)
    demo_molecular_generation()
    print("\n" + "=" * 60)
    demo_chemical_reasoning()
