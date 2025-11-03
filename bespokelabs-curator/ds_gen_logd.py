# %%
import json
import pandas as pd
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datasets import Dataset
from bespokelabs import curator
import argparse
from pathlib import Path
from typing import List, Dict
import litellm 
litellm.drop_params = True

import os
os.environ["AWS_PROFILE"] = "genai"
os.environ["CURATOR_VIEWER"] = "1"
os.environ["BESPOKE_API_KEY"] = "bespoke-9ff97e97445794907bec5ef8d57fd925da61e8ba7586c17ec2dca4bdffb87b9b"

#model_name = "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
model_name = "bedrock/us.amazon.nova-micro-v1:0"

# %%
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


# %%
# ============================================================================
# Core Dataset Generators
# ============================================================================
class PropertyPredictor(curator.LLM):
    """Generates property prediction tasks with reasoning."""
    
    response_format = MolecularProperty
    
    def prompt(self, input: Dict) -> str:
        smiles = input['canonical_smiles']
        property_type = input.get('standard_type', 'LogD')

        return f"""You are an expert medicinal chemistry research assistant. You have knowledge in areas including:

- Drug design and development - structure-activity relationships (SAR), lead optimization, pharmacophore modeling
- Synthetic chemistry - reaction mechanisms, synthetic routes, protecting group strategies
- Pharmacology - drug-target interactions, pharmacokinetics (ADME), pharmacodynamics
- Computational chemistry - molecular modeling, QSAR and ADME predictions
- Analytical techniques - NMR, MS, HPLC and other characterization methods
- Medicinal chemistry strategies - bioisosteres, prodrugs, fragment-based drug design

Analyze the following molecule and predict its {property_type}.

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
        }

# %%
df = pd.read_csv('/Users/ldodda/Documents/Codes/GraphGen/test_logd.csv')

# %% [markdown]
# ## Sampling from all bins of LogD Data to make sure I am getting a diverse set

# %%
# Create bins for logd values covering the full range -8 to +8
import numpy as np

# Create 16 bins from -8 to +8
bins = np.linspace(-8, 8, 17)  # 17 edges create 16 bins
df['logd_bin'] = pd.cut(df['logd'], bins=bins, include_lowest=True)

# Sample from each bin (e.g., 20 samples per bin)
samples_per_bin = 100
sampled_df = df.groupby('logd_bin', observed=False, group_keys=False).apply(
    lambda x: x.sample(min(len(x), samples_per_bin), random_state=42)
)

print(f"Original dataset size: {len(df)}")
print(f"Sampled dataset size: {len(sampled_df)}")
print("\nSamples per bin:")
print(sampled_df['logd_bin'].value_counts().sort_index())
print(f"\nBin width: {(8 - (-8))/16} = 1.0 LogD unit per bin")

# %%
sampled_df.drop(columns=['logd_bin'], inplace=True)

# %%
sampled_df

# %%
sampled_df.columns = ['canonical_smiles', 'standard_value']
sampled_df['standard_type'] = 'LogD'

# %%
mol_data_dict = sampled_df.to_dict(orient='records')

# %%
logd_dataset = Dataset.from_list(mol_data_dict)

# %%
logd_dataset

# %%
predictor = PropertyPredictor(
        model_name=model_name,
        backend="litellm"
)

# %%
print("Generating predictions...")
results = predictor(logd_dataset)


# %%
print("\n📊 Results:")
logD_dataset_df = results.dataset.to_pandas()

# %%
fname_suffix = f'logD_predictions_w_{model_name.split("/")[1]}'

# %%
fname_suffix

# %%
logD_dataset_df.to_csv(fname_suffix +'.csv')
logD_dataset_df

# %%
results.dataset

# %%
results.dataset.to_list()

# %%
output_file = fname_suffix + ".jsonl"
with open(output_file, 'w') as f:
    for item in results.dataset.to_list():
        f.write(json.dumps(item) + '\n')

# %%
"""
Data Preparation Script for Llama 3.2 3B Fine-tuning
Converts bespokelabs-curator dataset to instruction-following format
"""

system_prompt = """You are an expert medicinal chemistry research assistant. You have knowledge in areas including:

- Drug design and development - structure-activity relationships (SAR), lead optimization, pharmacophore modeling
- Synthetic chemistry - reaction mechanisms, synthetic routes, protecting group strategies
- Pharmacology - drug-target interactions, pharmacokinetics (ADME), pharmacodynamics
- Computational chemistry - molecular modeling, QSAR and predicting ADME properties
- Analytical techniques - NMR, MS, HPLC and other characterization methods
- Medicinal chemistry strategies - bioisosteres, prodrugs, fragment-based drug design

You analyze molecular structures using SMILES notation and provide accurate predictions for LogD values (distribution coefficient at pH 7.4). 
Your predictions are based on careful analysis of structural features, lipophilic and hydrophilic contributions, and established cheminformatics principles.
"""
class DatasetConverter:
    """Convert molecular property dataset to instruction-following format."""
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
    
    def convert_to_instruction_format(self) -> None:
        """Convert dataset to Llama 3.2 instruction format."""
        
        print(f"Reading dataset from: {self.input_path}")
        
        with open(self.input_path, 'r') as f_in:
            with open(self.output_path, 'w') as f_out:
                example_count = 0
                
                for line_num, line in enumerate(f_in, 1):
                    try:
                        example = json.loads(line)
                        converted = self._create_instruction_example(example)
                        f_out.write(json.dumps(converted) + '\n')
                        example_count += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    except KeyError as e:
                        print(f"Warning: Missing field on line {line_num}: {e}")
        
        print(f"✓ Converted {example_count} examples")
        print(f"✓ Output saved to: {self.output_path}")
    
    @staticmethod
    def _create_instruction_example(entry: Dict) -> Dict:
        """
        Convert single entry to instruction format.
        
        Args:
            entry: Original entry with SMILES, values, and reasoning
            
        Returns:
            Dictionary with 'text' field containing formatted conversation
        """
        
        smiles = entry['smiles']
        actual_value = entry['actual_value']
        reasoning = entry['reasoning']
        key_features = ', '.join(entry['key_features'])
        
        # Create the full conversation in Llama 3.2 format
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Analyze the following molecule and predict its LogD value.

SMILES: {smiles}

Key Structural Features: {key_features}

Chemical Analysis and Reasoning:
{reasoning}

Based on this analysis, what is the predicted LogD value for this molecule?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Based on the detailed structural analysis, considering the balance of lipophilic and hydrophilic features:

The predicted LogD value is: **{actual_value}**

This prediction reflects:
- The lipophilic contributions from the {key_features.split(',')[0] if ',' in key_features else 'aromatic and aliphatic components'}
- The hydrophilic contributions from polar functional groups
<|eot_id|>"""
        
        return {"text": text}
    
    def validate_output(self) -> bool:
        """Validate the converted dataset."""
        
        print("\nValidating dataset...")
        
        try:
            examples = []
            with open(self.output_path, 'r') as f:
                for i, line in enumerate(f):
                    ex = json.loads(line)
                    if 'text' not in ex:
                        raise ValueError(f"Line {i}: missing 'text' field")
                    examples.append(ex)
            
            print(f"✓ Valid JSONL format")
            print(f"✓ Number of examples: {len(examples)}")
            
            # Show statistics
            text_lengths = [len(ex['text']) for ex in examples]
            print(f"✓ Average text length: {sum(text_lengths) / len(text_lengths):.0f} chars")
            print(f"✓ Min length: {min(text_lengths)} chars")
            print(f"✓ Max length: {max(text_lengths)} chars")
            
            # Show sample
            print(f"\n--- Sample Example (first 500 chars) ---")
            print(examples[0]['text'][:500] + "...\n")
            
            return True
            
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            return False

# %%
converter = DatasetConverter(f'./{fname_suffix}.jsonl', f'./{fname_suffix}_formatted.jsonl')
converter.convert_to_instruction_format()

# %%
from datasets import Dataset


def load_jsonl_data(file_path):
    """Load JSONL dataset."""
    texts = []
    with open(file_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            texts.append(example['text'])
    return texts

DATASET_PATH = './logd_formatted.jsonl'
print(f"Loading dataset from {DATASET_PATH}...")
texts = load_jsonl_data(DATASET_PATH)
print(f"Loaded {len(texts)} examples")

# Create HF dataset
dataset = Dataset.from_dict({
    'text': texts
})

# %%
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# %%
dataset

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

# Calculate regression metrics
actual_values = logD_dataset_df['actual_value']
predicted_values = logD_dataset_df['predicted_value']

# Regression metrics
mse = mean_squared_error(actual_values, predicted_values)
rmse = mse ** 0.5
mae = mean_absolute_error(actual_values, predicted_values)
r2 = r2_score(actual_values, predicted_values)

# Spearman correlation
spearman_corr, spearman_p = spearmanr(actual_values, predicted_values)

# Pearson correlation (for comparison)
pearson_corr = actual_values.corr(predicted_values)

print("📊 LogD Prediction Performance Metrics:")
print("=" * 50)
print(f"Mean Squared Error (MSE):     {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE):    {mae:.4f}")
print(f"R² Score:                     {r2:.4f}")
print(f"Pearson Correlation:          {pearson_corr:.4f}")
print(f"Spearman Correlation:         {spearman_corr:.4f}")
print(f"Spearman p-value:             {spearman_p:.2e}")

# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(actual_values, predicted_values, alpha=0.6, s=50)
plt.plot([actual_values.min(), actual_values.max()], 
         [actual_values.min(), actual_values.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual LogD Values')
plt.ylabel('Predicted LogD Values')
plt.title(f'LogD Prediction Performance\nSpearman ρ = {spearman_corr:.3f}, R² = {r2:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{fname_suffix}_regression_performance.png', dpi=300)
plt.show()

# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Create LogD zones classification
def classify_logd(value):
    if value < -2:
        return 'Low (<-2)'
    elif value <= 4:
        return 'Medium (-2 to 4)'
    else:
        return 'High (>4)'

# Apply classification to actual and predicted values
logD_dataset_df['actual_zone'] = actual_values.apply(classify_logd)
logD_dataset_df['predicted_zone'] = predicted_values.apply(classify_logd)

# Generate classification statistics

# Calculate classification metrics
accuracy = accuracy_score(logD_dataset_df['actual_zone'], logD_dataset_df['predicted_zone'])

print("📊 LogD Zone Classification Performance:")
print("=" * 60)
print(f"Overall Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(logD_dataset_df['actual_zone'], logD_dataset_df['predicted_zone']))

# Create confusion matrix
cm = confusion_matrix(logD_dataset_df['actual_zone'], logD_dataset_df['predicted_zone'], 
                     labels=['Low (<-2)', 'Medium (-2 to 4)', 'High (>4)'])

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low (<-2)', 'Medium (-2 to 4)', 'High (>4)'],
            yticklabels=['Low (<-2)', 'Medium (-2 to 4)', 'High (>4)'])
plt.title('Confusion Matrix - LogD Zone Classification')
plt.xlabel('Predicted Zone')
plt.ylabel('Actual Zone')
plt.savefig(f'{fname_suffix}_confusion_matrix.png', dpi=300)
plt.show()

# Show distribution of zones
print("\n📈 Zone Distribution:")
print("Actual zones:")
print(logD_dataset_df['actual_zone'].value_counts().sort_index())
print("\nPredicted zones:")
print(logD_dataset_df['predicted_zone'].value_counts().sort_index())

# %%
#dataset.push_to_hub("lsdodda/LogD-Predictor")

# %%



