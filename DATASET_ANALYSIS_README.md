# Dataset Analysis Notebook

## Overview

The `dataset_analysis.ipynb` notebook provides comprehensive statistical analysis of your EHR dataset across three levels:

1. **Demographics Analysis**: Patient characteristics (age, gender, ethnicity, case/control status)
2. **Token Trajectory Analysis**: Original token sequences, temporal patterns, and token type distributions
3. **LLM Tokenization Analysis**: How the Qwen3-8B tokenizer processes natural language text

## Features

### 1. Demographics Analysis
- Case vs Control distribution by split
- Age distribution (overall, by split, by case/control)
- Gender distribution across splits
- Ethnicity distribution and diversity
- Cross-tabulations and statistical comparisons

### 2. Token Trajectory Analysis
- Sequence length distributions
- Token type breakdown:
  - Medical codes (`MEDICAL//`)
  - Lab measurements (`LAB//`)
  - Measurements (`MEASUREMENT//`)
  - Time intervals (`<time_interval_*>`)
  - Demographics, lifestyle, special tokens
  - Numeric values
- Temporal analysis:
  - Total timeline duration per patient
  - Time between events (delta times)
  - Temporal patterns by split and case/control

### 3. LLM Tokenization Analysis
- LLM token count distribution
- Text length analysis (characters)
- Compression/expansion ratios (EHR tokens → LLM tokens)
- Correlation between EHR and LLM token counts
- Characters per token metrics

### 4. Comparative Analysis
- Comprehensive split comparison tables
- Statistical tests for split similarity:
  - Chi-square test for case/control proportions
  - Kruskal-Wallis tests for age and token distributions
- Bias verification across splits

## Configuration

The notebook uses paths from `src/configs/llm_pretrain.yaml`:

- **Data Directory**: `/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi/`
- **Model**: `unsloth/Qwen3-8B-Base-unsloth-bnb-4bit`
- **Splits**: train, tuning, held_out

## Usage

### Running the Notebook

1. Ensure you have access to the data server with the paths specified in the config
2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn torch transformers tqdm scipy
   ```
3. Open the notebook:
   ```bash
   jupyter notebook dataset_analysis.ipynb
   ```
4. Run all cells sequentially

### Expected Runtime

- **Setup & Data Loading**: 5-15 minutes (depending on data size)
- **Demographics Analysis**: 2-5 minutes
- **Token Trajectory Analysis**: 5-10 minutes
- **LLM Tokenization**: 10-30 minutes (sampling applied to train split by default)
- **Comparative Analysis**: < 1 minute

**Total**: ~30-60 minutes for complete analysis

### Sampling Configuration

For faster iteration, the LLM tokenization section samples 1000 patients from the train split by default. To analyze all patients:

```python
# In cell 33, change:
sample_size = 1000 if split == 'train' else None
# To:
sample_size = None
```

## Output

The notebook generates:

1. **Visualizations**:
   - Histograms for age, token counts, durations
   - Bar charts for categorical variables
   - Box plots for split comparisons
   - Scatter plots for correlations
   - Pie charts for token type distributions

2. **Summary Tables**:
   - Demographics summary by split
   - Token trajectory metrics
   - LLM tokenization statistics
   - Comprehensive split comparison

3. **Statistical Tests**:
   - Chi-square tests
   - Kruskal-Wallis tests
   - Correlation coefficients

## Key Insights Provided

- Total patient counts and case/control ratios
- Age and demographic distributions
- Median token counts (EHR and LLM)
- Average token types per patient
- Temporal patterns (timeline duration, time between events)
- LLM tokenization expansion ratios
- Split balance verification

## Customization

### Adding New Analyses

To add custom analyses:

1. Extract data from `demographics_combined` or `trajectory_combined` DataFrames
2. Create visualizations using matplotlib/seaborn
3. Add summary statistics using pandas groupby operations

### Modifying Visualizations

All plots use consistent styling:
- Figure sizes: (12, 6) or (16, 12) for multi-panel
- Colors: Seaborn "husl" palette
- Fonts: 10pt default, 14pt for titles

Modify `plt.rcParams` in cell 2 to change global settings.

## Dependencies

- **Core**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **ML**: torch, transformers
- **Statistics**: scipy
- **Progress**: tqdm
- **Project**: src.data.token_translator, src.data.unified_dataset_v2

## Notes

- The notebook is read-only for the dataset - no modifications are made
- All file paths are absolute to work on the remote server
- Tokenizer loading may require HuggingFace authentication
- Warnings about missing tokenizer are expected if model access is not configured
- Statistical tests help verify that splits are unbiased

## Troubleshooting

### Tokenizer Not Loading
- Ensure you have HuggingFace token configured
- Check model access permissions for `unsloth/Qwen3-8B-Base-unsloth-bnb-4bit`
- The analysis will continue without LLM tokenization if tokenizer fails

### Memory Issues
- Reduce sample sizes in LLM tokenization
- Process splits separately instead of combining
- Use a machine with more RAM

### Path Errors
- Verify data paths exist on your server
- Update paths in cell 3 if your data is in a different location
- Check that all CSV lookup files are in `src/resources/`

## Citation

If you use this analysis in your research, please cite your project appropriately.

## Contact

For questions or issues, please contact the project maintainers.

