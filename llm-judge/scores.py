import pandas as pd

df = pd.read_csv("/home/liam/Documents/fellow-result/judge_results_snapshot.csv")
df_clean = df[df['error'].isna() | (df['error'] == '')].copy()

score_cols = ['faithfulness', 'context_leakage', 'citation_accuracy', 'answer_relevance', 'repetition']
df_clean[score_cols] = df_clean[score_cols].apply(pd.to_numeric, errors='coerce')

df_clean['faithfulness_n']      = df_clean['faithfulness'] / 3
df_clean['context_leakage_n']   = df_clean['context_leakage'] / 2
df_clean['citation_accuracy_n'] = df_clean['citation_accuracy'] / 3
df_clean['answer_relevance_n']  = df_clean['answer_relevance'] / 3
df_clean['repetition_n']        = df_clean['repetition'] / 2

df_clean['composite_equal'] = (
    df_clean['faithfulness_n'] +
    df_clean['context_leakage_n'] +
    df_clean['citation_accuracy_n'] +
    df_clean['answer_relevance_n'] +
    df_clean['repetition_n']
) / 5

df_clean['composite_weighted'] = (
    df_clean['faithfulness_n']      * 0.30 +
    df_clean['citation_accuracy_n'] * 0.30 +
    df_clean['answer_relevance_n']  * 0.20 +
    df_clean['context_leakage_n']   * 0.10 +
    df_clean['repetition_n']        * 0.10
)

def grade(score):
    if score >= 0.85: return 'A'
    if score >= 0.70: return 'B'
    if score >= 0.55: return 'C'
    if score >= 0.40: return 'D'
    return 'F'

grouped = df_clean.groupby(['model', 'precision'])[score_cols + ['composite_equal', 'composite_weighted']].mean().round(3)
grouped['grade_equal']    = grouped['composite_equal'].apply(grade)
grouped['grade_weighted'] = grouped['composite_weighted'].apply(grade)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(grouped)
