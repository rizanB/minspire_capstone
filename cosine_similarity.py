import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_and_prepare_data(csv_path='trekverse_cleaned.csv'):
    """
    Loads the cleaned dataset and computes the cosine similarity matrix.
    Returns the dataframe and the similarity matrix.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_path}'. Make sure you've run the preprocessing notebook.")
        return None, None
        
    # The 5 Trek DNA scores used for calculating similarity
    feature_cols = ['Adventure_Score', 'Comfort_DNA', 'Remoteness_Score', 'Effort_Score', 'Scenic_Score']
    
    # Extract feature matrix
    rec_features = df[feature_cols].values
    
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(rec_features)
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=df['Trek'], 
        columns=df['Trek']
    )
    
    return df, similarity_df

def recommend_treks(trek_name, df, similarity_df, top_n=5):
    """
    Given a trek name, returns the top-N most similar treks based on Trek DNA.
    """
    if trek_name not in similarity_df.index:
        print(f"Trek '{trek_name}' not found in the dataset.")
        return None
        
    # Get similarity scores for the given trek, exclude the trek itself
    similarities = similarity_df[trek_name].drop(trek_name).sort_values(ascending=False)
    top_similar = similarities.head(top_n)
    
    print(f"\\n🏔️ Treks similar to '{trek_name}':")
    print("=" * 70)
    
    source = df[df['Trek'] == trek_name].iloc[0]
    feature_cols = ['Adventure_Score', 'Comfort_DNA', 'Remoteness_Score', 'Effort_Score', 'Scenic_Score']
    labels = ['Adventure', 'Comfort', 'Remoteness', 'Effort', 'Scenic']
    
    results = []
    for rank, (rec_trek, score) in enumerate(top_similar.items(), 1):
        rec_row = df[df['Trek'] == rec_trek].iloc[0]
        
        # Generate explanation by finding features with similar values
        shared_traits = []
        for feat, label in zip(feature_cols, labels):
            if abs(source[feat] - rec_row[feat]) < 0.2:
                shared_traits.append(label)
                
        explanation = f"Similar in: {', '.join(shared_traits[:3])}" if shared_traits else "Overall profile match"
        
        persona = rec_row.get('Persona', 'N/A')
        cost = rec_row['Cost_USD']
        duration = rec_row['Duration_Days']
        
        print(f"  {rank}. {rec_trek}")
        print(f"     Similarity: {score:.3f} | Cost: ${cost:.0f} | {duration:.0f} days | {persona}")
        print(f"     {explanation}")
        
        results.append({
            'Trek': rec_trek, 
            'Similarity': score, 
            'Persona': persona,
            'Cost_USD': cost,
            'Duration_Days': duration
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Initializing Cosine Similarity Recommendation Engine...")
    df, similarity_df = load_and_prepare_data('trekverse_cleaned.csv')
    
    if df is not None and similarity_df is not None:
        # Provide a quick demo
        demo_trek = 'Everest Base Camp Trek'
        if demo_trek in df['Trek'].values:
            recommend_treks(demo_trek, df, similarity_df, top_n=5)
        else:
            print("Demo trek not found. Engine is ready for use with other treks.")
