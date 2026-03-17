import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from collections import Counter
import re
from typing import List, Dict, Tuple
import textwrap

class VADERSentimentAnalyzer:
    """Advanced sentiment analysis using VADER"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.results = []
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Returns:
            Dictionary with neg, neu, pos, and compound scores
        """
        scores = self.analyzer.polarity_scores(text)
        return scores
    
    def classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment based on compound score
        
        Args:
            compound_score: VADER compound score (-1 to 1)
        
        Returns:
            'Positive', 'Negative', or 'Neutral'
        """
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """Analyze multiple texts and return DataFrame"""
        results = []
        
        for i, text in enumerate(texts):
            scores = self.analyze_text(text)
            sentiment = self.classify_sentiment(scores['compound'])
            
            results.append({
                'text': text,
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'positive': scores['pos'],
                'compound': scores['compound'],
                'sentiment': sentiment
            })
        
        self.results = results
        return pd.DataFrame(results)
    
    def plot_sentiment_distribution(self, df: pd.DataFrame = None):
        """Plot sentiment distribution"""
        if df is None:
            df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Sentiment count
        sentiment_counts = df['sentiment'].value_counts()
        colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
        bar_colors = [colors[sent] for sent in sentiment_counts.index]
        
        axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, 
                       color=bar_colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, (idx, val) in enumerate(sentiment_counts.items()):
            axes[0, 0].text(i, val + 0.5, str(val), ha='center', fontweight='bold')
        
        # 2. Compound score distribution
        axes[0, 1].hist(df['compound'], bins=30, color='steelblue', 
                       alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0.05, color='green', linestyle='--', 
                          label='Positive threshold', linewidth=2)
        axes[0, 1].axvline(x=-0.05, color='red', linestyle='--', 
                          label='Negative threshold', linewidth=2)
        axes[0, 1].set_title('Compound Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Compound Score', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Sentiment components boxplot
        sentiment_data = df[['negative', 'neutral', 'positive']].melt(
            var_name='Component', value_name='Score'
        )
        
        box_colors = {'negative': 'red', 'neutral': 'gray', 'positive': 'green'}
        bp = axes[1, 0].boxplot([df['negative'], df['neutral'], df['positive']],
                                labels=['Negative', 'Neutral', 'Positive'],
                                patch_artist=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], ['red', 'gray', 'green']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[1, 0].set_title('Sentiment Component Distributions', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Score', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Pie chart
        axes[1, 1].pie(sentiment_counts.values, labels=sentiment_counts.index,
                      colors=[colors[sent] for sent in sentiment_counts.index],
                      autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        axes[1, 1].set_title('Sentiment Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*70)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nTotal texts analyzed: {len(df)}")
        print(f"\nSentiment breakdown:")
        for sent, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {sent:10s}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\nCompound score statistics:")
        print(f"  Mean:   {df['compound'].mean():7.4f}")
        print(f"  Median: {df['compound'].median():7.4f}")
        print(f"  Std:    {df['compound'].std():7.4f}")
        print(f"  Min:    {df['compound'].min():7.4f}")
        print(f"  Max:    {df['compound'].max():7.4f}")
    
    def plot_sentiment_over_time(self, df: pd.DataFrame = None):
        """Plot sentiment trends over time (assumes chronological order)"""
        if df is None:
            df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # 1. Compound score over time
        axes[0].plot(df.index, df['compound'], marker='o', linestyle='-', 
                    linewidth=2, markersize=6, color='steelblue', alpha=0.7)
        axes[0].axhline(y=0.05, color='green', linestyle='--', 
                       label='Positive threshold', alpha=0.7)
        axes[0].axhline(y=-0.05, color='red', linestyle='--', 
                       label='Negative threshold', alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].set_title('Sentiment Compound Score Over Time', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Text Index', fontsize=12)
        axes[0].set_ylabel('Compound Score', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].fill_between(df.index, df['compound'], 0, 
                            where=(df['compound'] > 0), alpha=0.3, color='green')
        axes[0].fill_between(df.index, df['compound'], 0, 
                            where=(df['compound'] < 0), alpha=0.3, color='red')
        
        # 2. Stacked area chart of components
        axes[1].stackplot(df.index, df['negative'], df['neutral'], df['positive'],
                         labels=['Negative', 'Neutral', 'Positive'],
                         colors=['red', 'gray', 'green'], alpha=0.6)
        axes[1].set_title('Sentiment Components Over Time', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Text Index', fontsize=12)
        axes[1].set_ylabel('Score Proportion', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def get_most_extreme(self, df: pd.DataFrame = None, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get most positive and negative texts"""
        if df is None:
            df = pd.DataFrame(self.results)
        
        most_positive = df.nlargest(n, 'compound')[['text_preview', 'compound', 'sentiment']]
        most_negative = df.nsmallest(n, 'compound')[['text_preview', 'compound', 'sentiment']]
        
        print("\n" + "="*70)
        print(f"TOP {n} MOST POSITIVE TEXTS")
        print("="*70)
        for idx, row in most_positive.iterrows():
            print(f"\nScore: {row['compound']:.4f}")
            print(f"Text: {row['text_preview']}")
        
        print("\n" + "="*70)
        print(f"TOP {n} MOST NEGATIVE TEXTS")
        print("="*70)
        for idx, row in most_negative.iterrows():
            print(f"\nScore: {row['compound']:.4f}")
            print(f"Text: {row['text_preview']}")
        
        return most_positive, most_negative
    
    def analyze_aspect(self, texts: List[str], aspect_keywords: Dict[str, List[str]]) -> pd.DataFrame:
        """Analyze sentiment for specific aspects"""
        results = []
        
        for aspect, keywords in aspect_keywords.items():
            aspect_texts = []
            for text in texts:
                # Check if any keyword is in the text
                if any(keyword.lower() in text.lower() for keyword in keywords):
                    aspect_texts.append(text)
            
            if aspect_texts:
                scores = [self.analyze_text(text)['compound'] for text in aspect_texts]
                avg_score = np.mean(scores)
                sentiment = self.classify_sentiment(avg_score)
                
                results.append({
                    'aspect': aspect,
                    'count': len(aspect_texts),
                    'avg_compound': avg_score,
                    'sentiment': sentiment
                })
        
        aspect_df = pd.DataFrame(results)
        
        # Plot
        if not aspect_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = [('green' if score > 0.05 else 'red' if score < -0.05 else 'gray') 
                     for score in aspect_df['avg_compound']]
            
            bars = ax.barh(aspect_df['aspect'], aspect_df['avg_compound'], 
                          color=colors, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.5)
            ax.axvline(x=-0.05, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Average Compound Score', fontsize=12)
            ax.set_title('Aspect-Based Sentiment Analysis', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add score labels
            for i, (idx, row) in enumerate(aspect_df.iterrows()):
                ax.text(row['avg_compound'], i, 
                       f" {row['avg_compound']:.3f} (n={row['count']})", 
                       va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
        
        return aspect_df
    
    def compare_texts(self, text1: str, text2: str):
        """Compare sentiment of two texts"""
        scores1 = self.analyze_text(text1)
        scores2 = self.analyze_text(text2)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        components = ['negative', 'neutral', 'positive']
        x = np.arange(len(components))
        width = 0.35
        
        # Bar chart comparison
        axes[0].bar(x - width/2, [scores1[c] for c in components], 
                   width, label='Text 1', alpha=0.7, color='steelblue')
        axes[0].bar(x + width/2, [scores2[c] for c in components], 
                   width, label='Text 2', alpha=0.7, color='orange')
        axes[0].set_xlabel('Sentiment Component', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Sentiment Components Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(['Negative', 'Neutral', 'Positive'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Compound score comparison
        compound_scores = [scores1['compound'], scores2['compound']]
        colors = ['green' if s > 0.05 else 'red' if s < -0.05 else 'gray' 
                 for s in compound_scores]
        
        axes[1].bar(['Text 1', 'Text 2'], compound_scores, 
                   color=colors, alpha=0.7, edgecolor='black')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].axhline(y=0.05, color='green', linestyle='--', alpha=0.5)
        axes[1].axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Compound Score', fontsize=12)
        axes[1].set_title('Overall Sentiment Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add score labels
        for i, score in enumerate(compound_scores):
            axes[1].text(i, score, f'{score:.3f}', ha='center', 
                        va='bottom' if score > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*70)
        print("TEXT COMPARISON")
        print("="*70)
        print("\nText 1:")
        print(textwrap.fill(text1, width=70))
        print(f"\nSentiment: {self.classify_sentiment(scores1['compound'])}")
        print(f"Compound: {scores1['compound']:.4f}")
        
        print("\n" + "-"*70)
        print("\nText 2:")
        print(textwrap.fill(text2, width=70))
        print(f"\nSentiment: {self.classify_sentiment(scores2['compound'])}")
        print(f"Compound: {scores2['compound']:.4f}")


def demo_product_reviews():
    """Demo: Analyze product reviews"""
    print("="*70)
    print("VADER SENTIMENT ANALYSIS - PRODUCT REVIEWS")
    print("="*70)
    
    reviews = [
        "This product is absolutely amazing! Best purchase I've ever made!",
        "Terrible quality. Broke after one day. Don't waste your money.",
        "It's okay, nothing special. Works as advertised.",
        "LOVE IT!!! Exceeded all my expectations! Highly recommend!",
        "Worst product ever. Customer service is horrible too.",
        "Pretty good for the price. Some minor issues but overall satisfied.",
        "Not worth it. Too expensive for what you get.",
        "Fantastic! Great quality and fast shipping. 5 stars!",
        "Disappointed. The description was misleading.",
        "Perfect! Exactly what I needed. Will buy again!",
        "Mediocre at best. Expected better quality.",
        "Amazing product! My family loves it too!",
        "Complete garbage. Save your money.",
        "Good value. Does the job well enough.",
        "Exceptional quality and great customer support!"
    ]
    
    analyzer = VADERSentimentAnalyzer()
    df = analyzer.analyze_batch(reviews)
    
    print("\nAnalyzed", len(reviews), "product reviews")
    print("\nFirst few results:")
    print(df[['text_preview', 'compound', 'sentiment']].head(10).to_string())
    
    # Visualizations
    analyzer.plot_sentiment_distribution(df)
    analyzer.plot_sentiment_over_time(df)
    analyzer.get_most_extreme(df, n=3)


def demo_social_media():
    """Demo: Analyze social media posts"""
    print("\n" + "="*70)
    print("VADER SENTIMENT ANALYSIS - SOCIAL MEDIA POSTS")
    print("="*70)
    
    posts = [
        "Just got promoted at work! 🎉 So excited for this new opportunity!",
        "Ugh, stuck in traffic again. This city's traffic is the WORST 😤",
        "Beautiful sunset today 🌅 #blessed #grateful",
        "Can't believe they cancelled my favorite show 😢😭",
        "Had an amazing dinner with friends tonight! ❤️",
        "This weather is terrible. Rain rain rain 🌧️",
        "Feeling blessed and grateful for all the love and support ❤️🙏",
        "So frustrated with this slow internet!!! 😡",
        "Best vacation ever! Can't wait to come back! 🏖️",
        "Disappointed with the new update. Lots of bugs 🐛",
    ]
    
    analyzer = VADERSentimentAnalyzer()
    df = analyzer.analyze_batch(posts)
    
    print("\nAnalyzed", len(posts), "social media posts")
    
    analyzer.plot_sentiment_distribution(df)
    analyzer.get_most_extreme(df, n=3)


def demo_aspect_analysis():
    """Demo: Aspect-based sentiment analysis for hotel reviews"""
    print("\n" + "="*70)
    print("ASPECT-BASED SENTIMENT ANALYSIS - HOTEL REVIEWS")
    print("="*70)
    
    reviews = [
        "The room was spacious and clean. Staff was very friendly and helpful.",
        "Terrible location, far from everything. But the food was delicious.",
        "Amazing service! The staff went above and beyond. Room was a bit small though.",
        "Beautiful location with great views. Breakfast was disappointing.",
        "Room was dirty and outdated. Staff was rude. Would not recommend.",
        "Excellent location, close to all attractions. Staff was okay, nothing special.",
        "The food at the restaurant was outstanding! Best hotel breakfast ever!",
        "Staff was unprofessional. Room was nice but overpriced for the location.",
        "Perfect location and amazing service. Room could use some updates.",
        "Clean room, comfortable bed. Location is convenient. Staff needs improvement."
    ]
    
    aspects = {
        'Room': ['room', 'bed', 'clean', 'spacious', 'dirty', 'comfortable'],
        'Location': ['location', 'close', 'far', 'convenient', 'views'],
        'Staff': ['staff', 'service', 'friendly', 'helpful', 'rude', 'professional'],
        'Food': ['food', 'breakfast', 'restaurant', 'delicious', 'disappointing']
    }
    
    analyzer = VADERSentimentAnalyzer()
    aspect_df = analyzer.analyze_aspect(reviews, aspects)
    
    print("\nAspect-based sentiment results:")
    print(aspect_df.to_string(index=False))


def demo_text_comparison():
    """Demo: Compare two texts"""
    print("\n" + "="*70)
    print("TEXT COMPARISON DEMO")
    print("="*70)
    
    text1 = "I absolutely love this new phone! The camera is incredible, battery life is amazing, and the design is beautiful. Best phone I've ever owned!"
    
    text2 = "This phone is terrible. The battery dies quickly, camera quality is poor, and it's way too expensive. Very disappointed with my purchase."
    
    analyzer = VADERSentimentAnalyzer()
    analyzer.compare_texts(text1, text2)


if __name__ == "__main__":
    # Run demos
    demo_product_reviews()
    demo_social_media()
    demo_aspect_analysis()
    demo_text_comparison()