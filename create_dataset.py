"""
This script creates a small sample dataset for testing the RAG-based hate speech classification app.
Run this script to generate a sample_hate_speech_dataset.csv file.
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create sample hate speech examples
hate_speech_examples = [
    "I hate all people from that country. They should all be deported.",
    "Those people are disgusting and don't deserve to live here.",
    "This ethnic group is ruining our country with their customs.",
    "Everyone from that religion is a terrorist and should be banned.",
    "Women don't deserve equal pay because they're naturally inferior.",
    "That racial group is inherently stupid and lazy.",
    "All immigrants are criminals who steal our jobs.",
    "People of that sexual orientation are all perverts.",
    "People with that disability are a burden on society.",
    "Old people should just die already instead of wasting resources.",
    "That nationality is full of thieves and liars.",
    "Those people are animals and should be treated as such.",
    "We should eliminate that group from our society completely.",
    "They're genetically inferior to normal people.",
    "That racial group is naturally violent and dangerous.",
    "Those religious people are brainwashed idiots who spread hate.",
    "People with mental illness should be locked away from normal society.",
    "Poor people are just lazy and deserve their situation.",
    "Those immigrants bring disease and crime to our country.",
    "That group should have their rights taken away.",
]

# Create sample non-hate speech examples
non_hate_speech_examples = [
    "I love sunny days at the beach.",
    "The new restaurant downtown has amazing food.",
    "My dog is the most loyal companion I could ask for.",
    "I'm looking forward to the concert next weekend.",
    "This book taught me a lot about world history.",
    "The movie was entertaining but a bit too long.",
    "I respect people's right to practice different religions.",
    "It's important to listen to diverse perspectives.",
    "The scientific research on this topic is fascinating.",
    "We should work together to solve community problems.",
    "Different cultures have valuable traditions to share.",
    "Equal opportunities benefit everyone in society.",
    "I disagree with that political view, but respect your right to hold it.",
    "Learning about other cultures broadens my understanding of the world.",
    "Everyone deserves to be treated with dignity and respect.",
    "The documentary presented multiple viewpoints on the issue.",
    "I appreciate how diverse our community has become.",
    "People from all backgrounds contribute valuable skills.",
    "It's interesting to learn about different religious practices.",
    "We should judge people by their character, not their appearance.",
]

# Create some more ambiguous examples
ambiguous_examples = [
    "I don't like the way that group behaves sometimes.",
    "There are cultural differences that can be challenging.",
    "That political party's policies concern me.",
    "Some traditions from that culture seem strange to me.",
    "We need stronger immigration policies.",
    "Crime rates in that neighborhood are concerning.",
    "There are biological differences between groups that affect performance.",
    "Their religious practices seem very unusual.",
    "That country has some serious problems with its government.",
    "I prefer to live in communities with people similar to me.",
]

# Create DataFrame
hate_speech_df = pd.DataFrame({
    "prompt": hate_speech_examples,
    "label": ["hate_speech"] * len(hate_speech_examples)
})

non_hate_speech_df = pd.DataFrame({
    "prompt": non_hate_speech_examples,
    "label": ["not_hate_speech"] * len(non_hate_speech_examples)
})

# Label half of ambiguous examples as hate speech and half as not hate speech
np.random.shuffle(ambiguous_examples)
mid_point = len(ambiguous_examples) // 2
ambiguous_hate = pd.DataFrame({
    "prompt": ambiguous_examples[:mid_point],
    "label": ["hate_speech"] * mid_point
})
ambiguous_non_hate = pd.DataFrame({
    "prompt": ambiguous_examples[mid_point:],
    "label": ["not_hate_speech"] * (len(ambiguous_examples) - mid_point)
})

# Combine all examples
combined_df = pd.concat([
    hate_speech_df, 
    non_hate_speech_df,
    ambiguous_hate,
    ambiguous_non_hate
])

# Shuffle the dataset
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Save to CSV
combined_df.to_csv("sample_hate_speech_dataset.csv", index=False)

print(f"Created sample dataset with {len(combined_df)} examples")
print(f"- Hate speech examples: {len(hate_speech_df) + len(ambiguous_hate)}")
print(f"- Non-hate speech examples: {len(non_hate_speech_df) + len(ambiguous_non_hate)}")
print("Saved to sample_hate_speech_dataset.csv")
