import pandas as pd
import numpy as np
import random
import csv
from datetime import datetime
import argparse

def generate_hate_speech_dataset(num_samples=100, output_file='hate_speech_dataset.csv'):
    """
    Generate a dataset of labeled examples for hate speech classification.
    
    Parameters:
    num_samples (int): Number of samples to generate
    output_file (str): Path to save the CSV file
    
    Returns:
    pandas.DataFrame: The generated dataset
    """
    
    # Templates for hate speech 
    hate_speech_templates = [
        "I hate all {group}. They should all be {action}.",
        "All {group} are {negative_trait} and don't deserve to {right}.",
        "{group} are ruining our {place} with their {trait}.",
        "We should {violent_action} every {group} we see.",
        "The world would be better if all {group} would just {disappear}.",
        "{group} are naturally {negative_trait} and {negative_trait2}.",
        "No {group} deserves to {right}, they're all {negative_trait}.",
        "I can't stand {group}, they make me want to {violent_reaction}.",
        "Why do we allow {group} to even {basic_action}? They're {negative_trait}.",
        "{group} shouldn't be allowed to {right} because they're {negative_trait}."
    ]
    
    # Templates for non-hate speech
    non_hate_speech_templates = [
        "I love {positive_thing}. It makes me feel {positive_emotion}.",
        "The new {place} downtown has amazing {positive_thing}.",
        "My {pet} is the most {positive_trait} companion I could ask for.",
        "I'm looking forward to {positive_event} next weekend.",
        "This {media} taught me a lot about {topic}.",
        "I respect people's right to {neutral_action}, even if I disagree.",
        "It's important to listen to diverse perspectives on {topic}.",
        "The research on {topic} is fascinating and worth exploring.",
        "We should work together to solve {problem} in our community.",
        "Different cultures have valuable {positive_thing} to share.",
        "Everyone deserves to be treated with {positive_trait} and respect.",
        "I prefer to focus on solutions rather than blame when discussing {topic}.",
        "Learning about other perspectives broadens my understanding of {topic}.",
        "The documentary presented multiple viewpoints on {topic}.",
        "I appreciate how diverse our community has become."
    ]
    
    # Fill-in data
    groups = [
        "immigrants", "refugees", "people from that country", "foreigners", 
        "members of that religion", "people of that race", "minorities",
        "people with that disability", "LGBTQ+ people", "liberals", "conservatives"
    ]
    
    negative_traits = [
        "disgusting", "lazy", "stupid", "dangerous", "violent", "dishonest", 
        "criminal", "worthless", "inferior", "primitive", "corrupt", "evil",
        "greedy", "manipulative", "untrustworthy", "selfish", "invasive"
    ]
    
    negative_traits2 = [
        "parasitic", "destructive", "harmful", "conniving", "deceitful",
        "immoral", "backward", "savage", "hateful", "vile", "repulsive"
    ]
    
    rights = [
        "live here", "have equal rights", "vote", "have jobs", "receive benefits", 
        "speak freely", "practice their customs", "be treated equally", 
        "have healthcare", "get an education", "feel safe", "own property"
    ]
    
    actions = [
        "deported", "banned", "excluded", "removed", "restricted", 
        "monitored", "controlled", "registered", "isolated", "segregated"
    ]
    
    places = [
        "country", "neighborhood", "community", "city", "schools", 
        "workplace", "society", "nation", "public spaces", "culture"
    ]
    
    traits = [
        "customs", "traditions", "practices", "beliefs", "behaviors", 
        "values", "attitudes", "habits", "languages", "religions"
    ]
    
    violent_actions = [
        "eliminate", "get rid of", "purge", "cleanse society of",
        "remove", "drive out", "push out", "banish", "expel"
    ]
    
    disappear = [
        "disappear", "leave", "die off", "go back where they came from",
        "get out", "be gone", "vanish", "stop existing", "be eliminated"
    ]
    
    violent_reactions = [
        "scream", "vomit", "punch something", "get violent", 
        "lose my temper", "fight back", "take action", "retaliate"
    ]
    
    basic_actions = [
        "exist", "walk around freely", "have rights", "have opinions", 
        "be in public", "speak their language", "practice their customs"
    ]
    
    # Non-hate speech fillers
    positive_things = [
        "music", "food", "art", "culture", "books", "movies", "stories",
        "ideas", "innovations", "traditions", "festivals", "cuisines"
    ]
    
    positive_emotions = [
        "happy", "joyful", "inspired", "peaceful", "content", "excited",
        "grateful", "optimistic", "hopeful", "relaxed", "energized"
    ]
    
    places_positive = [
        "restaurant", "park", "museum", "café", "theater", "library",
        "garden", "market", "community center", "art gallery", "concert hall"
    ]
    
    pets = [
        "dog", "cat", "bird", "fish", "rabbit", "hamster",
        "guinea pig", "turtle", "lizard", "pet", "companion"
    ]
    
    positive_traits = [
        "loyal", "kind", "loving", "creative", "intelligent", "thoughtful",
        "honest", "helpful", "patient", "diligent", "compassionate", "respectful"
    ]
    
    positive_events = [
        "concert", "festival", "party", "gathering", "celebration",
        "workshop", "trip", "vacation", "meeting", "reunion", "exhibition"
    ]
    
    media = [
        "book", "documentary", "podcast", "article", "course", 
        "lecture", "seminar", "video", "program", "blog post"
    ]
    
    topics = [
        "history", "science", "art", "technology", "philosophy",
        "psychology", "economics", "environment", "education", "health",
        "politics", "sociology", "anthropology", "astronomy", "literature"
    ]
    
    neutral_actions = [
        "practice their religion", "express their views", "make their choices",
        "live according to their values", "pursue their interests", 
        "celebrate their traditions", "raise their children", "vote their conscience"
    ]
    
    problems = [
        "poverty", "climate change", "inequality", "pollution", "injustice",
        "homelessness", "hunger", "discrimination", "violence", "healthcare access",
        "education gaps", "unemployment", "housing shortages", "infrastructure issues"
    ]
    
    # Generate data
    data = []
    
    # Ensure roughly balanced classes (slight bias towards non-hate speech, which is more common)
    hate_speech_count = int(num_samples * 0.4)  # 40% hate speech
    non_hate_speech_count = num_samples - hate_speech_count  # 60% non-hate speech
    
    # Generate hate speech examples
    for _ in range(hate_speech_count):
        template = random.choice(hate_speech_templates)
        
        # Fill in the template with random choices from the lists
        text = template.format(
            group=random.choice(groups),
            negative_trait=random.choice(negative_traits),
            negative_trait2=random.choice(negative_traits2),
            right=random.choice(rights),
            action=random.choice(actions),
            place=random.choice(places),
            trait=random.choice(traits),
            violent_action=random.choice(violent_actions),
            disappear=random.choice(disappear),
            violent_reaction=random.choice(violent_reactions),
            basic_action=random.choice(basic_actions)
        )
        
        data.append({"prompt": text, "label": "hate_speech"})
    
    # Generate non-hate speech examples
    for _ in range(non_hate_speech_count):
        template = random.choice(non_hate_speech_templates)
        
        # Fill in the template with random choices from the lists
        text = template.format(
            positive_thing=random.choice(positive_things),
            positive_emotion=random.choice(positive_emotions),
            place=random.choice(places_positive),
            pet=random.choice(pets),
            positive_trait=random.choice(positive_traits),
            positive_event=random.choice(positive_events),
            media=random.choice(media),
            topic=random.choice(topics),
            neutral_action=random.choice(neutral_actions),
            problem=random.choice(problems)
        )
        
        data.append({"prompt": text, "label": "not_hate_speech"})
    
    # Shuffle the data
    random.shuffle(data)
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"Generated {num_samples} samples and saved to {output_file}")
    print(f"Hate speech examples: {hate_speech_count}")
    print(f"Non-hate speech examples: {non_hate_speech_count}")
    
    return df

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate a hate speech dataset for classification')
    parser.add_argument('--samples', type=int, default=200, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    
    args = parser.parse_args()
    
    # Generate a timestamp for the filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hate_speech_dataset_{timestamp}.csv"
    else:
        filename = args.output
    
    # Generate the dataset
    df = generate_hate_speech_dataset(num_samples=args.samples, output_file=filename)
    
    # Display the first few examples
    print("\nExample data:")
    print(df.head())