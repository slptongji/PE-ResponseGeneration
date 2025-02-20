from openai import OpenAI

import os
import openai
from tqdm import tqdm



completion = client.chat.completions.create(
    model="your_model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": """
            ### User’s persona categories:
1. Personality,
2. Culture,
3. Hobby.
User’s big five personality: (0.113, 0.857, 0.057, 0.86, 0.711)
The Big Five personality vector describes the degree of each trait, with a value between 0 and 1: Openness, Conscientiousness, Extraversion, Agreeableness and Neuroticism.
Openness: Closer to 1 means the user is more imaginative and spontaneous; closer to 0 means the user prefers routine and is practical;
Conscientiousness: Closer to 1 means the user is disciplined and careful; closer to 0 means the user is impulsive and disorganized;
Extroversion: Closer to 1 means the user is sociable and fun-loving; closer to 0 means the user is reserved and thoughtful;
Agreeableness: Closer to 1 means the user is trusting and helpful; closer to 0 means the user is suspicious and uncooperative;
Neuroticism: Closer to 1 means the user is calm and confident; closer to 0 means the user is anxious and pessimistic.
Generate six profile sentences related to the given user’s persona, personaliity and the “personality trait, sport, media genre” in each sentence and a description about
the user’s personality:
1. I’m an introvert who loves spending time alone. (personality: introvert)
2. I always enjoy watching traditional dramas when I’m alone. (culture: traditional drama)
3. I’m an independent thinker who likes to go against the grain. (personality: independent thinker)
4. I’m a highly sensitive person who feels things deeply. (personality: highly sensitive)
5. I’m an empathetic person who feels connected to others’ emotions.(personality: empathetic)
6. I enjoy reading Shakespeare’s tragedies very much.(hobby: tragedy)
The description of (0.113, 0.857, 0.057, 0.86, 0.111): The user is introverted and may prefer spending time alone. The user shows a relatively high degree of
conscientiousness, being responsible and dedicated. The user also has a certain degree of neuroticism, being sensitive and having some challenges in emotional control.

### User’s persona categories:
1. Hobby,
2. Lifestyle,
3. Mood.
User’s big five personality: (0.974, 0.014, 0.897, 0.145, 0.955)
            """
        }
    ]
)
response = completion.choices[0].message.content
print(response)