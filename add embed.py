from openai import OpenAI 

client = OpenAI()
response = client.embeddings.create(
    input="A TMU community member reported to TMU Security that an individual approached them begging for change. The community member declined and the individual knocked the community member’s items out of their hand and left. Security located the individual and barred them from campus. The community member declined assistance from Toronto Police Service.",
    model="text-embedding-3-small",
    dimensions=256
)

print(response.data[0].embedding)
