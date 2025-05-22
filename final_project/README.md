```
cd final_project

docker build -t ai-agents-final-project .

docker run -it --rm -v "$(pwd):/app" --env="ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" ai-agents-final-project python app.py  "You are an AI assistant tasked with helping a user plan a trip to Paris. The user wants recommendations for historical landmarks, local cuisine, and transportation options. Provide a detailed itinerary for a 3-day trip, including costs and travel times."
```

or

docker run -it --rm -v "$(pwd):/app" --env="ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" ai-agents-final-project python gaia.py