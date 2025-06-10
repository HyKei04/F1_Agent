# F1 Multi-agent system 

## This repository contains the code implementing the F1 multi-agent system developed in the Bachelor's Thesis "Introducció als agents d'IA basats en LLMs: Disseny d'un sistema multiagent per a la visualització i consulta de dades de Fórmula 1"

### The main funcionalities of this system are:

- Graph generation via user query (chatbot) based on F1 data such as telemetry, timing information or race results.
- The system can retrieve data from 2018 onwards from an F1 API (https://github.com/theOehrly/Fast-F1). This data includes:
    - Telemetry data for the fastest lap of each driver during a specific session of the Grand Prix
    - Lap data (lap time, position, tyre used, stint duration, etc.) for every lap of a driver during a specific session (useful for race comparisons)
    - Fastest lap data for every driver in a session (can also be obtained with the data above, but it's implemented to simplify agent workflow)
    - Session results data (position, points, quali position, Q1, Q2, Q3 laptimes and more).
- It can also perform web searches to retrieve more general information from the internet (like current standings) and build graphs with them.
- It also has information about driver penalties and car upgrades (stored in a vector database) for the 2025 Season (until Spanish GP included) and the current sporting regulation to give additional info.
- When a graph is generated, the system can analyse it for you, just type "Analyse this graph".

To use this system an OpenRouter API Key is needed to use a LLM (gpt-4o-mini is the one used in this project) (https://openrouter.ai/openai/gpt-4o-mini). Also, to performs web searches, a Serper API key is also needed (https://serper.dev/).
The structure of the .env should look like this:
```
OPENROUTER_API_KEY="your_api_key"
SERPER_API_KEY="your_api_key"
```

In the examples folder, there are images with graphs built using this agent.
