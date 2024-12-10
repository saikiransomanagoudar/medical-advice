# GenAI-Powered Medical Advice Chatbot with Rasa & LangGraph

A sophisticated AI-powered chatbot designed to provide medical advice using the Rasa framework and LangGraph. This project leverages natural language understanding (NLU) to deliver tailored responses to user queries, making healthcare assistance more accessible and user-friendly.

---

## Project Description

The **GenAI-Powered Medical Advice Chatbot** is a conversational assistant aimed at providing medical advice and guidance. It combines the robust capabilities of Rasa for managing dialogues and intents with LangGraph for advanced language processing. This tool is ideal for users seeking quick medical information or guidance in a conversational format.

---

## Features

- **Natural Language Understanding (NLU):** Enables the chatbot to process user queries with high accuracy.
- **Customizable Intents and Responses:** Easily expandable to handle various use cases in healthcare and beyond.
- **Integrates with Python Actions:** Extends functionality with custom actions written in Python.
- **Scenario-Based Training:** Includes predefined stories to handle complex conversation flows.
- **Interactive Dialogue Management:** Provides context-aware responses.
- **Lightweight and Scalable:** Designed to work efficiently in various deployment environments.

---

## Installation Instructions

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.7 or higher
- pip (Python package manager)

### Steps to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/saikiransomanagoudar/medical-advice
   cd medical-advice
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Mac: source venv/bin/activate
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Train the NLU and Core models:
   ```bash
   rasa train
5. Run the chatbot:
   ```bash
   rasa run
6. Start the action server:
   ```bash
   rasa run actions

### Usage
* To interact with the chatbot, use the Rasa shell:
  ```bash
  rasa shell
* For web or app integration, configure the endpoints.yml file and deploy on a server.

