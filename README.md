# GenAI-Powered Medical Advice Chatbot with Autogen

This project is a chatbot powered by **Rasa** and **OpenAI GPT-3.5**, utilizing **Autogen** for dynamic interaction generation. The bot provides medical advice related to symptoms, conditions, and treatment options. It leverages multiple specialized agents (such as **GreetingAgent**, **MedicineAgent**, **MedicalHospitalAgent**, and **MedicalDepartmentAgent**) to handle different user queries in healthcare contexts.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Agents Overview](#agents-overview)
- [File Structure](#file-structure)

## Overview

This chatbot interacts with users to provide AI-powered (GenAI) medical advice for conditions like cold, fever, headache, and more. It dynamically generates responses using **OpenAI's GPT-3.5** model through the **Autogen** process, ensuring accurate and personalized recommendations.

## Features

- Provides dynamic, AI-generated medical advice using **GenAI** (GPT-3.5).
- Suggests hospitals or departments to visit based on symptoms.
- Utilizes **Autogen** through multiple specialized agents for flexible, real-time responses.
- Integrates Rasa for managing user intent and dialogue flow.
- Calls **OpenAI's GPT-3.5** API for detailed, real-time medical responses.

## Installation

### Prerequisites
- Python 3.8+
- Rasa 2.x
- OpenAI API Key
- `dotenv` for managing environment variables

### Steps to Install

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv rasa-venv
   rasa-venv\Scripts\activate
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
4. **Set up environment variables:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
5. **Train the Rasa model:
   ```bash
   rasa train
6. **Run the Rasa action server:
   ```bash
   rasa run actions
7. Run the chatbot:
   ```bash
   rasa shell
