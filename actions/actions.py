import os
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import openai
import logging
from openai import OpenAI
import time
from typing import Dict, List, Text, Any

# Load environment variables from the .env file
load_dotenv()

# Set OpenAI API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Logging setup
logging.basicConfig(level=logging.INFO)

# Helper class for GPT-3.5-turbo interaction
class GPTModel:
    @staticmethod
    def get_gpt_response(prompt: str) -> str:
        try:
            logging.info(f"Sending prompt to GPT-3.5: {prompt}")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            for _ in range(3):  # Retry 3 times in case of failure
                try:
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="gpt-3.5-turbo",
                        max_tokens=150
                    )
                    reply = response.choices[0].message.content.strip()
                    logging.info(f"OpenAI response: {reply}")
                    return reply
                except Exception as e:
                    logging.error(f"Error interacting with OpenAI: {e}, retrying...")
                    time.sleep(2)
            return "Sorry, I couldn't get a response. Please try again later."
        except Exception as e:
            logging.error(f"Final error interacting with OpenAI: {e}")
            return "Sorry, I couldn't get a response. Please try again later."

# Base class for agents
class AgentBase:
    def process_intent(self, prompt: str, tracker: Tracker = None) -> str:
        raise NotImplementedError("This method should be implemented by subclasses")

# Specialized agents
class GreetingAgent(AgentBase):
    def process_intent(self, prompt: str, tracker: Tracker = None) -> str:
        response = GPTModel.get_gpt_response(f"Greeting: {prompt}")
        return response.strip()  # Remove any leading/trailing spaces or unintended characters

class MedicineAgent(AgentBase):
    def process_intent(self, prompt: str, tracker: Tracker = None) -> str:
        response = GPTModel.get_gpt_response(f"Medicine: {prompt}")
        if "visit a hospital" in response.lower() or "see a doctor" in response.lower():
            hospital_agent = MedicalHospitalAgent()
            return hospital_agent.process_intent(response, tracker)
        return response

class MedicalHospitalAgent(AgentBase):
    def process_intent(self, prompt: str, tracker: Tracker = None) -> str:
        return GPTModel.get_gpt_response(f"Suggest hospitals for {prompt}")

class MedicalDepartmentAgent(AgentBase):
    def process_intent(self, prompt: str, tracker: Tracker = None) -> str:
        return GPTModel.get_gpt_response(f"Department: {prompt}")

# OperatorAgent handles routing between specialized agents
class OperatorAgent:
    def __init__(self):
        self.agents = {
            "greeting": GreetingAgent(),
            "medicine": MedicineAgent(),
            "hospital": MedicalHospitalAgent(),
            "department": MedicalDepartmentAgent()
        }

    def determine_intent(self, prompt: str, tracker: Tracker = None) -> str:
        try:
            logging.info(f"Determining intent for prompt: {prompt}")
            # Manual keyword matching for health-related intents
            if any(keyword in prompt.lower() for keyword in ["greet", "goodbye"]):
                return self.agents["greeting"].process_intent(prompt, tracker)
            elif any(keyword in prompt.lower() for keyword in ["medicine", "headache", "fever", "cold"]):
                return self.agents["medicine"].process_intent(prompt, tracker)
            elif "hospital" in prompt.lower():
                return self.agents["hospital"].process_intent(prompt, tracker)
            elif "department" in prompt.lower():
                return self.agents["department"].process_intent(prompt, tracker)
            else:
                return "Sorry, I couldn't determine your intent."
        except Exception as e:
            logging.error(f"Error determining intent: {e}")
            return "Sorry, something went wrong. Please try again later."

# UserProxyAgent handles interaction between user and operator
class UserProxyAgent:
    def send_prompt(self, prompt: str, tracker: Tracker = None) -> str:
        operator = OperatorAgent()
        return operator.determine_intent(prompt, tracker)
    
class ActionGreetUser(Action):
    def name(self) -> str:
        return "action_greet_user"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get('text')  # Get the user's greeting message
        response = GPTModel.get_gpt_response(f"Greeting: {user_message}")  # Send dynamic greeting to GPT
        dispatcher.utter_message(text=response)
        return []

class ActionProvideMedicine(Action):
    def name(self) -> str:
        return "action_provide_medicine"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Extract the 'disease' entity or slot from the user's input
        disease = tracker.get_slot("disease")  # Fetch the disease from the slot

        if not disease:
            # If no disease is detected, prompt the user to provide it
            dispatcher.utter_message(text="Please provide the name of the condition or disease you are experiencing.")
        else:
            # Use GPT-3.5 to generate advice for the given disease
            response = GPTModel.get_gpt_response(f"What medicine should I take for {disease}?")
            dispatcher.utter_message(text=response)
        
        return []

# Rasa action to handle the user's message dynamically
class ActionUseAutoGen(Action):
    def name(self) -> str:
        return "action_use_autogen"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get('text')

        # Initialize UserProxyAgent to handle interaction
        user_proxy = UserProxyAgent()

        # Send user input to the UserProxyAgent, which forwards it to OperatorAgent
        response = user_proxy.send_prompt(user_message, tracker)

        # Send the response back to the user
        dispatcher.utter_message(text=response)
        return []
