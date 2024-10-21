import os
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import openai
import logging
from openai import OpenAI
import time
from typing import Dict, List, Text, Any

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
logging.basicConfig(level=logging.INFO)

class GPTModel:
    @staticmethod
    def get_gpt_response(prompt: str) -> str:
        try:
            logging.info(f"Sending prompt to GPT-3.5: {prompt}")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            for _ in range(3):
                try:
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="gpt-3.5-turbo",
                        max_tokens=150
                    )
                    reply = response.choices[0].message.content.strip()
                    if reply.lower().startswith("response:"):
                        reply = reply[len("response:"):].strip()
                    logging.info(f"OpenAI response: {reply}")
                    return reply
                except Exception as e:
                    logging.error(f"Error interacting with OpenAI: {e}, retrying...")
                    time.sleep(2)
            return "Sorry, I couldn't get a response. Please try again later."
        except Exception as e:
            logging.error(f"Final error interacting with OpenAI: {e}")
            return "Sorry, I couldn't get a response. Please try again later."

class AgentBase:
    def process_intent(self, prompt: str, tracker: Tracker = None) -> str:
        raise NotImplementedError("This method should be implemented by subclasses")

class GreetingAgent(AgentBase):
    def process_intent(self, prompt: str, tracker: Tracker = None) -> str:
        response = GPTModel.get_gpt_response(f"Greeting: {prompt}")
        return response.strip()

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
    
class HealthAdviceAgent(AgentBase):
    def process_intent(self, prompt: str, tracker: Tracker = None) -> str:
        response = GPTModel.get_gpt_response(f"Health Advice: {prompt}")
        logging.info(f"GPT-3.5 response for health advice: {response}")

        if "specialist" in response.lower() or "department" in response.lower():
            department_agent = MedicalDepartmentAgent()
            return department_agent.process_intent(response, tracker)
        
        return response

class OperatorAgent:
    def __init__(self):
        self.agents = {
            "greeting": GreetingAgent(),
            "medicine": MedicineAgent(),
            "hospital": MedicalHospitalAgent(),
            "department": MedicalDepartmentAgent(),
            "health_advice": HealthAdviceAgent()
        }

    def determine_intent(self, prompt: str, tracker: Tracker = None) -> str:
        try:
            logging.info(f"Determining intent for prompt: {prompt}")
            if any(keyword in prompt.lower() for keyword in ["greet", "goodbye"]):
                return self.agents["greeting"].process_intent(prompt, tracker)
            elif any(keyword in prompt.lower() for keyword in ["medicine", "headache", "fever", "cold"]):
                return self.agents["medicine"].process_intent(prompt, tracker)
            elif "hospital" in prompt.lower():
                return self.agents["hospital"].process_intent(prompt, tracker)
            elif "department" in prompt.lower():
                return self.agents["department"].process_intent(prompt, tracker)
            elif "advice" in prompt.lower() or "health" in prompt.lower() or "stress" in prompt.lower():
                return self.agents["health_advice"].process_intent(prompt, tracker)
            else:
                return "Sorry, I couldn't determine your intent."
        except Exception as e:
            logging.error(f"Error determining intent: {e}")
            return "Sorry, something went wrong. Please try again later."

class UserProxyAgent:
    def send_prompt(self, prompt: str, tracker: Tracker = None) -> str:
        operator = OperatorAgent()
        return operator.determine_intent(prompt, tracker)
    
class ActionGreetUser(Action):
    def name(self) -> str:
        return "action_greet_user"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get('text')
        response = GPTModel.get_gpt_response(f"Greeting: {user_message}") 
        dispatcher.utter_message(text=response)
        return []

class ActionProvideMedicine(Action):
    def name(self) -> str:
        return "action_provide_medicine"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        disease = tracker.get_slot("disease")  # Fetch the disease from the slot

        if not disease:
            dispatcher.utter_message(text="Please provide the name of the condition or disease you are experiencing.")
        else:
            response = GPTModel.get_gpt_response(f"What medicine should I take for {disease}?")
            dispatcher.utter_message(text=response)
        
        return []

class ActionUseAutoGen(Action):
    def name(self) -> str:
        return "action_use_autogen"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get('text')

        user_proxy = UserProxyAgent()

        response = user_proxy.send_prompt(user_message, tracker)

        dispatcher.utter_message(text=response)
        return []
