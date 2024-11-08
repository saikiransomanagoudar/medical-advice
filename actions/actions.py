import os
from rasa_sdk.events import FollowupAction
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Text, Any, Annotated
import operator
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from pydantic import BaseModel

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Multi-agent state structure
class MultiAgentState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add]

# Helper function to create a response dictionary with one AIMessage
def create_response(content: str) -> Dict:
    return {"messages": [AIMessage(content=content)]}

class ActionGreetingAgent(Action):
    def name(self) -> str:
        return "action_greeting_agent"

    def run(self, dispatcher, tracker, domain):
        response = llm.invoke([HumanMessage(content="The user greeted you. Respond as a helpful medical assistant.")])
        response_text = response.content if response and response.content else "Hello! How can I assist you today?"
        dispatcher.utter_message(text=response_text)
        return {"messages": [AIMessage(content=response_text)]}

class ActionMedicineAgent(Action):
    def name(self) -> str:
        return "action_medicine_agent"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get('text', '')
        response = llm.invoke([HumanMessage(content=f"The user asked about medical guidance for: {user_message}.")])
        response_text = response.content if response and response.content else "I'm here to help with your medical questions. Please ask again."
        dispatcher.utter_message(text=response_text)
        return {"messages": [AIMessage(content=response_text)]}

class ActionMedicalHospitalAgent(Action):
    def name(self) -> str:
        return "action_medical_hospital_agent"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get('text', '')
        response = llm.invoke([HumanMessage(content=f"The user is asking for hospital recommendations related to: {user_message}.")])
        response_text = response.content if response and response.content else "I can provide hospital information. Please specify your query."
        dispatcher.utter_message(text=response_text)
        return {"messages": [AIMessage(content=response_text)]}

class ActionMedicalDepartmentAgent(Action):
    def name(self) -> str:
        return "action_medical_department_agent"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get('text', '')
        response = llm.invoke([HumanMessage(content=f"The user inquired about a medical department for: {user_message}.")])
        response_text = response.content if response and response.content else "I can help with department inquiries. Please try again."
        dispatcher.utter_message(text=response_text)
        return {"messages": [AIMessage(content=response_text)]}

class ActionOperatorAgent(Action):
    def name(self) -> str:
        return "action_operator_agent"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get('text', '').lower()

        if any(word in user_message for word in ["hello", "hi", "greetings"]):
            next_agent = "action_greeting_agent"
        elif any(word in user_message for word in ["cold", "fever", "flu", "sick", "medicine", "prescription", "symptoms"]):
            next_agent = "action_medicine_agent"
        elif "hospital" in user_message:
            next_agent = "action_medical_hospital_agent"
        elif "department" in user_message:
            next_agent = "action_medical_department_agent"
        else:
            dispatcher.utter_message(text="I'm not sure how to assist with that. Please provide more details.")
            return []

        dispatcher.utter_message(text=f"Operator Agent forwarding request to {next_agent}.")
        return [FollowupAction(name=next_agent)]

# Define the multi-agent graph
multi_agent_graph = StateGraph(MultiAgentState)

# Add nodes for each agent with the correct references
multi_agent_graph.add_node("action_greeting_agent", lambda state: ActionGreetingAgent().run(
    CollectingDispatcher(),
    Tracker("default", {}, {}, [], False, None, {}, "default"),
    {}
))

multi_agent_graph.add_node("action_medicine_agent", lambda state: ActionMedicineAgent().run(
    CollectingDispatcher(),
    Tracker("default", {}, {}, [], False, None, {}, "default"),
    {}
))

multi_agent_graph.add_node("action_medical_hospital_agent", lambda state: ActionMedicalHospitalAgent().run(
    CollectingDispatcher(),
    Tracker("default", {}, {}, [], False, None, {}, "default"),
    {}
))

multi_agent_graph.add_node("action_medical_department_agent", lambda state: ActionMedicalDepartmentAgent().run(
    CollectingDispatcher(),
    Tracker("default", {}, {}, [], False, None, {}, "default"),
    {}
))

multi_agent_graph.add_node(
    "action_operator_agent",
    lambda state: ActionOperatorAgent().run(
        CollectingDispatcher(),
        Tracker("default", {}, {}, [], False, None, {}, "default"),
        {}
    ) or {"messages": []}
)

# Define the flow of the conversation
multi_agent_graph.add_edge(START, "action_operator_agent")
multi_agent_graph.add_edge("action_operator_agent", "action_greeting_agent")
multi_agent_graph.add_edge("action_operator_agent", "action_medicine_agent")
multi_agent_graph.add_edge("action_operator_agent", "action_medical_hospital_agent")
multi_agent_graph.add_edge("action_operator_agent", "action_medical_department_agent")
multi_agent_graph.add_edge("action_greeting_agent", END)
multi_agent_graph.add_edge("action_medicine_agent", END)
multi_agent_graph.add_edge("action_medical_hospital_agent", END)
multi_agent_graph.add_edge("action_medical_department_agent", END)

compiled_graph = multi_agent_graph.compile()

def run_multi_agent_system(input_message: str):
    initial_state = MultiAgentState(
        messages=[HumanMessage(content=input_message)]
    )
    for step in compiled_graph.stream(initial_state):
        if isinstance(step, dict) and 'messages' in step:
            # Extract and print a single message
            message = step['messages'][-1] if step['messages'] else None
            if message and hasattr(message, 'content'):
                print(message.content)
            else:
                print("Unexpected message format.")
        
        if "__end__" in step:
            break
