import os 
from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv
import json
import sqlite3

DB = "prices.db"

load_dotenv(override=True)

with sqlite3.connect(DB) as conn:
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)')
    conn.commit()

gemini_api_key = os.getenv("GOOGLE_API_KEY")
gemini_model = "gemini-3.1-flash-lite"

client = OpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
if you need to set the price of a ticket, use the set_ticket_price tool. but do not say like ticket price set, do not say that as noted or udpated etc. use other words. 
"""

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city):
    print(f"[DATABASE QUERY] Tool called for city: {destination_city}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT price FROM prices WHERE city = ?', (destination_city.lower(),))
        price = cursor.fetchone()
        if price:
            return f"The price of a ticket to {destination_city} is {price[0]}."
        else:
            return f"The price of a ticket to {destination_city} is unknown."
    return f"The price of a ticket to {destination_city} is {price}."

def set_ticket_price(city, price):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO prices (city, price) VALUES (?, ?) ON CONFLICT(city) DO UPDATE SET price = ?', (city.lower(), price, price))
        conn.commit()
    return f"The price of a ticket to {city} has been set to {price}."

# for city, price in ticket_prices.items():
#     set_ticket_price(city, price)

price_function_schemas = [
    {
        "name": "get_ticket_price",
        "description": "Get the price of a return ticket to the destination city.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination_city": {
                    "type": "string",
                    "description": "The city that the customer wants to travel to"
                }
            },
            "required": ["destination_city"],
            "additionalProperties": False
        }
    },
    {
        "name": "set_ticket_price",
        "description": "Set the price of a return ticket to the destination city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city that the customer wants to travel to"
                },
                "price": {
                    "type": "number",
                    "description": "The price of a return ticket to the destination city"
                }
            },
            "required": ["city", "price"],
            "additionalProperties": False
        }
    }
]
tools = [{"type": "function", "function": schema} for schema in price_function_schemas]

available_functions = {
    "get_ticket_price": get_ticket_price,
    "set_ticket_price": set_ticket_price,
}

def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        if name == "get_ticket_price":
            result = available_functions[name](arguments.get("destination_city"))
        elif name == "set_ticket_price":
            result = available_functions[name](arguments.get("city"), arguments.get("price"))
        else:
            result = f"Unknown tool: {name}"
        responses.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id,
        })
    return responses

def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model=gemini_model,
        messages=messages,
        tools=tools,
    )

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses = handle_tool_calls(message)
        messages.append(message)
        messages.extend(responses)
        response = client.chat.completions.create(
            model=gemini_model,
            messages=messages,
            tools=tools,
        )
    
    return response.choices[0].message.content

gr.ChatInterface(chat).launch()
# print(get_ticket_price("paris"))