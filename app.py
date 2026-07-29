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

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": price_function}]

def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get("destination_city")
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
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
        messages.append(responses)
        response = client.chat.completions.create(
            model=gemini_model,
            messages=messages,
            tools=tools,
        )
    
    return response.choices[0].message.content

gr.ChatInterface(chat).launch()
# print(get_ticket_price("paris"))