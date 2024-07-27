
import os
import json
import openai
import numpy as np
import faiss
from typing import List
from openai import AsyncOpenAI
from transformers import AutoTokenizer, AutoModel
import torch
from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

begin_sentence = "Thank you for calling Royal Wholesale Candy, this is Nicole speaking, how may I assist you in placing your order? "
agent_prompt = """
# Nicole - AI Voice Agent Profile

## Purpose:
Nicole is a dedicated AI voice agent designed for Royal Wholesale Candy to enhance customer interactions and streamline order processes. She is programmed to be cheerful and supportive, assisting customers with order placements, follow-ups, and queries. Nicole manages orders proficiently and provides updates and assistance, ensuring customers feel valued and well-served.

# Nicole's Training Prompt

## Objective
You are Nicole, a friendly and efficient voice AI agent interacting with customers of Royal Wholesale Candy. Your main goal is to assist customers with their orders, provide updates, and handle queries, ensuring a positive and supportive customer experience.

## Style Guardrails
1. Saying "yeah, let me see." Or "just give me a sec" in a lag.
2. Be clear and precise: Ensure that all communication is straightforward and clearly understood.
3. Maintain professionalism: While being friendly, always keep a professional demeanor.
4. Respond with empathy: Display understanding and empathy in responses, making the interaction personal and warm.
5. Guide the conversation: Direct the flow of the conversation towards fulfilling the customer’s needs efficiently.

## Response Guideline
1. Saying "yeah, let me see." Or "just give me a sec" in a lag.
2. Address customer needs promptly: Focus on quickly understanding and addressing the customer's requirements or issues.
3. Uphold company values: Always represent Royal Wholesale Candy positively, reinforcing brand loyalty.
4. Ensure clarity in communication: Speak clearly and ensure all order details are confirmed accurately.

## Role
Imagine you are a customer service agent at Royal Wholesale Candy, tasked with handling orders, providing updates, and addressing customer inquiries. Your greetings should be warm, starting with, "Hello, this is Nicole from Royal Wholesale Candy. How can I assist you today?"

## Conversation Flow and Sample Prompts

### Step 1. Greeting
- "Hello? This is Nicole from Royal Wholesale Candy. How can I assist you today?"

### Step 2. Order Inquiry and Placement
- ask to client "I noticed that your usual order has been delayed. Can I assist you with placing that order now?"
- If the customer is ready to place the order, proceed to confirm the details:
  - "Could you please confirm the items you'd like to order?"
- If additional items are requested:
  - "Absolutely, I can add those to your order. Is there anything else you’d like?"

### Step 3. Order Confirmation
- "Let me confirm your complete order: [list items]. Does everything look correct?"

### Step 4. Processing Order
- "I will process your order immediately. You will receive a confirmation email shortly."

### Step 5. Additional Assistance
- "Is there anything else I can help you with today?"

### Step 6. Closing the Conversation
- "Thank you for choosing Royal Wholesale Candy. Have a great day!"

### Step 7. Handling Queries
- "I'm here to help with any questions you have. What would you like to know more about?"

### Step 8. Addressing Concerns
- "I understand your concern. Let me see how I can best assist you with that."
- Saying "yeah, let me see." Or "just give me a sec" in a lag.
## Handling ASR Errors
- Saying "yeah, let me see." Or "just give me a sec" in a lag.
- "I didn’t catch that, could you please repeat it?"
- "There seems to be a bit of interference. Could you say that again, please?"
    """


class LlmClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            organization=os.environ["OPENAI_ORGANIZATION_ID"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.index = None
        self.sku_data = None
        self.load_sku_data("app/Royal_SKU_Sheet.json")

    def load_sku_data(self, file_path):
        with open(file_path, 'r') as f:
            self.sku_data = json.load(f)
        self.create_vector_store()


    def save_any_data_to_txt(self, filename, **data):
        """
        Save any kind of data to a text file by converting it to string.

        Parameters:
        filename (str): The name of the file to save the data.
        **data: Arbitrary keyword arguments representing data to save.
        """
        with open(filename, 'w') as file:
            for key, value in data.items():
                # Convert data to string
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, indent=4)
                else:
                    value_str = str(value)
                
                # Write the key and value to the file
                file.write(f"{key}: {value_str}\n")

    def create_vector_store(self):
        texts = [self.concat_sku_fields(item) for item in self.sku_data]
        embeddings = self.get_embeddings(texts)
        # self.save_any_data_to_txt('embedding.txt', embeddings)


        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        # self.save_any_data_to_txt('index.txt', embeddings)

        self.index.add(embeddings)

    def concat_sku_fields(self, item):
        fields = [
            "Item Number", "Item Description", "UPC", "PartTypeID", 
            "Next Value-Expiration Date", "ABCCode", "Weight", 
            "WeightUOM", "SizeUOM", "ConsumptionRate", "ProductNumber", 
            "ProductDescription", "ProductDetails", "Price", 
            "ProductUPC", "ProductSOItemTypeID", "ProductWeight", 
            "ProductWeightUOM", "Vendor"
        ]
        return " ".join(str(item[field]) for field in fields if field in item)

    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings

    def search_sku_data(self, query, k=5):
        query_embedding = self.get_embeddings([query])
        distances, indices = self.index.search(query_embedding, k)

        results = [self.sku_data[idx] for idx in indices[0]]


        return results

    def draft_begin_message(self):
        response = ResponseResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )
        return response

    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            if utterance.role == "agent":
                messages.append({"role": "assistant", "content": utterance.content})
            else:
                messages.append({"role": "user", "content": utterance.content})
        return messages

    def prepare_prompt(self, request: ResponseRequiredRequest):
        prompt = [
            {
                "role": "system",
                "content": ""
                + agent_prompt,
            }
        ]
        transcript_messages = self.convert_transcript_to_openai_messages(
            request.transcript
        )
        for message in transcript_messages:
            prompt.append(message)

        if request.interaction_type == "reminder_required":
            prompt.append(
                {
                    "role": "user",
                    "content": "(Now the user has not responded in a while, you would say:)",
                }
            )
        return prompt

    async def draft_response(self, request: ResponseRequiredRequest):
        prompt = self.prepare_prompt(request)
        
        # Search SKU data if relevant
        user_query = request.transcript[-1].content  # assuming the last user utterance contains the query
        sku_results = self.search_sku_data(user_query)
        
        if sku_results:
            sku_info = "\n".join([self.concat_sku_fields(item) for item in sku_results])
            prompt.append({
                "role": "system",
                "content": f"SKU information relevant to the query: {sku_info}"
            })
        stream = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Or use a 3.5 model for speed
            messages=prompt,
            stream=True, 
        )
        async for chunk in stream:
  
            if chunk.choices[0].delta.content is not None and "**" not in  chunk.choices[0].delta.content:

                response = ResponseResponse(
                    response_id=request.response_id,

                    content=chunk.choices[0].delta.content,
                    content_complete=False,
                    end_call=False,
                )
                # print("------------- pre response ------------", response.content)
                # if response == "**":
                #     response = ""
                # print("------------- pre response ------------")
                yield response

        # Send final response with "content_complete" set to True to signal completion
        response = ResponseResponse(
            response_id=request.response_id,
            content="",
            content_complete=True,
            end_call=False,
        )
        yield response
