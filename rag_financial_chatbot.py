from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import mistralai
from mistralai import Mistral
import os

qa_data = [
    {
        "question": "What is FinTechX?",
        "answer": "FinTechX is an online platform that offers a range of financial services, including personal loans, investments, and budgeting tools."
    },
    {
        "question": "How can I open an account with FinTechX?",
        "answer": "To open an account with FinTechX, visit our website, click on 'Sign Up', and complete the registration process by providing your personal information and verifying your identity."
    },
    {
        "question": "What investment options are available on FinTechX?",
        "answer": "FinTechX offers a variety of investment options, including stocks, ETFs, mutual funds, and robo-advisory services for automated portfolio management."
    },
    {
        "question": "How do I apply for a personal loan on FinTechX?",
        "answer": "To apply for a personal loan, log in to your FinTechX account, navigate to the 'Loans' section, and follow the steps to complete your loan application. Approval is subject to credit checks."
    },
    {
        "question": "What are the fees for investing with FinTechX?",
        "answer": "FinTechX charges a 0.25% annual management fee on all invested assets. There are no fees for account opening or transfers, but transaction fees may apply for specific products."
    },
    {
        "question": "How does FinTechX ensure the security of my financial data?",
        "answer": "FinTechX uses state-of-the-art encryption technologies to protect your data, including two-factor authentication, data encryption, and continuous monitoring for fraudulent activities."
    },
    {
        "question": "Can I access FinTechX on mobile devices?",
        "answer": "Yes, FinTechX offers a mobile app available on both iOS and Android. You can manage your investments, apply for loans, and track your finances on the go."
    },
    {
        "question": "How long does it take to withdraw funds from FinTechX?",
        "answer": "Withdrawals from FinTechX typically take 1-3 business days to process, depending on your bank’s processing time."
    },
    {
        "question": "Does FinTechX offer customer support?",
        "answer": "Yes, FinTechX offers 24/7 customer support through live chat, email, and phone. You can also access a comprehensive help center for common queries."
    },
    {
        "question": "What is FinTechX's minimum deposit requirement?",
        "answer": "The minimum deposit requirement for FinTechX investment accounts is $500. There is no minimum deposit for personal loan applications."
    }
]


# 1. Connect to ElasticSearch :

es_username = "elastic"
es_password = "GLKCYmtANmxVwVxdNZum"

# Create an Elasticsearch client with authentication
es = Elasticsearch(
    ["http://localhost:9200"],
    basic_auth=(es_username, es_password)
)

# Test the connection
try:
    if es.ping():
        print('\n' , "Successefully Connected to Elasticsearch!" , '\n')
    else:
        print("Could not connect to Elasticsearch")
except Exception as e:
    print(f"An error occurred: {e}")



# 2. Index the documents : 

index_name = 'fintechx_faq_vector'

mapping = {
    "mappings": {
        "properties": {
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "question_vector": {"type": "dense_vector", "dims": 384}  # Adjust dims based on your embedding model
        }
    }
}

# Create the index in Elasticsearch
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)



# 3. Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Index each FAQ in Elasticsearch

for i, qa in enumerate(qa_data):
    question = qa['question']
    embedding = model.encode(question)  # Generate the vector embedding

    # Create a document with the question, answer, and vector
    doc = {
        "question": question,
        "answer": qa['answer'],
        "question_vector": embedding.tolist()  # Convert vector to list
    }

    # 5. Index the document in Elasticsearch
    es.index(index=index_name, id=i, body=doc)




# 6. Convert the user prompt into a vector and search for similar vector embeddings :

def search_similar_question(prompt):
    query_vector = model.encode(prompt)  # Convert user prompt to vector

    # Define a query for vector search using cosine similarity
    query = {
        "size": 1,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                    "params": {"query_vector": query_vector.tolist()}
                }
            }
        }
    }

    # 7. Perform the search in Elasticsearch
    response = es.search(index=index_name, body=query)
    return response['hits']['hits'][0]['_source']



# 8. Enhance response with MISTRAL AI : 


api_key = os.environ["MISTRAL_API_KEY"] = 'joukTqVPkc1Z7XI34QIE2vmEGyncaNsy'


"""
def generate_enhanced_answer(prompt, context):
    response = mistralai.Completion.create(
        model="mistral",
        prompt=f"{context}\nUser: {prompt}\nBot:",
        max_tokens=150
    )
    return response['choices'][0]['text'].strip()
"""


def generate_enhanced_answer(prompt, context, api_key):
    client = Mistral(api_key=api_key)
    
    response = client.chat.complete(
        model="mistral-large-latest", 
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    
    return response.choices[0].message.content.strip()