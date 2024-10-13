from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.models import FlexSendMessage, TextSendMessage, QuickReply, QuickReplyButton, MessageAction
import requests
from bs4 import BeautifulSoup
import json
import time
from selenium import webdriver
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Neo4j connection details
URI = "neo4j://localhost"
AUTH = ("neo4j", "6410110004")

ACCESS_TOKEN = 'ACCESS_TOKEN Line'
SECRET = 'SECRET Line'
line_bot_api = LineBotApi(ACCESS_TOKEN)
handler = WebhookHandler(SECRET)

# Function to run Neo4j query
def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

# Function to upsert user and log conversation to Neo4j
def upsert_user_and_log_conversation(user_id, question, response,scraped_text=None):
    timestamp = datetime.now().isoformat()
    # Ensure the user exists in the database
    query = '''
    MERGE (u:User {uid: $user_id})
    SET u.last_keyword = $question  // Store the last keyword searched
    CREATE (c:Chat {question: $question, timestamp: $timestamp})
    CREATE (b:Bot {response: $response, scraped_text: $scraped_text, timestamp: $timestamp})
    MERGE (u)-[:QUESTION]->(c)-[:ANSWER]->(b)
    '''
    parameters = {
        'user_id': user_id,
        'question': question,
        'response': response,
        'scraped_text': scraped_text,
        'timestamp': timestamp
    }
    
    run_query(query, parameters)

# Function to retrieve last keyword from Neo4j
def get_last_keyword(user_id):
    query = '''
    MATCH (u:User {uid: $user_id})
    RETURN u.last_keyword AS last_keyword
    '''
    parameters = {'user_id': user_id}
    result = run_query(query, parameters)
    
    if result and result[0]['last_keyword']:
        return result[0]['last_keyword']
    return None

encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def create_faiss_index(phrases):
    vectors = encoder.encode(phrases)
    vector_dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index, vectors

intent_phrases = [
    "สวัสดี",
    "สอบถาม",
    "แนะนำเมกอัป",
    "แนะนำการบำรุงผิว",
    "แนะนำผลิตลดสิว",
    "แนะนำผิวกระจ่างใส",
    "ปาก",
    "ดวงตา",
    "โทนเนอร์",
    "คลีนเซอร์",
    "เซรั่ม", 
    "มอยเจอร์ไรเซอร์", 
    "ใบหน้า",
    "อุปกรณ์ดูแลผิวและแต่งหน้า",
    "ครีมกันแดด",
    "คุชชั่น",
    "คอนซีลเลอร์",
    "รองพื้น",
    "แป้งฝุ่น",
    "แป้งแข็ง",
    "พาเลท",
    "ดินสอเขียนคิ้ว",
    "มาสคาร่า",
    "ขอบคุณ",
]
index, vectors = create_faiss_index(intent_phrases)

def faiss_search(sentence):
    search_vector = encoder.encode(sentence)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    distances, ann = index.search(_vector, k=1)

    distance_threshold = 0.5
    if distances[0][0] > distance_threshold:
        return 'unknown'
    else:
        return intent_phrases[ann[0][0]]

def llama_change(response):
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }

    full_prompt = f"{response}"
    
    print("full_prompt",full_prompt)
    
    payload = {
        "model": "supachai/llama-3-typhoon-v1.5",  
        "prompt": " คำตอบ หรือ response ไม่เกิน 10 คำ และเป็นภาษาไทยเท่านั้น"+full_prompt,      
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            response_data = response.json()
            return response_data["response"]
        else:
            return "ขอโทษครับ ไม่สามารถประมวลผลได้ในขณะนี้"
    except Exception as e:
        return f"เกิดข้อผิดพลาด: {e}"



# Function to perform web scraping for skincare or makeup products
def scrape_amway(url):
    try:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(5)        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()
        
        if "Categories" in url:
            return scrape_category_page(soup)  # Category page
        elif "search" in url:
            return scrape_search_page(soup)    # Search page
        return None

    except Exception as e:
        print(f"Error scraping website: {e}")
        return None

# Scraping for a Category Page (e.g., https://www.amway.co.th/Categories/...)
def scrape_category_page(soup):
    result = []
    
    # Find product cards on category page
    product_elements = soup.find_all("div", {"class": "amway-pd-card no-select"})
    
    base_url = "https://www.amway.co.th"
    
    for product in product_elements:
        title_element = product.find("a", class_="amway-pd-card-name")
        price_element = product.find("div", class_="amway-pd-card-price-bolder")
        image_element = product.find("img", class_="amway-pd-card-img")
        link_tag = title_element.get('href') if title_element else None
        
        # Construct the full product URL
        product_url = base_url + link_tag if link_tag else None
        
        # Ensure both title and price are found and have non-empty text
        title_text = title_element.text.strip() if title_element else 'No title'
        price_text = price_element.text.strip() if price_element else 'No price'
        
        if title_text and price_text and product_url:
            result.append({
                'title': title_text,
                'price': price_text,
                'image': image_element['src'] if image_element else None,
                'url': product_url
            })
    
    return result

# Scraping for a Search Page (e.g., https://www.amway.co.th/search/?text=...)
def scrape_search_page(soup):
    result = []
    
    # Find product cards on search result page
    product_elements = soup.find_all("div", {"class": "product-list__item js-qty-selector clearfix"})
    
    base_url = "https://www.amway.co.th"
    
    for product in product_elements:
        title_element = product.find("a", class_="product-list__item-title")
        price_element = product.find("span", class_="product-list__item-abovalue")
        image_element = product.find("img", class_="product-list__thumbnail")
        link_tag = title_element.get('href') if title_element else None
        
        # Construct the full product URL
        product_url = base_url + link_tag if link_tag else None
        
        # Ensure both title and price are found and have non-empty text
        title_text = title_element.text.strip() if title_element else 'No title'
        price_text = price_element.text.strip() if price_element else 'No price'
        
        if title_text and price_text and product_url:
            result.append({
                'title': title_text,
                'price': price_text,
                'image': image_element['src'] if image_element else None,
                'url': product_url
            })
    
    return result

# Function to send Flex Message with product details
def send_flex_message(reply_token, products):
    if not products:
        text_message = TextSendMessage(text="ไม่พบสินค้า")
        line_bot_api.reply_message(reply_token, text_message)
        return

    bubbles = [{
        "type": "bubble",
        "hero": {
            "type": "image",
            "url": prod['image'],
            "size": "full",
            "aspectRatio": "1:1",
            "aspectMode": "cover",
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {"type": "text", "text": prod['title'], "weight": "bold", "size": "md", "wrap": True},
                {"type": "text", "text": f"Price: {prod['price']}", "size": "lg", "color": "#999999"}
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",  # Use primary style for filled button
                    "height": "sm",
                    "color": "#6495ED",  # Set the button color to blue
                    "action": {
                        "type": "uri",
                        "label": "View Product",
                        "uri": prod['url']
                    }
                }
            ]
        }
    } for prod in products[:8]]

    contents = {"type": "carousel", "contents": bubbles}

    flex_message = FlexSendMessage(alt_text="Product List", contents=contents)
    line_bot_api.reply_message(reply_token, messages=[flex_message])

# ฟังก์ชันหลักในการจัดการข้อความที่เข้ามาจากผู้ใช้
@app.route("/", methods=['POST'])
def linebot():
    body = request.get_json()

    try:
        # ตรวจสอบว่ามี events ใน request body หรือไม่
        if 'events' not in body or len(body['events']) == 0:
            return 'No events found', 400
        
        event = body['events'][0]
        if 'replyToken' not in event or 'message' not in event:
            return 'Invalid event structure', 400

        reply_token = event['replyToken']
        original_msg = event['message'].get('text', '').strip().lower()
        user_id = event['source']['userId']

        # เรียกใช้ faiss_search เพื่อค้นหา Intent
        msg = faiss_search(original_msg)
        
        # ตรวจสอบว่าผู้ใช้ส่งข้อความทักทาย "สวัสดี" หรือไม่
        if msg == "สวัสดี":         
            response = "สวัสดี ฉันชื่อ AmBeauty Bot มาให้คำปรึกษาเกี่ยวกับผลิตภัณฑ์เสริมความงามของ Amway"
            line_bot_api.reply_message(reply_token, TextSendMessage(
                text=response,
                quick_reply=QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="แนะนำการบำรุงผิว", text="แนะนำการบำรุงผิว")),
                    QuickReplyButton(action=MessageAction(label="แนะนำเมกอัป", text="แนะนำเมกอัป")),
                    QuickReplyButton(action=MessageAction(label="แนะนำผลิตลดสิว", text="แนะนำผลิตลดสิว")),
                    QuickReplyButton(action=MessageAction(label="แนะนำผิวกระจ่างใส", text="แนะนำผิวกระจ่างใส"))
                ])
            ))
            upsert_user_and_log_conversation(user_id, msg, response)
            return 'OK'


        # Handle the second-level quick reply for "บำรุงผิว"
        elif msg == "แนะนำการบำรุงผิว":
            response_prompt = "แนะนำการบำรุงผิวหน้า:"
            response = llama_change(response_prompt)
            response = response+"\nเลือกประเภทผลิตภัณฑ์บำรุงผิวหน้า:"
            line_bot_api.reply_message(reply_token, TextSendMessage(
                text=response,
                quick_reply=QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="โทนเนอร์", text="โทนเนอร์")),
                    QuickReplyButton(action=MessageAction(label="คลีนเซอร์", text="คลีนเซอร์")),
                    QuickReplyButton(action=MessageAction(label="เซรั่ม", text="เซรั่ม")),
                    QuickReplyButton(action=MessageAction(label="มอยเจอร์ไรเซอร์", text="มอยเจอร์ไรเซอร์"))
                ])
            ))
            upsert_user_and_log_conversation(user_id, original_msg, response)
            return 'OK'

        elif msg == "แนะนำเมกอัป":
            response = "เลือกประเภทผลิตภัณฑ์เมกอัป:"
            line_bot_api.reply_message(reply_token, TextSendMessage(
                text= response,
                quick_reply=QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="ใบหน้า", text="ใบหน้า")),
                    QuickReplyButton(action=MessageAction(label="ดวงตา", text="ดวงตา")),
                    QuickReplyButton(action=MessageAction(label="ปาก", text="ปาก")),
                    QuickReplyButton(action=MessageAction(label="อุปกรณ์ดูแลผิวและแต่งหน้า", text="อุปกรณ์ดูแลผิวและแต่งหน้า"))
                ])
            ))
            upsert_user_and_log_conversation(user_id, msg, response)
            return 'OK'
        
        elif msg == "ใบหน้า":
            response = "เลือกประเภทผลิตภัณฑ์ใบหน้า:"
            line_bot_api.reply_message(reply_token, TextSendMessage(
                text=response,
                quick_reply=QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="ครีมกันแดด", text="ครีมกันแดด")),
                    QuickReplyButton(action=MessageAction(label="คุชชั่น", text="คุชชั่น")),
                    QuickReplyButton(action=MessageAction(label="คอนซีลเลอร์", text="คอนซีลเลอร์")),
                    QuickReplyButton(action=MessageAction(label="รองพื้น", text="รองพื้น")),
                    QuickReplyButton(action=MessageAction(label="แป้งฝุ่น", text="แป้งฝุ่น")),
                    QuickReplyButton(action=MessageAction(label="แป้งแข็ง", text="แป้งแข็ง")),
                                    ])
            ))
            upsert_user_and_log_conversation(user_id, msg, response)
            return 'OK'

        elif msg == "ดวงตา":
            response = "เลือกประเภทผลิตภัณฑ์ดวงตา:"
            line_bot_api.reply_message(reply_token, TextSendMessage(
                text=response,
                quick_reply=QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="พาเลท", text="พาเลท")),
                    QuickReplyButton(action=MessageAction(label="ดินสอเขียนคิ้ว", text="ดินสอเขียนคิ้ว")),
                    QuickReplyButton(action=MessageAction(label="มาสคาร่า", text="มาสคาร่า")),
                                    ])
            ))
            
            upsert_user_and_log_conversation(user_id, msg, response)
            
            return 'OK'
        
        elif msg == "สอบถาม":
            response = "คุณต้องการสอบถามเกี่ยวกับอะไร?"
            line_bot_api.reply_message(reply_token, TextSendMessage(
                text=response,
                quick_reply=QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="แนะนำการบำรุงผิว", text="แนะนำการบำรุงผิว")),
                    QuickReplyButton(action=MessageAction(label="แนะนำเมกอัป", text="แนะนำเมกอัป")),
                    QuickReplyButton(action=MessageAction(label="แนะนำผลิตลดสิว", text="แนะนำผลิตลดสิว")),
                    QuickReplyButton(action=MessageAction(label="แนะนำผิวกระจ่างใส", text="แนะนำผิวกระจ่างใส"))
                ])
            ))
            upsert_user_and_log_conversation(user_id, msg, response)
            return 'OK'
        
        elif msg == "ขอบคุณ":
            response = "ด้วยความยินดีค่ะ"
            line_bot_api.reply_message(reply_token, TextSendMessage(text=response))
            upsert_user_and_log_conversation(user_id, msg, response)
            return 'OK'
        
        
        
        # Mapping URLs to user input for specific skincare products
        url_map = {
            "โทนเนอร์": "https://www.amway.co.th/Categories/%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%87%E0%B8%B2%E0%B8%A1/%E0%B8%9A%E0%B8%B3%E0%B8%A3%E0%B8%B8%E0%B8%87%E0%B8%9C%E0%B8%B4%E0%B8%A7/%E0%B9%82%E0%B8%97%E0%B8%99%E0%B9%80%E0%B8%99%E0%B8%AD%E0%B8%A3%E0%B9%8C/c/toner",
            "คลีนเซอร์": "https://www.amway.co.th/Categories/%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%87%E0%B8%B2%E0%B8%A1/%E0%B8%9A%E0%B8%B3%E0%B8%A3%E0%B8%B8%E0%B8%87%E0%B8%9C%E0%B8%B4%E0%B8%A7/%E0%B8%84%E0%B8%A5%E0%B8%B5%E0%B8%99%E0%B9%80%E0%B8%8B%E0%B8%AD%E0%B8%A3%E0%B9%8C/c/cleanser",
            "เซรั่ม": "https://www.amway.co.th/Categories/%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%87%E0%B8%B2%E0%B8%A1/%E0%B8%9A%E0%B8%B3%E0%B8%A3%E0%B8%B8%E0%B8%87%E0%B8%9C%E0%B8%B4%E0%B8%A7/%E0%B9%80%E0%B8%8B%E0%B8%A3%E0%B8%B1%E0%B9%88%E0%B8%A1/c/serum",
            "มอยเจอร์ไรเซอร์": "https://www.amway.co.th/Categories/%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%87%E0%B8%B2%E0%B8%A1/%E0%B8%9A%E0%B8%B3%E0%B8%A3%E0%B8%B8%E0%B8%87%E0%B8%9C%E0%B8%B4%E0%B8%A7/%E0%B8%A1%E0%B8%AD%E0%B8%A2%E0%B8%AA%E0%B9%8C%E0%B9%80%E0%B8%88%E0%B8%AD%E0%B9%84%E0%B8%A3%E0%B9%80%E0%B8%8B%E0%B8%AD%E0%B8%A3%E0%B9%8C/c/moisturizer",
           
            "ใบหน้า": "https://www.amway.co.th/Categories/%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%87%E0%B8%B2%E0%B8%A1/%E0%B9%80%E0%B8%A1%E0%B8%81%E0%B8%AD%E0%B8%B1%E0%B8%9B/%E0%B9%83%E0%B8%9A%E0%B8%AB%E0%B8%99%E0%B9%89%E0%B8%B2/c/face",
            "ดวงตา": "https://www.amway.co.th/Categories/%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%87%E0%B8%B2%E0%B8%A1/%E0%B9%80%E0%B8%A1%E0%B8%81%E0%B8%AD%E0%B8%B1%E0%B8%9B/%E0%B8%94%E0%B8%A7%E0%B8%87%E0%B8%95%E0%B8%B2%E0%B9%81%E0%B8%A5%E0%B8%B0%E0%B9%81%E0%B8%81%E0%B9%89%E0%B8%A1/c/eyes",
            "ปาก": "https://www.amway.co.th/Categories/%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%87%E0%B8%B2%E0%B8%A1/%E0%B9%80%E0%B8%A1%E0%B8%81%E0%B8%AD%E0%B8%B1%E0%B8%9B/%E0%B8%A3%E0%B8%B4%E0%B8%A1%E0%B8%9D%E0%B8%B5%E0%B8%9B%E0%B8%B2%E0%B8%81/c/lips",
            "อุปกรณ์ดูแลผิวและแต่งหน้า": "https://www.amway.co.th/Categories/%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%87%E0%B8%B2%E0%B8%A1/%E0%B9%80%E0%B8%A1%E0%B8%81%E0%B8%AD%E0%B8%B1%E0%B8%9B/%E0%B8%AD%E0%B8%B8%E0%B8%9B%E0%B8%81%E0%B8%A3%E0%B8%93%E0%B9%8C%E0%B8%94%E0%B8%B9%E0%B9%81%E0%B8%A5%E0%B8%9C%E0%B8%B4%E0%B8%A7%E0%B9%81%E0%B8%A5%E0%B8%B0%E0%B9%81%E0%B8%95%E0%B9%88%E0%B8%87%E0%B8%AB%E0%B8%99%E0%B9%89%E0%B8%B2/c/beautyaccessories",
           
            "ครีมกันแดด": "https://www.amway.co.th/search/?text=%E0%B8%84%E0%B8%A3%E0%B8%B5%E0%B8%A1%E0%B8%81%E0%B8%B1%E0%B8%99%E0%B9%81%E0%B8%94%E0%B8%94",
            "คุชชั่น": "https://www.amway.co.th/search/?text=%E0%B8%84%E0%B8%B8%E0%B8%8A%E0%B8%8A%E0%B8%B1%E0%B9%88%E0%B8%99",
            "คอนซีลเลอร์": "https://www.amway.co.th/search/?text=%E0%B8%84%E0%B8%AD%E0%B8%99%E0%B8%8B%E0%B8%B5%E0%B8%A5%E0%B9%80%E0%B8%A5%E0%B8%AD%E0%B8%A3%E0%B9%8C",
            "รองพื้น": "https://www.amway.co.th/search/?text=%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%9E%E0%B8%B7%E0%B9%89%E0%B8%99",
            "แป้งฝุ่น": "https://www.amway.co.th/search/?text=%E0%B9%81%E0%B8%9B%E0%B9%89%E0%B8%87%E0%B8%9D%E0%B8%B8%E0%B9%88%E0%B8%99",
            "แป้งแข็ง": "https://www.amway.co.th/search?text=%E0%B9%81%E0%B8%9B%E0%B9%89%E0%B8%87%E0%B9%81%E0%B8%82%E0%B9%87%E0%B8%87",
            
            "พาเลท": "https://www.amway.co.th/search/?text=%E0%B8%9E%E0%B8%B2%E0%B9%80%E0%B8%A5%E0%B8%97",
            "ดินสอเขียน": "https://www.amway.co.th/search/?text=%E0%B8%94%E0%B8%B4%E0%B8%99%E0%B8%AA%E0%B8%AD%E0%B9%80%E0%B8%82%E0%B8%B5%E0%B8%A2%E0%B8%99",
            "มาสคาร่า": "https://www.amway.co.th/search/?text=%E0%B8%A1%E0%B8%B2%E0%B8%AA%E0%B8%84%E0%B8%B2%E0%B8%A3%E0%B9%88%E0%B8%B2",
            "ลิปสติก": "https://www.amway.co.th/Categories/%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%87%E0%B8%B2%E0%B8%A1/%E0%B9%80%E0%B8%A1%E0%B8%81%E0%B8%AD%E0%B8%B1%E0%B8%9B/%E0%B8%A3%E0%B8%B4%E0%B8%A1%E0%B8%9D%E0%B8%B5%E0%B8%9B%E0%B8%B2%E0%B8%81/c/lips",
            
            "แนะนำผลิตลดสิว": "https://www.amway.co.th/Categories/%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%87%E0%B8%B2%E0%B8%A1/%E0%B8%9B%E0%B8%B1%E0%B8%8D%E0%B8%AB%E0%B8%B2%E0%B8%9C%E0%B8%B4%E0%B8%A7/%E0%B8%9C%E0%B8%B4%E0%B8%A7%E0%B9%82%E0%B8%81%E0%B8%A5%E0%B8%A7%E0%B9%8C%E0%B9%83%E0%B8%AA%E0%B9%84%E0%B8%81%E0%B8%A5%E0%B8%AA%E0%B8%B4%E0%B8%A7/c/studio-skin",
            "แนะนำผิวกระจ่างใส":"https://www.amway.co.th/Categories/%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%87%E0%B8%B2%E0%B8%A1/%E0%B8%9B%E0%B8%B1%E0%B8%8D%E0%B8%AB%E0%B8%B2%E0%B8%9C%E0%B8%B4%E0%B8%A7/%E0%B8%9C%E0%B8%B4%E0%B8%A7%E0%B8%81%E0%B8%A3%E0%B8%B0%E0%B8%88%E0%B9%88%E0%B8%B2%E0%B8%87%E0%B9%83%E0%B8%AA/c/artistryidealradiance"
        }

        if msg in url_map:
            url = url_map[msg]
            try:
                products = scrape_amway(url)
                if products:
                    send_flex_message(reply_token, products)
                    upsert_user_and_log_conversation(user_id, original_msg, f"Displaying products for {msg}")
                else:
                    line_bot_api.reply_message(reply_token, TextSendMessage(text="ไม่พบสินค้าในหมวดหมู่นี้"))
            except Exception as e:
                line_bot_api.reply_message(reply_token, TextSendMessage(text="ขออภัย มีปัญหาในการโหลดสินค้า โปรดลองใหม่อีกครั้ง"))
                print(f"Error fetching products: {e}")
            return 'OK'

        # ตรวจสอบ intent ที่ไม่รู้จัก
        else:
            line_bot_api.reply_message(reply_token, TextSendMessage(text="หมวดหมู่ที่ไม่รู้จัก โปรดลองอีกครั้ง"))
            return 'OK'

    except Exception as e:
        print(f"Error processing the LINE event: {e}")
        return 'Error', 500
    
app = Flask(__name__)
    
if __name__ == '__main__':
    app.run(port=5000, debug=True)
