from pymongo import MongoClient
import datetime

client = MongoClient("mongodb+srv://seanbartolome7slm:cap2419it@busmateph.vfi4r.mongodb.net/?tlsAllowInvalidCertificates=true")
db = client["BusMatePH"]
collection = db["capacity"]

collection.update_one(
    {"busID": 4},
    {"$set": {"capacity": 999, "date": datetime.datetime.now()}},
    upsert=True
)

print("Test insert successful!")
