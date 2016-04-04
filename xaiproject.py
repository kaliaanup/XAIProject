'''Created on Apr 4, 2016
@author: Anup Kalia
'''
import pymongo

#------------CONNECT TO MONGODB-------------------------*/
client = pymongo.MongoClient("localhost", 27017)
#db name is dbxai
db = client.dbxai
print(db.name)

#-------------ACCESS EACH ROW---------------------------*/
#collection name is xai
cursor = db.xai.find({"dataPoint.label"})
print(cursor)