# coding: utf-8

import numpy as np
from flask import Flask, request
from flask_cors import CORS
import sys
import pymongo as pm
import json
import pandas as pd
import urllib
import subprocess
import os
import warnings
from applicationinsights.exceptions import enable
from applicationinsights import TelemetryClient
from applicationinsights.flask.ext import AppInsights

if "APPINSIGHTS_INSTRUMENTATIONKEY" in os.environ:
    tc = TelemetryClient(os.environ["APPINSIGHTS_INSTRUMENTATIONKEY"])


class Recipe_recommender:

    def __init__(self):
        self.connect_db()
        self.load_recommend_data()


    #mongodb connections
    def connect_db(self):
        self.client = pm.MongoClient(os.environ["CONNECTION_RECIPES"])
        #self.rec_db = self.client.recipes
        self.rec_db = self.client['recipes_v2']
        self.rec_col = self.rec_db.recipes
        self.rec_decorr_col = self.rec_db.recipes_decorr
        self.lab_dict = self.rec_db.labels.find_one(projection={"_id":0})
        self.lus_col = self.rec_db.label_user_scores
        self.rec_sc_col = self.rec_db.recipe_scores
        self.users_all = self.lus_col.distinct("user_id")
        self.recipes_all = self.rec_col.distinct("recipeId")
        self.labels_all = self.rec_col.distinct("labels")       
        self.tag_products_mapping = self.rec_db.tag_products
        self.label_bias_col = self.rec_db.label_bias
        self.ingredientName_tag_col = self.rec_db.ingredientName_tag


    def load_recommend_data(self):
        #OLD
        label_recipe = pd.DataFrame([[i["recipeId"], l] for i in self.rec_col.find({"labels" : {"$exists": True}}, {"recipeId":1, "labels":1}) for l in i["labels"]], columns=["rec_id", "label"])
        label_recipe["val"] = 1
        label_recipe = label_recipe.pivot(index="label", columns="rec_id", values="val")
        label_recipe.fillna(0, inplace=True)
        
        
        U,S,V = np.linalg.svd(label_recipe.T.cov())
        epsilon = 1e-5
        zca = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
        label_recipe.values[:] = zca.dot(label_recipe)
        #OLD

        # label_recipe = pd.DataFrame([[i["label"], i["rec_id"], i["value"]] for i in self.rec_decorr_col.find()], columns=["label", "rec_id", "val"])
        # label_recipe = label_recipe.pivot(index="label", columns="rec_id", values="val")
        # label_recipe = label_recipe.astype("float32")

        #per label: mean number of labels in recipes the label occurs in
        #needed to normalize label preferences, otherwise bias towards labels which typically 
        #occur together with few other labels
        # OLD-commented
        lab_means = (label_recipe * label_recipe.sum()).sum(axis=1) / label_recipe.sum(axis=1)
        label_recipe /= label_recipe.sum(axis=0)
        # print("lab_means:\n", lab_means)
        # print("label_recipe: \n", label_recipe)
        # OLD-commented
        
        #normalization term for recipe popularity bias
        ind, sc = np.asarray([[i["recipeId"], i["score"]] for i in self.rec_sc_col.find()]).T
        recipe_sum = pd.DataFrame(sc, index=ind, columns=["rec_score"])

        ind, val = np.asarray([[i["label"], i["value"]] for i in self.label_bias_col.find()]).T
        lab_bias = pd.Series(val, index=ind, dtype="float32")
        
        print("loaded")

        #return user_recipe, recipe_sum
        self.label_recipe = label_recipe
        self.recipe_sum = recipe_sum
        #self.lab_bias = lab_bias
        self.lab_bias = lab_means

    def tags_to_ingredientNames(self, tags):
        ingredients = []
        for i in tags:
            found = [j["ingredientName"] for j in self.ingredientName_tag_col.find({"tag" : i})]
            ingredients.extend(found)
            print("tag: ", i, "\ningredients: ", list(found))
        return ingredients

    #parameters:
    #top: how many recipes are selected
    #customer: customer id
    #time: maximum work time for recipes
    #difficulty: difficulty for recipes (exclusive)
    #random_size: factor > 1 with which the top is multiplied. Actual top is then sampled proportionally to scores. 1 for no sampling
    #random_heterogenity: higher value increases probability for sampling high affinity recipes 
    #products: recommend recipes based on match to current shopping cart items instead of affinity scores
    #labels: all recipes must contain these labels
    def recommend(self, top=10, customer=None, time=None, difficulty=None,
        random_size=1, random_homogenity=1,labels=None,seed=None, products=None, image_url=None):
        
        #setting default values, since request gives null
        if seed != None:
            seed = int(seed)
            np.random.seed(seed)

        if random_homogenity == None:
            random_homogenity = 1
        else:
            random_homogenity = float(random_homogenity) 

        if random_size == None:
            random_size=1

        if top == None:
            top = 10
        else:
            top = int(top)

        top_q = int(top*float(random_size))        
        
        #mongodb query
        qry_dict = {}
        
        if labels and len(labels)>0:
            labels = json.loads(urllib.parse.unquote(labels))
            if not set(labels) <= set(self.labels_all):
                warnings.warn("one of the requested labels does not exist in any recipe", Warning)
            labels = list(set(labels) & set(self.labels_all))
            if len(labels) > 0:
                qry_dict["labels"] = {"$all":labels}

        if difficulty:
            qry_dict["difficulty"] = {"$lte": int(difficulty)}

        if time:
            qry_dict["time"] = {"$lte": int(time)}

        #TODO: 
        # if image_url:
        #     qry_dict["image_url"] = image_url

        recs = [i["recipeId"] for i in self.rec_col.find(qry_dict, {"_id":0, "recipeId":1})]
        print("QUERY DICT: ", qry_dict, "\nFound recipes: ", len(recs))

        #TODO: selectively recommend labels to avoid dominant label

        #get top recipes
        if len(recs) < top_q:
            top_q = len(recs)
        if len(recs) < top:
            top = len(recs)

        print("Customer-ID:", customer, "\nTotal customers:", len(self.users_all))

        if customer:
            customer = urllib.parse.unquote(customer)

        if customer and customer in self.users_all:
            #customer = int(customer)
            ind, sc = np.asarray([[i["label"], i["score"]] for i in self.lus_col.find_one({"user_id":customer})["scores"]]).T
            label_customer = pd.DataFrame(sc, index=ind, dtype="float32", columns=["rec_score"])
            #label_customer **= random_homogenity
            #label_customer.sort_values(ascending=False, inplace=True)
            #label_customer[10:] = 0

            # print("Label-customer:\n", label_customer)

            label_customer["rec_score"] *= self.lab_bias

            user_recipe = label_customer.T.dot(self.label_recipe).T

            top_recs = user_recipe.loc[recs]

            #print("User-recipe:\n ", user_recipe)
            if top_recs["rec_score"].sum() <= 0:
                top_recs = self.recipe_sum.loc[recs]
        else:
            top_recs = self.recipe_sum.loc[recs]

        #get number of times the selected products occur in recipes
        #two matches for product to reduce unwind
        if products:
            products = json.loads(urllib.parse.unquote(products))
            tags = [i["tag"] for i in self.tag_products_mapping.find({"products":{"$in":products}}, {"tag":1})]
            ingredients = self.tags_to_ingredientNames(tags)

            prod_subs = self.rec_col.aggregate([
                    {"$match":qry_dict},
                    {"$project":{"recipeId":1,"ingredientGroups.ingredientGroupIngredients.ingredient":1}},
                    {"$unwind":"$ingredientGroups"},
                    {"$unwind":"$ingredientGroups.ingredientGroupIngredients"},
                    {"$group":{"_id": "$recipeId","ings": {"$addToSet":"$ingredientGroups.ingredientGroupIngredients.ingredient"}}}, 
                    {"$unwind":"$ings"},
                    {"$match":{"ings":{"$in":ingredients}}},
                    {"$group":{"_id":"$_id", "count":{"$sum":1}}},
                    {"$sort":{"count":-1}},
                    {"$limit":top}
                    ])

            prod_subs = list(prod_subs)
            top_recs.loc[list(map(str,[i["_id"] for i in prod_subs])), "prod_score"] = [i["count"] for i in prod_subs]   
        else:
            top_recs["prod_score"] = 0

        top_recs = top_recs.sort_values(ascending=False, by=["prod_score", "rec_score"])[:top_q]

        #sample with probabilities proportional to affinities
        
        #recs[1]+=1
        #recs[1] /= recs[1].mean()

        print("Top Recommendations:\n", top_recs)

        if len(top_recs) == 0:
            return []
        
        top_recs["rec_score"] = top_recs["rec_score"].astype("float64")
        top_recs["rec_score"] **= float(random_homogenity)
        if top_recs["rec_score"].sum() <= 0:
            top_recs = self.recipe_sum.loc[recs].sort_values(ascending=False)[:top_q]
        top_recs["rec_score"] /= np.sum(top_recs["rec_score"])

        print("Recommendation score:\n",list(top_recs["rec_score"]))

        recs_out = np.random.choice(list(top_recs.index), top, replace=False, p=list(top_recs["rec_score"]))

        return list(map(str, recs_out))
    

rr = Recipe_recommender()

app = Flask(__name__)

@app.route('/labelstructure')
def get_labelstructure():
    token=request.headers.get("token")
    if not token == os.environ["WHIZCART_TOKEN"]:
        return "wrong authentification token"
    return json.dumps(rr.lab_dict)

@app.route('/recommend', methods=['GET'])
def get_recommend():
    try:
        user_id = request.args.get('id')
        difficulty=request.args.get('difficulty')
        top=request.args.get('top')
        time=request.args.get('time')
        random_size=request.args.get('random_size')
        random_homogenity=request.args.get('random_homogenity')
        labels=request.args.get("labels")
        seed=request.args.get("seed")  
        products=request.args.get("products")  
        token=request.headers.get("token")
        # store=request.args.get("store")

        if not token == os.environ["WHIZCART_TOKEN"]:
            return "wrong authentification token"

        rec_ids = rr.recommend(customer=user_id, difficulty=difficulty,top=top,time=time, random_size=random_size, 
            random_homogenity=random_homogenity, labels=labels, seed=seed, products=products, image_url={"$ne":None})
        print("Recipe ids: \n", rec_ids, "size: ", len(rec_ids))
        
        if top and len(rec_ids) < int(top):
            imageless = rr.recommend(customer=user_id, difficulty=difficulty,top=int(top)-len(rec_ids),time=time, 
            random_size=random_size, random_homogenity=random_homogenity, labels=labels, seed=seed, products=products, image_url={"$eq":None})
            rec_ids.extend(imageless)
        
        out = [i for i in rr.rec_col.find({"recipeId": {"$in":rec_ids}}, {"_id":0})]
        
        return json.dumps(out, indent=4).replace("\n", "")

    except Exception as ex:
        tc.track_exception()
        print("Exception: ", ex)


@app.route('/recipe', methods=['GET'])
def recipe():
    try:
        rec_id = request.args.get('id')
        token=request.headers.get("token")
        # store=request.args.get("store")

        if not token == os.environ["WHIZCART_TOKEN"]:
            return "wrong authentification token"

        if not rec_id:
            return "please specify a recipe id as id="

        rec = rr.rec_col.find_one({"recipeId":rec_id}, {"_id":0})

        if not rec:
            return "no recipe exists with the specified id"


        return json.dumps(rec, indent=4).replace("\n", "")

    except Exception as ex:
        tc.track_exception()
        print("Exception: ", ex)

@app.route('/recalculate')
def recalculate():
    token=request.headers.get("token")
    if not token == os.environ["WHIZCART_TOKEN"]:
        return "wrong authentification token"
    
    print("Recalculating...")
    print("Dropping collections...")

    #drop some collections before recalc.
    rr.rec_db.drop_collection(rr.rec_col)  
    rr.rec_db.drop_collection(rr.rec_db.recipes_raw_edeka)
    rr.rec_db.drop_collection(rr.tag_products_mapping)
    rr.rec_db.drop_collection(rr.rec_sc_col)
    rr.rec_db.drop_collection(rr.lus_col)

    #import new data from Edeka API.
    create_recipes_sp = ["python", "src/Create_recipes.py"]
    proc = subprocess.Popen(create_recipes_sp)
    proc.wait()

    #subprocess.call(["python", "Calculate_affinities_azure.py"]) 
    calculate_affinities_sp = ["python", "src/Calculate_affinities.py"]
    
    customers=request.args.get("customers")
    if customers:
        customers = json.loads(urllib.parse.unquote(customers))
        calculate_affinities_sp.extend(customers)
    
    proc = subprocess.Popen(calculate_affinities_sp)
    proc.wait()
    
    return "True"

@app.route('/reinit')
def reinit():
    token=request.headers.get("token")
    if not token == os.environ["WHIZCART_TOKEN"]:
        return "wrong authentification token"
    
    global rr
    rr = Recipe_recommender()

    return "True"

@app.route('/get_mapping')
def get_mapping():
    token=request.headers.get("token")
    if not token == os.environ["WHIZCART_TOKEN"]:
        return "wrong authentification token"

    ingredient=request.args.get("ingredient")
    if not ingredient:
            return "please specify an ingredient in your args as ingredient="
            
    products = rr.tag_products_mapping.find_one({"tag":ingredient})["products"]

    return json.dumps(products)


@app.route('/update_mapping')
def update_mapping():
    token=request.headers.get("token")
    if not token == os.environ["WHIZCART_TOKEN"]:
        return "wrong authentification token"

    p = request.args.get("products")
    print("prod:", p)
    print("ingredient:", request.args.get("ingredient"))
    print("json ", json.loads(p))

    ingredient=request.args.get("ingredient")
    products=json.loads(urllib.parse.unquote(request.args.get("products")))
    if ingredient and products:
        rr.tag_products_mapping.update_one({"tag":ingredient}, {"$set": {"products":products}}, True)

    return "True"

CORS(app)

if "APPINSIGHTS_INSTRUMENTATIONKEY" in os.environ:
    app.config['APPINSIGHTS_INSTRUMENTATIONKEY'] = os.environ["APPINSIGHTS_INSTRUMENTATIONKEY"]
    appinsights = AppInsights(app)
    enable(os.environ["APPINSIGHTS_INSTRUMENTATIONKEY"])

if __name__ == '__main__':
    app.run(debug=True)