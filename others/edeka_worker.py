import requests, json, os
import pymongo as pm

token_url = os.environ["TOKEN_URL"]
edeka_api_url = os.environ["EDEKA_API_URL"]

def get_next_page(api_call_response):
    y = json.loads(api_call_response.text)
    next_page_url = y["_links"]["next"]["href"].split('{')[0]
    if not next_page_url:
        return 
    api_next_call_url = edeka_api_url.split('?')[0] + '?' + next_page_url.split('?')[1]
    print (api_next_call_url)
    return api_next_call_url

def import_data(data, collection_name):
    recipe_client = pm.MongoClient(os.environ["CONNECTION_RECIPES"])
    #db_recipes = recipe_client['recipes']
    db_recipes = recipe_client['recipes_v2']
    coll = db_recipes[collection_name]
    coll.insert_many(data)

def get_all_recipes_raw(client_id, client_secret):
    #step A, B - single call with client credentials as the basic auth header - will return access_token
    data = {'grant_type': 'client_credentials'}
    access_token_response = requests.post(token_url, data=data, auth=(client_id, client_secret))

    tokens = json.loads(access_token_response.text)
    #print ("access token: " + tokens['access_token'])

    #step B - with the returned access_token we can make as many calls as we want
    api_call_headers = {'Authorization': 'Bearer ' + tokens['access_token']}
    api_call_response = requests.get(edeka_api_url, headers=api_call_headers)

    y = json.loads(api_call_response.text)
    recipes_list = y["recipes"]

    print ("Current batch size: ", y["currentSize"])

    api_next_call_url = get_next_page(api_call_response)
    while api_next_call_url:
        api_call_response = requests.get(api_next_call_url, headers=api_call_headers)
        y = json.loads(api_call_response.text)
        if 'recipes' not in y:
            break
        recipes_list.extend(y["recipes"])
        api_next_call_url = get_next_page(api_call_response)
        
    print (len(recipes_list), " recipes found.")
    return recipes_list





